"""Runtime consumption of one immutable ``deploy/default.yaml`` composition.

The builder reads no SysID or training state.  It authenticates frozen neural
archives, maps explicit joint names to one-DOF Newton model axes, constructs
explicit PD actuators, and delegates schema-v3 neural execution through a
mapping-aware factory.  Newton, Warp, and Torch remain lazy imports.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

import numpy as np

from anchor_bench.core.hashing import contained_path, require_relative_posix_path, sha256_file
from anchor_bench.core.profiles import load_yaml_mapping
from anchor_bench.deploy.schema import (
    FrozenDeployState,
    FrozenNeuralActuator,
    FrozenNeuralMapping,
    FrozenPDActuator,
)
from anchor_bench.training.archive import read_runtime_metadata_archive
from anchor_bench.training.metadata import RuntimeGRUMetadata, RuntimeMapping, StatValue


@dataclass(frozen=True)
class NewtonActuatorBindings:
    """Lazy Newton/Warp constructors used by explicit PD runtime creation."""

    warp: Any
    actuator_type: Callable[..., Any]
    controller_pd_type: Callable[..., Any]
    clamping_max_effort_type: Callable[..., Any]
    joint_target_mode_type: Any
    joint_dof_properties_flag: Any
    clear_owned_efforts: Callable[[Any, Any, Any], None]


@dataclass(frozen=True)
class MappedJointAxes:
    """World-major Newton indices for one ordered physical joint axis."""

    joints: tuple[str, ...]
    worlds: tuple[int, ...]
    position_indices: tuple[tuple[int, ...], ...]
    velocity_indices: tuple[tuple[int, ...], ...]
    target_position_indices: tuple[tuple[int, ...], ...]

    def flatten_positions(self) -> tuple[int, ...]:
        """Return world-major state-position indices."""

        return tuple(index for row in self.position_indices for index in row)

    def flatten_velocities(self) -> tuple[int, ...]:
        """Return world-major state-velocity and effort indices."""

        return tuple(index for row in self.velocity_indices for index in row)

    def flatten_target_positions(self) -> tuple[int, ...]:
        """Return world-major position-target indices for the model layout."""

        return tuple(index for row in self.target_position_indices for index in row)


@dataclass(frozen=True)
class MappedNeuralRequest:
    """Authenticated schema-v3 contract for one physical neural mapping.

    A factory must implement the network path declared by ``actuator.target``
    and ``mapping.clamping``. External feedforward and delay must not be added.
    Full-torque factories supply the clamped total network torque. Residual
    factories supply only the clamped network residual: frozen PD gains and
    dry/viscous friction are installed in Newton and remain solver-owned.
    """

    model: Any
    artifact_path: Path
    metadata: RuntimeGRUMetadata
    actuator: FrozenNeuralActuator
    mapping: FrozenNeuralMapping
    input_axes: MappedJointAxes
    output_axes: MappedJointAxes
    control_dt_s: float
    external_feedforward_allowed: bool = False
    external_delay_steps: int = 0

    @property
    def solver_gains_disabled(self) -> bool:
        """Return whether outputs use direct effort instead of frozen solver PD."""

        return self.actuator.target == "torque"

    @property
    def factory_output_semantics(self) -> str:
        """Return the exact effort quantity the mapped factory must scatter-add."""

        return (
            "clamped_residual"
            if self.actuator.target == "torque_residual"
            else "clamped_total_torque"
        )

    @property
    def requires_dynamic_bias(self) -> bool:
        """Return whether this mapping consumes MuJoCo dynamic bias."""

        return any(feature.name == "dynamic_bias" for feature in self.metadata.input_features)


class MappedNeuralFactory(Protocol):
    """Factory for one mapping-aware schema-v3 recurrent runtime actuator."""

    def __call__(self, request: MappedNeuralRequest) -> Any:
        """Return one stateful actuator implementing the request exactly."""


@dataclass(frozen=True)
class _JointInstance:
    """One single-coordinate, single-DOF model joint occurrence."""

    world: int
    position_index: int
    velocity_index: int
    target_position_index: int


@dataclass(frozen=True)
class _VerifiedNeural:
    """Authenticated neural archive reused by all selected mappings."""

    path: Path
    metadata: RuntimeGRUMetadata


@dataclass(frozen=True)
class RuntimeActuatorEntry:
    """One runtime actuator and its frozen ownership identity."""

    name: str
    kind: str
    mapping_index: int | None
    worlds: tuple[int, ...]
    actuator: Any


@dataclass
class DeploymentRuntimeState:
    """Double-buffered state for independent recurrent mapping actuators."""

    entry_keys: tuple[tuple[str, int | None], ...]
    worlds: tuple[int, ...]
    current: list[Any | None]
    next: list[Any | None]

    def reset(self, mask: Any | None = None) -> None:
        """Reset both recurrent buffers for all or selected environments."""

        if mask is not None and len(mask) != len(self.worlds):
            raise ValueError(
                f"deployment reset mask must contain {len(self.worlds)} environment values"
            )
        for value in (*self.current, *self.next):
            if value is None:
                continue
            reset = getattr(value, "reset", None)
            if not callable(reset):
                raise TypeError("runtime actuator state must expose reset(mask)")
            reset(mask)

    def swap(self) -> None:
        """Promote next recurrent buffers after one successful control step."""

        self.current, self.next = self.next, self.current


class DeploymentRuntime:
    """Executable frozen actuator composition for one Newton model."""

    def __init__(
        self,
        state: FrozenDeployState,
        entries: Sequence[RuntimeActuatorEntry],
        *,
        worlds: tuple[int, ...],
        solver: Any | None,
        requires_dynamic_bias: bool,
        owned_effort_indices: Any,
        clear_owned_efforts: Callable[[Any, Any, Any], None],
        device: Any,
    ) -> None:
        """Store validated runtime entries without rereading mutable state."""

        self.frozen_state = state
        self.entries = tuple(entries)
        self.worlds = worlds
        self.control_dt_s = 1.0 / state.simulation.control_freq
        self._solver = solver
        self._requires_dynamic_bias = requires_dynamic_bias
        self._dynamic_bias: Any | None = None
        self._has_recurrent = any(entry.kind == "neural" for entry in self.entries)
        self._owned_effort_indices = owned_effort_indices
        self._clear_owned_efforts = clear_owned_efforts
        self._device = device

    def state(self) -> DeploymentRuntimeState:
        """Allocate independent current/next state for every neural mapping."""

        current: list[Any | None] = []
        next_values: list[Any | None] = []
        for entry in self.entries:
            state_factory = getattr(entry.actuator, "state", None)
            if not callable(state_factory):
                raise TypeError(f"runtime actuator {entry.name!r} must expose state()")
            current_value = state_factory()
            next_value = state_factory()
            if entry.kind == "neural" and (current_value is None or next_value is None):
                raise TypeError(
                    f"mapped neural actuator {entry.name!r} index {entry.mapping_index} "
                    "must allocate recurrent state"
                )
            current.append(current_value)
            next_values.append(next_value)
        return DeploymentRuntimeState(
            entry_keys=tuple((entry.name, entry.mapping_index) for entry in self.entries),
            worlds=self.worlds,
            current=current,
            next=next_values,
        )

    def step(
        self,
        sim_state: Any,
        sim_control: Any,
        runtime_state: DeploymentRuntimeState | None = None,
        *,
        dt: float | None = None,
    ) -> None:
        """Replace owned efforts for one step while preserving omitted DOFs."""

        step_dt = self.control_dt_s if dt is None else float(dt)
        if not math.isfinite(step_dt) or not math.isclose(
            step_dt,
            self.control_dt_s,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"deployment timestep {step_dt} does not match frozen control timestep "
                f"{self.control_dt_s}"
            )
        if self._has_recurrent and runtime_state is None:
            raise ValueError("neural deployment requires DeploymentRuntime.state()")
        if runtime_state is not None:
            expected = tuple((entry.name, entry.mapping_index) for entry in self.entries)
            if runtime_state.entry_keys != expected or runtime_state.worlds != self.worlds:
                raise ValueError("deployment runtime state belongs to another composition")

        joint_f = getattr(sim_control, "joint_f", None)
        if joint_f is None:
            raise ValueError("deployment sim_control must expose joint_f")
        self._clear_owned_efforts(
            joint_f,
            self._owned_effort_indices,
            self._device,
        )

        dynamic_bias = None
        if self._requires_dynamic_bias:
            dynamic_bias = self._solver.eval_dynamic_bias(
                sim_state,
                out=self._dynamic_bias,
            )
            self._dynamic_bias = dynamic_bias

        for index, entry in enumerate(self.entries):
            current = runtime_state.current[index] if runtime_state is not None else None
            next_value = runtime_state.next[index] if runtime_state is not None else None
            entry.actuator.step(
                sim_state,
                sim_control,
                current,
                next_value,
                dt=step_dt,
                dynamic_bias=dynamic_bias,
            )
        if runtime_state is not None:
            runtime_state.swap()


def _load_newton_bindings() -> NewtonActuatorBindings:
    """Import the pinned Newton actuator surface only for real construction."""

    import warp as wp
    from newton import JointTargetMode, ModelFlags
    from newton.actuators import Actuator, ClampingMaxEffort, ControllerPD

    @wp.kernel
    def clear_owned_efforts_kernel(
        efforts: wp.array[float],
        indices: wp.array[wp.uint32],
    ):
        """Clear one composition-owned effort per thread."""

        efforts[indices[wp.tid()]] = 0.0

    def clear_owned_efforts(efforts: Any, indices: Any, device: Any) -> None:
        """Launch selected-DOF clearing on the model device."""

        wp.launch(
            kernel=clear_owned_efforts_kernel,
            dim=len(indices),
            inputs=[efforts, indices],
            device=device,
        )

    return NewtonActuatorBindings(
        warp=wp,
        actuator_type=Actuator,
        controller_pd_type=ControllerPD,
        clamping_max_effort_type=ClampingMaxEffort,
        joint_target_mode_type=JointTargetMode,
        joint_dof_properties_flag=ModelFlags.JOINT_DOF_PROPERTIES,
        clear_owned_efforts=clear_owned_efforts,
    )


def load_frozen_deploy_state(path: str | Path) -> FrozenDeployState:
    """Load one strict regular ``deploy/default.yaml`` without upstream reads."""

    source = Path(path)
    if source.name != "default.yaml" or source.parent.name != "deploy":
        raise ValueError("deployment runtime must consume deploy/default.yaml")
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"deploy/default.yaml must be a regular file: {source}")
    return FrozenDeployState.from_dict(load_yaml_mapping(source), source)


def _stat_values(value: StatValue, width: int, *, label: str) -> tuple[float, ...]:
    """Expand one schema-v3 scalar/vector value to its declared width."""

    values = value if isinstance(value, tuple) else (value,)
    if len(values) != width:
        raise ValueError(f"{label} must contain {width} values")
    return tuple(float(item) for item in values)


def _metadata_map(
    value: StatValue,
    joints: Sequence[str],
    *,
    label: str,
) -> dict[str, float]:
    """Convert one ordered metadata vector to a frozen joint-keyed map."""

    values = _stat_values(value, len(joints), label=label)
    return dict(zip(joints, values, strict=True))


def _mapping_document(mapping: RuntimeMapping) -> dict[str, Any]:
    """Return the physical portion of one schema-v3 mapping for comparison."""

    pd_baseline = None
    if mapping.pd_baseline is not None:
        pd_baseline = {
            "kp": _metadata_map(
                mapping.pd_baseline.kp,
                mapping.input_joints,
                label="pd_baseline.kp",
            ),
            "kd": _metadata_map(
                mapping.pd_baseline.kd,
                mapping.input_joints,
                label="pd_baseline.kd",
            ),
            "velocity_target": mapping.pd_baseline.velocity_target,
        }
    friction_baseline = None
    if mapping.friction_baseline is not None:
        friction_baseline = {
            "dry": _metadata_map(
                mapping.friction_baseline.dry,
                mapping.output_joints,
                label="friction_baseline.dry",
            ),
            "viscous": _metadata_map(
                mapping.friction_baseline.viscous,
                mapping.output_joints,
                label="friction_baseline.viscous",
            ),
        }
    armature = None
    if mapping.armature is not None:
        armature = _metadata_map(
            mapping.armature.values,
            mapping.output_joints,
            label="armature.values",
        )
    return {
        "input_joints": mapping.input_joints,
        "output_joints": mapping.output_joints,
        "pd_baseline": pd_baseline,
        "friction_baseline": friction_baseline,
        "armature": armature,
        "effort_limits": _metadata_map(
            mapping.effort_limits,
            mapping.output_joints,
            label="effort_limits",
        ),
        "clamping": mapping.clamping,
    }


def _frozen_mapping_document(mapping: FrozenNeuralMapping) -> dict[str, Any]:
    """Return the frozen physical mapping fields in metadata comparison form."""

    return {
        "input_joints": mapping.input_joints,
        "output_joints": mapping.output_joints,
        "pd_baseline": (
            mapping.pd_baseline.model_dump(mode="python")
            if mapping.pd_baseline is not None
            else None
        ),
        "friction_baseline": (
            mapping.friction_baseline.model_dump(mode="python")
            if mapping.friction_baseline is not None
            else None
        ),
        "armature": dict(mapping.armature) if mapping.armature is not None else None,
        "effort_limits": dict(mapping.effort_limits),
        "clamping": mapping.clamping,
    }


def _verify_neural_artifact(
    root: Path,
    actuator: FrozenNeuralActuator,
) -> _VerifiedNeural:
    """Authenticate one archive and require exact frozen metadata agreement."""

    relative = require_relative_posix_path(actuator.artifact.path)
    raw_path = root.joinpath(*PurePosixPath(relative).parts)
    if raw_path.is_symlink() or not raw_path.is_file():
        raise ValueError(f"frozen neural artifact must be a regular file: {relative}")
    artifact_path = contained_path(root, relative, must_exist=True)
    actual_hash = sha256_file(artifact_path)
    if actual_hash != actuator.artifact.sha256:
        raise ValueError(
            f"neural artifact hash mismatch for {actuator.name!r}: "
            f"expected {actuator.artifact.sha256}, got {actual_hash}"
        )
    embedded = read_runtime_metadata_archive(artifact_path)
    if embedded.sha256 != actuator.artifact.metadata_sha256:
        raise ValueError(
            f"neural metadata hash mismatch for {actuator.name!r}: "
            f"expected {actuator.artifact.metadata_sha256}, got {embedded.sha256}"
        )
    metadata = embedded.metadata
    if metadata.target_columns != (actuator.target,):
        raise ValueError(f"neural target mismatch for {actuator.name!r}")
    if not math.isclose(
        metadata.sample_dt_s,
        actuator.sample_dt_s,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(f"neural sample_dt_s mismatch for {actuator.name!r}")
    for frozen_mapping in actuator.mappings:
        if frozen_mapping.index >= len(metadata.mappings):
            raise ValueError(
                f"neural mapping index {frozen_mapping.index} is absent from "
                f"{actuator.name!r} metadata"
            )
        metadata_mapping = metadata.mappings[frozen_mapping.index]
        if _mapping_document(metadata_mapping) != _frozen_mapping_document(frozen_mapping):
            raise ValueError(
                f"neural mapping index {frozen_mapping.index} no longer matches "
                f"frozen deployment {actuator.name!r}"
            )
    return _VerifiedNeural(artifact_path, metadata)


def _verify_frozen_source_identities(state: FrozenDeployState, root: Path) -> None:
    """Verify frozen robot and asset identities without reading phase state."""

    if state.source.robot is None:
        return
    if state.source.robot_identity is None:  # pragma: no cover - schema invariant
        raise RuntimeError("frozen deployment robot identity is missing")
    from anchor_bench.robots.fingerprints import (
        verify_robot_identity_fingerprint,
        verify_runtime_asset_fingerprint,
    )
    from anchor_bench.robots.registry import get_robot_from_root
    from anchor_bench.runtime.assets import resolve_runtime_asset

    bundle = get_robot_from_root(state.source.robot, root)
    verify_robot_identity_fingerprint(
        state.source.robot_identity,
        bundle,
        repository_root=root,
    )
    if state.source.asset_identity is not None:
        verify_runtime_asset_fingerprint(
            state.source.asset_identity,
            resolve_runtime_asset(bundle),
            runtime_usd=bundle.spec.runtime_usd,
        )


def _model_joint_instances(
    model: Any,
    requested_joints: Sequence[str],
) -> dict[str, dict[int, _JointInstance]]:
    """Resolve explicit joint names to one-DOF model occurrences by world."""

    labels = tuple(getattr(model, "joint_label", ()))
    if not labels:
        raise ValueError("Newton model must expose non-empty joint_label")
    q_start = np.asarray(model.joint_q_start.numpy(), dtype=np.int64)
    qd_start = np.asarray(model.joint_qd_start.numpy(), dtype=np.int64)
    worlds = np.asarray(model.joint_world.numpy(), dtype=np.int64)
    joint_count = len(labels)
    if q_start.shape != (joint_count + 1,) or qd_start.shape != (joint_count + 1,):
        raise ValueError("Newton model joint start arrays do not match joint_label")
    if worlds.shape != (joint_count,):
        raise ValueError("Newton model joint_world does not match joint_label")

    requested = set(requested_joints)
    result: dict[str, dict[int, _JointInstance]] = {joint: {} for joint in requested}
    coord_targets = bool(getattr(model, "use_coord_layout_targets", False))
    for joint_id, label in enumerate(labels):
        name = str(label).rsplit("/", maxsplit=1)[-1]
        if name not in requested:
            continue
        q_count = int(q_start[joint_id + 1] - q_start[joint_id])
        qd_count = int(qd_start[joint_id + 1] - qd_start[joint_id])
        if q_count != 1 or qd_count != 1:
            raise ValueError(
                f"deployment joint {name!r} must have one position coordinate and one DOF; "
                f"got {q_count} and {qd_count}"
            )
        world = int(worlds[joint_id])
        if world in result[name]:
            raise ValueError(
                f"Newton model contains multiple joints named {name!r} in world {world}"
            )
        position_index = int(q_start[joint_id])
        velocity_index = int(qd_start[joint_id])
        result[name][world] = _JointInstance(
            world=world,
            position_index=position_index,
            velocity_index=velocity_index,
            target_position_index=(position_index if coord_targets else velocity_index),
        )
    missing = sorted(joint for joint, instances in result.items() if not instances)
    if missing:
        raise ValueError("Newton model is missing deployment joints: " + ", ".join(missing))
    return result


def _joint_axes(
    joints: Sequence[str],
    instances: Mapping[str, Mapping[int, _JointInstance]],
) -> MappedJointAxes:
    """Build world-major indices and require one matching robot per world."""

    ordered = tuple(joints)
    world_sets = {joint: set(instances[joint]) for joint in ordered}
    expected = world_sets[ordered[0]]
    mismatched = [joint for joint, worlds in world_sets.items() if worlds != expected]
    if mismatched:
        raise ValueError(
            "deployment joints do not share identical model worlds: "
            + ", ".join(mismatched)
        )
    worlds = tuple(sorted(expected))
    position_rows: list[tuple[int, ...]] = []
    velocity_rows: list[tuple[int, ...]] = []
    target_rows: list[tuple[int, ...]] = []
    for world in worlds:
        row = tuple(instances[joint][world] for joint in ordered)
        position_rows.append(tuple(item.position_index for item in row))
        velocity_rows.append(tuple(item.velocity_index for item in row))
        target_rows.append(tuple(item.target_position_index for item in row))
    return MappedJointAxes(
        joints=ordered,
        worlds=worlds,
        position_indices=tuple(position_rows),
        velocity_indices=tuple(velocity_rows),
        target_position_indices=tuple(target_rows),
    )


def _warp_array(bindings: NewtonActuatorBindings, model: Any, values: Sequence[Any], dtype: Any) -> Any:
    """Create one model-device Warp array through injected bindings."""

    return bindings.warp.array(values, dtype=dtype, device=model.device)


def _build_pd_actuator(
    entry: FrozenPDActuator,
    axes: MappedJointAxes,
    model: Any,
    bindings: NewtonActuatorBindings,
) -> Any:
    """Construct explicit ControllerPD with a total-output effort clamp."""

    worlds = len(axes.worlds)
    kp = [entry.kp[joint] for _ in range(worlds) for joint in entry.joints]
    kd = [entry.kd[joint] for _ in range(worlds) for joint in entry.joints]
    limits = [entry.effort_limits[joint] for _ in range(worlds) for joint in entry.joints]
    wp = bindings.warp
    velocity_indices = _warp_array(
        bindings,
        model,
        axes.flatten_velocities(),
        wp.uint32,
    )
    position_indices = _warp_array(
        bindings,
        model,
        axes.flatten_positions(),
        wp.uint32,
    )
    target_position_indices = _warp_array(
        bindings,
        model,
        axes.flatten_target_positions(),
        wp.uint32,
    )
    controller = bindings.controller_pd_type(
        kp=_warp_array(bindings, model, kp, wp.float32),
        kd=_warp_array(bindings, model, kd, wp.float32),
    )
    clamp = bindings.clamping_max_effort_type(
        max_effort=_warp_array(bindings, model, limits, wp.float32)
    )
    return bindings.actuator_type(
        indices=velocity_indices,
        pos_indices=position_indices,
        target_pos_indices=target_position_indices,
        effort_indices=velocity_indices,
        controller=controller,
        clamping=[clamp],
        control_target_pos_attr="joint_target_q",
        control_target_vel_attr="joint_target_qd",
        control_feedforward_attr=None,
        control_output_attr="joint_f",
        control_computed_output_attr=None,
    )


def _claim_update(
    updates: dict[str, dict[int, float]],
    field: str,
    index: int,
    value: float,
) -> None:
    """Record one deterministic model mutation and reject conflicts."""

    previous = updates.setdefault(field, {}).get(index)
    if previous is not None and previous != value:
        raise ValueError(f"conflicting frozen deployment values for {field}[{index}]")
    updates[field][index] = float(value)


def _model_updates(
    state: FrozenDeployState,
    instances: Mapping[str, Mapping[int, _JointInstance]],
    target_mode_type: Any,
) -> dict[str, dict[int, float]]:
    """Resolve owned-DOF gains and physical parameters without touching omissions."""

    updates: dict[str, dict[int, float]] = {}
    residual_outputs = {
        joint
        for entry in state.composition.neural
        if entry.target == "torque_residual"
        for mapping in entry.mappings
        for joint in mapping.output_joints
    }
    for joint in state.ownership:
        if joint in residual_outputs:
            continue
        for instance in instances[joint].values():
            _claim_update(updates, "joint_target_ke", instance.velocity_index, 0.0)
            _claim_update(updates, "joint_target_kd", instance.velocity_index, 0.0)
            _claim_update(
                updates,
                "joint_target_mode",
                instance.velocity_index,
                int(target_mode_type.EFFORT),
            )
    for entry in state.composition.pd:
        for joint in entry.joints:
            for instance in instances[joint].values():
                dof = instance.velocity_index
                _claim_update(updates, "joint_friction", dof, entry.dry_friction[joint])
                _claim_update(updates, "joint_damping", dof, entry.viscous_friction[joint])
                if entry.armature is not None:
                    _claim_update(updates, "joint_armature", dof, entry.armature[joint])
    for entry in state.composition.neural:
        for mapping in entry.mappings:
            for joint in mapping.output_joints:
                for instance in instances[joint].values():
                    dof = instance.velocity_index
                    if entry.target == "torque_residual":
                        if mapping.pd_baseline is None or mapping.friction_baseline is None:
                            raise ValueError(
                                f"residual mapping {entry.name!r}/{mapping.index} "
                                "is missing its frozen baseline"
                            )
                        kp = float(mapping.pd_baseline.kp[joint])
                        kd = float(mapping.pd_baseline.kd[joint])
                        mode = target_mode_type.from_gains(
                            kp,
                            kd,
                            has_drive=True,
                        )
                        _claim_update(updates, "joint_target_ke", dof, kp)
                        _claim_update(updates, "joint_target_kd", dof, kd)
                        _claim_update(updates, "joint_target_mode", dof, int(mode))
                        _claim_update(
                            updates,
                            "joint_friction",
                            dof,
                            mapping.friction_baseline.dry[joint],
                        )
                        _claim_update(
                            updates,
                            "joint_damping",
                            dof,
                            mapping.friction_baseline.viscous[joint],
                        )
                    if mapping.armature is not None:
                        _claim_update(
                            updates,
                            "joint_armature",
                            dof,
                            mapping.armature[joint],
                        )
    return updates


def _apply_model_updates(
    model: Any,
    updates: Mapping[str, Mapping[int, float]],
) -> dict[str, np.ndarray]:
    """Apply owned-DOF updates and return exact arrays for rollback."""

    originals: dict[str, np.ndarray] = {}
    try:
        for field, values in updates.items():
            array = getattr(model, field, None)
            if array is None:
                raise ValueError(f"Newton model is missing required array {field}")
            original = np.asarray(array.numpy()).copy()
            updated = original.copy()
            for index, value in values.items():
                if index < 0 or index >= updated.shape[0]:
                    raise ValueError(f"{field} index {index} is outside the Newton model")
                updated[index] = value
            originals[field] = original
            array.assign(updated)
    except BaseException:
        _restore_model_arrays(model, originals)
        raise
    return originals


def _restore_model_arrays(model: Any, originals: Mapping[str, np.ndarray]) -> None:
    """Restore model arrays after a failed solver notification."""

    for field, values in originals.items():
        getattr(model, field).assign(values)


def _validate_runtime_actuator(actuator: Any, *, label: str, neural: bool) -> None:
    """Require the narrow Newton-like runtime surface from a factory product."""

    for method in ("state", "step"):
        if not callable(getattr(actuator, method, None)):
            raise TypeError(f"{label} must expose {method}()")
    if neural:
        stateful = getattr(actuator, "is_stateful", None)
        if not callable(stateful) or not stateful():
            raise TypeError(f"{label} must be stateful for recurrent reset semantics")


def _default_mapped_neural_factory(request: MappedNeuralRequest) -> Any:
    """Load Anchor's schema-v3 mapped GRU implementation only when needed."""

    from anchor_bench.deploy.neural_gru import build_mapped_gru_actuator

    return build_mapped_gru_actuator(request)


def _build_runtime_entries(
    state: FrozenDeployState,
    model: Any,
    instances: Mapping[str, Mapping[int, _JointInstance]],
    verified: Mapping[str, _VerifiedNeural],
    *,
    expected_dt: float,
    neural_factory: MappedNeuralFactory | None,
    bindings: NewtonActuatorBindings,
) -> tuple[list[RuntimeActuatorEntry], tuple[int, ...]]:
    """Construct all explicit actuators after model ownership is configured."""

    runtime_entries: list[RuntimeActuatorEntry] = []
    common_worlds: tuple[int, ...] | None = None
    for entry in state.composition.pd:
        axes = _joint_axes(entry.joints, instances)
        common_worlds = _require_common_worlds(common_worlds, axes.worlds)
        actuator = _build_pd_actuator(entry, axes, model, bindings)
        _validate_runtime_actuator(actuator, label=f"PD actuator {entry.name!r}", neural=False)
        runtime_entries.append(
            RuntimeActuatorEntry(entry.name, "pd", None, axes.worlds, actuator)
        )

    for entry in state.composition.neural:
        archive = verified[entry.artifact.path]
        for mapping in entry.mappings:
            input_axes = _joint_axes(mapping.input_joints, instances)
            output_axes = _joint_axes(mapping.output_joints, instances)
            if input_axes.worlds != output_axes.worlds:
                raise ValueError(
                    f"neural actuator {entry.name!r} mapping {mapping.index} input/output "
                    "worlds do not match"
                )
            common_worlds = _require_common_worlds(common_worlds, input_axes.worlds)
            request = MappedNeuralRequest(
                model=model,
                artifact_path=archive.path,
                metadata=archive.metadata,
                actuator=entry,
                mapping=mapping,
                input_axes=input_axes,
                output_axes=output_axes,
                control_dt_s=expected_dt,
            )
            if neural_factory is None:  # pragma: no cover - checked before mutation
                raise RuntimeError("mapped neural factory disappeared during construction")
            runtime_actuator = neural_factory(request)
            label = f"mapped neural actuator {entry.name!r} index {mapping.index}"
            _validate_runtime_actuator(runtime_actuator, label=label, neural=True)
            runtime_entries.append(
                RuntimeActuatorEntry(
                    entry.name,
                    "neural",
                    mapping.index,
                    input_axes.worlds,
                    runtime_actuator,
                )
            )
    if common_worlds is None:  # pragma: no cover - schema requires composition
        raise RuntimeError("deployment composition produced no runtime entries")
    return runtime_entries, common_worlds


def build_deployment_runtime(
    frozen: FrozenDeployState | str | Path,
    model: Any,
    *,
    repository_root: str | Path,
    solver: Any | None = None,
    neural_factory: MappedNeuralFactory | None = None,
    bindings: NewtonActuatorBindings | None = None,
) -> DeploymentRuntime:
    """Build an executable runtime solely from one frozen deployment state."""

    state = (
        frozen
        if isinstance(frozen, FrozenDeployState)
        else load_frozen_deploy_state(frozen)
    )
    if not isinstance(state, FrozenDeployState):
        raise TypeError("frozen must be FrozenDeployState or deploy/default.yaml")
    root_source = Path(repository_root)
    if root_source.is_symlink() or not root_source.is_dir():
        raise ValueError(f"repository_root must be a real directory: {root_source}")
    root = root_source.resolve(strict=True)
    _verify_frozen_source_identities(state, root)
    if state.composition.neural and neural_factory is None:
        neural_factory = _default_mapped_neural_factory

    expected_dt = 1.0 / state.simulation.control_freq
    verified: dict[str, _VerifiedNeural] = {}
    requires_dynamic_bias = False
    for entry in state.composition.neural:
        archive = verified.setdefault(
            entry.artifact.path,
            _verify_neural_artifact(root, entry),
        )
        if not math.isclose(entry.sample_dt_s, expected_dt, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"neural actuator {entry.name!r} timestep mismatch")
        requires_dynamic_bias |= any(
            feature.name == "dynamic_bias" for feature in archive.metadata.input_features
        )

    if solver is not None and getattr(solver, "model", model) is not model:
        raise ValueError("deployment solver belongs to a different Newton model")
    if requires_dynamic_bias and (
        solver is None or not callable(getattr(solver, "eval_dynamic_bias", None))
    ):
        raise ValueError(
            "dynamic_bias neural features require SolverMuJoCo.eval_dynamic_bias"
        )

    requested = set(state.ownership)
    for entry in state.composition.neural:
        for mapping in entry.mappings:
            requested.update(mapping.input_joints)
    instances = _model_joint_instances(model, tuple(sorted(requested)))
    loaded_bindings = bindings or _load_newton_bindings()
    owned_axes = _joint_axes(tuple(state.ownership), instances)
    updates = _model_updates(state, instances, loaded_bindings.joint_target_mode_type)
    originals = _apply_model_updates(model, updates)
    try:
        runtime_entries, common_worlds = _build_runtime_entries(
            state,
            model,
            instances,
            verified,
            expected_dt=expected_dt,
            neural_factory=neural_factory,
            bindings=loaded_bindings,
        )
        if solver is not None:
            notify = getattr(solver, "notify_model_changed", None)
            if not callable(notify):
                raise TypeError("deployment solver must expose notify_model_changed(flags)")
            notify(loaded_bindings.joint_dof_properties_flag)
    except BaseException:
        _restore_model_arrays(model, originals)
        raise
    return DeploymentRuntime(
        state,
        runtime_entries,
        worlds=common_worlds,
        solver=solver,
        requires_dynamic_bias=requires_dynamic_bias,
        owned_effort_indices=_warp_array(
            loaded_bindings,
            model,
            owned_axes.flatten_velocities(),
            loaded_bindings.warp.uint32,
        ),
        clear_owned_efforts=loaded_bindings.clear_owned_efforts,
        device=model.device,
    )


def _require_common_worlds(
    current: tuple[int, ...] | None,
    candidate: tuple[int, ...],
) -> tuple[int, ...]:
    """Require every runtime actuator to cover the same model worlds."""

    if current is not None and current != candidate:
        raise ValueError("deployment actuators do not cover identical Newton model worlds")
    return candidate


__all__ = [
    "DeploymentRuntime",
    "DeploymentRuntimeState",
    "MappedJointAxes",
    "MappedNeuralFactory",
    "MappedNeuralRequest",
    "NewtonActuatorBindings",
    "RuntimeActuatorEntry",
    "build_deployment_runtime",
    "load_frozen_deploy_state",
]
