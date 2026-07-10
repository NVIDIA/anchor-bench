"""Lazy, candidate-authenticated replay on Newton's MuJoCo solver.

The host-safe orchestration in this module owns interpolation, schema-v3 GRU
feature construction, recurrent boundaries, metrics, and CSV serialization.
The production adapter imports Newton only when a non-dry benchmark starts.
Small simulator and network protocols keep those semantics testable without a
GPU runtime.
"""

from __future__ import annotations

import csv
import importlib
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np

from anchor_bench.benchmark.phase import (
    BenchmarkExecutionResult,
    BenchmarkTrajectoryResult,
    ResolvedBenchmarkRun,
    SysIdBenchmarkCandidate,
    TrainingBenchmarkCandidate,
)
from anchor_bench.core.hashing import sha256_file, tree_fingerprint
from anchor_bench.data.motion import uniform_time_grid
from anchor_bench.data.sage import load_sage_motion
from anchor_bench.training.metadata import RuntimeGRUMetadata, RuntimeMapping, StatValue


@dataclass(frozen=True)
class ReplayState:
    """Finite robot position and velocity vectors in adapter joint order."""

    joint_names: tuple[str, ...]
    position: np.ndarray
    velocity: np.ndarray

    def __post_init__(self) -> None:
        """Freeze arrays and validate their shared one-dimensional axis."""

        names = _ordered_names(self.joint_names, label="replay state joint_names")
        position = _finite_array(self.position, ndim=1, label="replay state position")
        velocity = _finite_array(self.velocity, ndim=1, label="replay state velocity")
        expected = (len(names),)
        if position.shape != expected or velocity.shape != expected:
            raise ValueError(
                f"replay state position and velocity must have shape {expected}"
            )
        object.__setattr__(self, "joint_names", names)
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "velocity", velocity)


@dataclass(frozen=True)
class PreparedReplayMotion:
    """Linearly resampled benchmark data in canonical trajectory-joint order."""

    name: str
    joint_names: tuple[str, ...]
    timestamps: np.ndarray
    command: np.ndarray
    position: np.ndarray
    velocity: np.ndarray | None

    def __post_init__(self) -> None:
        """Validate all time and joint dimensions and freeze the arrays."""

        names = _ordered_names(self.joint_names, label="motion joint_names")
        timestamps = _finite_array(self.timestamps, ndim=1, label="motion timestamps")
        command = _finite_array(self.command, ndim=2, label="motion command")
        position = _finite_array(self.position, ndim=2, label="motion position")
        velocity = (
            None
            if self.velocity is None
            else _finite_array(self.velocity, ndim=2, label="motion velocity")
        )
        expected = (len(timestamps), len(names))
        if not len(timestamps):
            raise ValueError("benchmark motion must contain at least one resampled row")
        if command.shape != expected or position.shape != expected:
            raise ValueError(f"motion command and position must have shape {expected}")
        if velocity is not None and velocity.shape != expected:
            raise ValueError(f"motion velocity must have shape {expected}")
        if len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0.0):
            raise ValueError("motion timestamps must be strictly increasing")
        object.__setattr__(self, "joint_names", names)
        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "command", command)
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "velocity", velocity)


@dataclass(frozen=True)
class JointActuationOverride:
    """Candidate-owned physical values applied to one Newton scalar DOF."""

    kp: float
    kd: float
    force_only: bool
    dry_friction: float | None = None
    viscous_friction: float | None = None
    armature: float | None = None

    def __post_init__(self) -> None:
        """Require finite non-negative physical values."""

        for name in ("kp", "kd", "dry_friction", "viscous_friction", "armature"):
            value = getattr(self, name)
            if value is None:
                continue
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"joint override {name} must be finite and non-negative")


class BenchmarkReplayAdapter(Protocol):
    """Narrow state/control boundary implemented by the Newton runtime."""

    @property
    def joint_names(self) -> tuple[str, ...]:
        """Return canonical robot joints represented by state vectors."""

    def reset(
        self,
        *,
        initial_position: Mapping[str, float],
        initial_velocity: Mapping[str, float],
        initial_command: Mapping[str, float],
        buffer_time_s: float,
    ) -> ReplayState:
        """Reset, settle, and synchronize one independent motion."""

    def step(
        self,
        *,
        command: Mapping[str, float],
        effort: Mapping[str, float],
    ) -> ReplayState:
        """Advance exactly one candidate control timestep."""

    def eval_dynamic_bias(self) -> ReplayState:
        """Return live qfrc_bias in adapter joint order without advancing."""

    def close(self) -> None:
        """Release runtime resources."""


class NeuralBatchRunner(Protocol):
    """Shared-weight recurrent inference boundary for all metadata mappings."""

    def reset(self, batch_size: int) -> None:
        """Reset hidden state for every physical mapping in one motion."""

    def predict(self, normalized_features: np.ndarray) -> np.ndarray:
        """Return normalized output with shape ``[mappings, output_width]``."""

    def close(self) -> None:
        """Release network resources."""


ReplayAdapterFactory = Callable[[ResolvedBenchmarkRun], BenchmarkReplayAdapter]
NeuralRunnerFactory = Callable[[TrainingBenchmarkCandidate], NeuralBatchRunner]


def _ordered_names(values: Sequence[str], *, label: str) -> tuple[str, ...]:
    """Return non-empty unique joint names without coercing invalid entries."""

    if isinstance(values, (str, bytes)):
        raise ValueError(f"{label} must be an ordered sequence")
    names = tuple(values)
    if not names or any(
        not isinstance(name, str) or not name or name != name.strip()
        for name in names
    ):
        raise ValueError(f"{label} must contain non-empty names")
    if len(set(names)) != len(names):
        raise ValueError(f"{label} must not contain duplicates")
    return names


def _finite_array(values: Any, *, ndim: int, label: str) -> np.ndarray:
    """Return an immutable float64 array with finite values and fixed rank."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != ndim:
        raise ValueError(f"{label} must be a {ndim}-D array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} contains non-finite values")
    result = np.array(array, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _stat_vector(value: StatValue, width: int, *, label: str) -> np.ndarray:
    """Expand one scalar/vector metadata statistic to an exact float32 width."""

    raw = value if isinstance(value, tuple) else (value,)
    if len(raw) != width:
        raise ValueError(f"{label} must contain {width} value(s)")
    result = np.asarray(raw, dtype=np.float32)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must contain finite values")
    return result


def _mapping_value(
    value: StatValue,
    names: Sequence[str],
    joint: str,
    *,
    label: str,
) -> float:
    """Select one metadata vector entry by its explicitly paired joint axis."""

    raw = value if isinstance(value, tuple) else (value,)
    if len(raw) != len(names):
        raise ValueError(f"{label} must contain {len(names)} value(s)")
    try:
        index = tuple(names).index(joint)
    except ValueError as error:
        raise ValueError(f"{label} has no value for output joint {joint!r}") from error
    result = float(raw[index])
    if not math.isfinite(result):
        raise ValueError(f"{label} must contain finite values")
    return result


def _required_sysid_cell(
    parameters: Mapping[str, Mapping[str, float]], parameter: str, joint: str
) -> float:
    """Read one required finite SysID candidate cell."""

    joint_map = parameters.get(parameter)
    if joint_map is None or joint not in joint_map:
        raise ValueError(f"SysID benchmark candidate is missing {parameter}/{joint}")
    value = float(joint_map[joint])
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"SysID candidate cell {parameter}/{joint} must be finite and non-negative")
    return value


def candidate_joint_overrides(
    candidate: SysIdBenchmarkCandidate | TrainingBenchmarkCandidate,
) -> Mapping[str, JointActuationOverride]:
    """Resolve exactly the candidate-owned joint properties for Newton.

    SysID owns its selected joints and installs explicit PD and friction from
    the reconstructed candidate. Neural total-torque mappings disable solver
    PD only on their outputs. Residual mappings install their frozen baseline
    only on their outputs. All omitted joints are absent from the result and
    therefore retain USD-authored behavior.
    """

    overrides: dict[str, JointActuationOverride] = {}
    if isinstance(candidate, SysIdBenchmarkCandidate):
        parameters = candidate.loaded.state.parameters
        for joint in candidate.selected_joints:
            armature_map = parameters.get("armature", {})
            overrides[joint] = JointActuationOverride(
                kp=_required_sysid_cell(parameters, "stiffness", joint),
                kd=_required_sysid_cell(parameters, "damping", joint),
                dry_friction=_required_sysid_cell(parameters, "dry_friction", joint),
                viscous_friction=_required_sysid_cell(
                    parameters, "viscous_friction", joint
                ),
                armature=(
                    _required_sysid_cell(parameters, "armature", joint)
                    if joint in armature_map
                    else None
                ),
                force_only=False,
            )
        return MappingProxyType(overrides)

    metadata = candidate.metadata
    residual = metadata.target_columns == ("torque_residual",)
    for mapping in metadata.mappings:
        for joint in mapping.output_joints:
            if joint in overrides:
                raise ValueError(f"neural metadata owns output joint {joint!r} more than once")
            if residual:
                if mapping.pd_baseline is None or mapping.friction_baseline is None:
                    raise ValueError("residual neural mapping is missing its PD/friction baseline")
                kp = _mapping_value(
                    mapping.pd_baseline.kp,
                    mapping.pd_baseline.joint_names,
                    joint,
                    label="pd_baseline.kp",
                )
                kd = _mapping_value(
                    mapping.pd_baseline.kd,
                    mapping.pd_baseline.joint_names,
                    joint,
                    label="pd_baseline.kd",
                )
                dry = _mapping_value(
                    mapping.friction_baseline.dry,
                    mapping.friction_baseline.joint_names,
                    joint,
                    label="friction_baseline.dry",
                )
                viscous = _mapping_value(
                    mapping.friction_baseline.viscous,
                    mapping.friction_baseline.joint_names,
                    joint,
                    label="friction_baseline.viscous",
                )
            else:
                kp = kd = 0.0
                dry = viscous = None
            armature = (
                None
                if mapping.armature is None
                else _mapping_value(
                    mapping.armature.values,
                    mapping.armature.joint_names,
                    joint,
                    label="armature.values",
                )
            )
            overrides[joint] = JointActuationOverride(
                kp=kp,
                kd=kd,
                dry_friction=dry,
                viscous_friction=viscous,
                armature=armature,
                force_only=not residual,
            )
    return MappingProxyType(overrides)


def _candidate_sample_dt(
    candidate: SysIdBenchmarkCandidate | TrainingBenchmarkCandidate,
) -> float:
    """Return the authenticated candidate control timestep."""

    if isinstance(candidate, SysIdBenchmarkCandidate):
        return 1.0 / candidate.control_freq
    return float(candidate.metadata.sample_dt_s)


def _candidate_outputs(
    candidate: SysIdBenchmarkCandidate | TrainingBenchmarkCandidate,
) -> tuple[str, ...]:
    """Return the physical joints included in benchmark metrics."""

    if isinstance(candidate, SysIdBenchmarkCandidate):
        return candidate.selected_joints
    return candidate.output_joints


def _verify_replay_inputs(resolved: ResolvedBenchmarkRun) -> None:
    """Reject robot, asset, dataset, or motion changes made after resolution."""

    from anchor_bench.robots.fingerprints import (
        verify_robot_identity_fingerprint,
        verify_runtime_asset_fingerprint,
    )
    from anchor_bench.robots.registry import get_robot_from_root
    from anchor_bench.runtime.assets import resolve_runtime_asset

    bundle = get_robot_from_root(resolved.robot, resolved.repository_root)
    expected_robot = resolved.fingerprints.get("robot")
    if not isinstance(expected_robot, Mapping):
        raise ValueError("resolved benchmark is missing its robot fingerprint")
    verify_robot_identity_fingerprint(
        expected_robot,
        bundle,
        repository_root=resolved.repository_root,
    )
    expected_asset = resolved.fingerprints.get("asset")
    if not isinstance(expected_asset, Mapping):
        raise ValueError("resolved benchmark is missing its asset fingerprint")
    verify_runtime_asset_fingerprint(
        expected_asset,
        resolve_runtime_asset(bundle),
        runtime_usd=bundle.spec.runtime_usd,
    )

    expected_dataset = resolved.fingerprints.get("dataset")
    if not isinstance(expected_dataset, Mapping):
        raise ValueError("resolved benchmark is missing its dataset fingerprint")
    actual_dataset = tree_fingerprint(resolved.dataset_path)
    if actual_dataset.get("tree_sha256") != expected_dataset.get("tree_sha256"):
        raise RuntimeError("benchmark dataset changed after resolution")
    for source in resolved.motions:
        actual_motion = tree_fingerprint(source.path)
        if actual_motion != source.fingerprint:
            raise RuntimeError(
                f"benchmark motion {source.name!r} changed after resolution"
            )


def _prepare_motion(
    source: Any,
    *,
    sample_dt_s: float,
    signals: Sequence[str],
    sync_initial_state: bool,
) -> PreparedReplayMotion:
    """Load and linearly interpolate one motion without using torque columns."""

    motion = load_sage_motion(source.path)
    if tuple(motion.joints.names) != tuple(source.joint_names):
        raise ValueError(f"benchmark motion joint list changed after resolution: {source.path}")
    if len(source.canonical_joint_names) != len(source.joint_names):
        raise ValueError("resolved canonical motion joint axis has the wrong width")

    required_velocity = "velocity" in signals or sync_initial_state
    if required_velocity and motion.velocity is None:
        raise ValueError(f"benchmark motion {source.name!r} is missing required velocity")
    participating = [motion.command_position, motion.position]
    if required_velocity and motion.velocity is not None:
        participating.append(motion.velocity)
    if any(not len(signal.time) for signal in participating):
        raise ValueError(f"benchmark motion {source.name!r} contains an empty signal")
    start = max(float(signal.time[0]) for signal in participating)
    end = min(float(signal.time[-1]) for signal in participating)
    if end < start:
        raise ValueError(f"benchmark motion {source.name!r} control/state traces do not overlap")
    timestamps = uniform_time_grid(start, end, sample_dt_s)
    return PreparedReplayMotion(
        name=source.name,
        joint_names=tuple(source.canonical_joint_names),
        timestamps=timestamps,
        command=motion.command_position.interpolate(timestamps).values,
        position=motion.position.interpolate(timestamps).values,
        velocity=(
            None
            if motion.velocity is None
            else motion.velocity.interpolate(timestamps).values
        ),
    )


def _row_mapping(names: Sequence[str], row: np.ndarray) -> dict[str, float]:
    """Convert one aligned row to a finite name-to-value mapping."""

    if np.asarray(row).shape != (len(names),):
        raise ValueError("trajectory row width does not match its joint axis")
    return {name: float(value) for name, value in zip(names, row, strict=True)}


def _selected_columns(
    names: Sequence[str], requested: Sequence[str], *, label: str
) -> np.ndarray:
    """Return ordered column indices or reject a missing physical joint."""

    lookup = {name: index for index, name in enumerate(names)}
    missing = [name for name in requested if name not in lookup]
    if missing:
        raise ValueError(f"{label} is missing joints: {', '.join(missing)}")
    return np.asarray([lookup[name] for name in requested], dtype=int)


class TorchScriptGRURunner:
    """Lazy CPU TorchScript runner with one hidden row per physical mapping."""

    def __init__(self, candidate: TrainingBenchmarkCandidate) -> None:
        """Load the authenticated archive only after benchmark execution starts."""

        expected_network_sha = candidate.artifact_hashes.get("network")
        if expected_network_sha is None or sha256_file(candidate.network_path) != expected_network_sha:
            raise RuntimeError("neural network archive changed after benchmark resolution")
        try:
            torch = importlib.import_module("torch")
        except ImportError as error:  # pragma: no cover - full runtime dependency
            raise RuntimeError("neural benchmark replay requires PyTorch") from error
        self._torch = torch
        self._metadata = candidate.metadata
        try:
            self._network = torch.jit.load(
                str(candidate.network_path), map_location="cpu"
            ).eval()
        except Exception as error:  # pragma: no cover - archive/runtime specific
            raise RuntimeError(
                f"cannot load schema-v3 GRU archive {candidate.network_path}"
            ) from error
        self._hidden: Any | None = None
        self._batch_size = 0

    def reset(self, batch_size: int) -> None:
        """Allocate a physical-zero hidden tensor for a new motion."""

        if isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("GRU mapping batch_size must be positive")
        recurrent = self._metadata.recurrent
        self._batch_size = int(batch_size)
        self._hidden = self._torch.zeros(
            recurrent.num_layers,
            self._batch_size,
            recurrent.hidden_size,
            dtype=self._torch.float32,
        )

    def predict(self, normalized_features: np.ndarray) -> np.ndarray:
        """Evaluate one batch-first timestep and retain the returned hidden state."""

        expected = (self._batch_size, self._metadata.input_size)
        features = np.asarray(normalized_features, dtype=np.float32)
        if features.shape != expected:
            raise ValueError(f"GRU normalized features must have shape {expected}")
        if self._hidden is None:
            raise RuntimeError("GRU runner must be reset before inference")
        with self._torch.inference_mode():
            output, hidden = self._network(
                self._torch.from_numpy(features).unsqueeze(1), self._hidden
            )
        expected_output = (
            self._batch_size,
            1,
            self._metadata.output_size,
        )
        expected_hidden = (
            self._metadata.recurrent.num_layers,
            self._batch_size,
            self._metadata.recurrent.hidden_size,
        )
        if tuple(output.shape) != expected_output:
            raise ValueError(
                f"GRU network output shape is {tuple(output.shape)}, expected {expected_output}"
            )
        if tuple(hidden.shape) != expected_hidden:
            raise ValueError(
                f"GRU hidden shape is {tuple(hidden.shape)}, expected {expected_hidden}"
            )
        self._hidden = hidden
        result = output[:, 0, :].detach().cpu().numpy().astype(np.float32)
        if not np.all(np.isfinite(result)):
            raise ValueError("GRU network returned non-finite output")
        return result

    def close(self) -> None:
        """Drop network references; Torch owns any underlying allocations."""

        self._hidden = None


def _default_neural_runner_factory(
    candidate: TrainingBenchmarkCandidate,
) -> NeuralBatchRunner:
    """Construct the production schema-v3 TorchScript runner lazily."""

    return TorchScriptGRURunner(candidate)


class _NeuralReplaySession:
    """Schema-v3 feature assembly and recurrent state for one artifact."""

    def __init__(
        self,
        candidate: TrainingBenchmarkCandidate,
        runner: NeuralBatchRunner,
    ) -> None:
        """Precompute normalization and mapping constants from metadata."""

        self._candidate = candidate
        self._metadata = candidate.metadata
        self._runner = runner
        self._mappings = self._metadata.mappings
        self._previous = np.zeros(
            (len(self._mappings), self._metadata.output_size), dtype=np.float32
        )
        self._input_mean, self._input_std = _input_normalization(self._metadata)
        target = self._metadata.target_columns[0]
        self._target_mean = _stat_vector(
            self._metadata.normalization.targets.mean[target],
            self._metadata.output_size,
            label=f"normalization.targets.mean.{target}",
        )
        self._target_std = _stat_vector(
            self._metadata.normalization.targets.std[target],
            self._metadata.output_size,
            label=f"normalization.targets.std.{target}",
        )
        self._limits = np.vstack(
            [
                _stat_vector(
                    mapping.effort_limits,
                    self._metadata.output_size,
                    label="mapping.effort_limits",
                )
                for mapping in self._mappings
            ]
        )

    @property
    def needs_dynamic_bias(self) -> bool:
        """Return whether feature assembly requires live solver qfrc_bias."""

        return any(feature.name == "dynamic_bias" for feature in self._metadata.input_features)

    def reset(self) -> None:
        """Reset hidden state and previous raw network torque per motion/mapping."""

        self._previous.fill(0.0)
        self._runner.reset(len(self._mappings))

    def effort(
        self,
        *,
        state: ReplayState,
        command: Mapping[str, float],
        dynamic_bias: ReplayState | None,
    ) -> dict[str, float]:
        """Compute, denormalize, clamp, and route one neural control output."""

        features = _runtime_feature_batch(
            self._metadata,
            state=state,
            command=command,
            dynamic_bias=dynamic_bias,
            previous=self._previous,
        )
        normalized = (features - self._input_mean) / self._input_std
        raw_normalized = np.asarray(self._runner.predict(normalized), dtype=np.float32)
        expected = (len(self._mappings), self._metadata.output_size)
        if raw_normalized.shape != expected:
            raise ValueError(f"neural runner output must have shape {expected}")
        if not np.all(np.isfinite(raw_normalized)):
            raise ValueError("neural runner returned non-finite output")
        raw_physical = raw_normalized * self._target_std + self._target_mean
        applied = np.clip(raw_physical, -self._limits, self._limits)
        self._previous[...] = raw_physical
        routed: dict[str, float] = {}
        for row, mapping in enumerate(self._mappings):
            for column, joint in enumerate(mapping.output_joints):
                if joint in routed:
                    raise ValueError(f"neural mappings overlap on output joint {joint!r}")
                routed[joint] = float(applied[row, column])
        return routed

    def close(self) -> None:
        """Close the injected recurrent runner."""

        self._runner.close()


def _input_normalization(metadata: RuntimeGRUMetadata) -> tuple[np.ndarray, np.ndarray]:
    """Flatten schema-v3 input statistics in feature-major order."""

    means: list[np.ndarray] = []
    stds: list[np.ndarray] = []
    for feature in metadata.input_features:
        means.append(
            _stat_vector(
                metadata.normalization.inputs.mean[feature.name],
                feature.width,
                label=f"normalization.inputs.mean.{feature.name}",
            )
        )
        std = _stat_vector(
            metadata.normalization.inputs.std[feature.name],
            feature.width,
            label=f"normalization.inputs.std.{feature.name}",
        )
        if np.any(std <= 0.0):
            raise ValueError("input normalization standard deviations must be positive")
        stds.append(std)
    return np.concatenate(means), np.concatenate(stds)


def _state_values(state: ReplayState, names: Sequence[str], *, label: str) -> tuple[np.ndarray, np.ndarray]:
    """Select ordered position/velocity values from one full adapter state."""

    indices = _selected_columns(state.joint_names, names, label=label)
    return state.position[indices], state.velocity[indices]


def _command_values(command: Mapping[str, float], names: Sequence[str]) -> np.ndarray:
    """Select finite commands in one mapping's input-joint order."""

    missing = [name for name in names if name not in command]
    if missing:
        raise ValueError("benchmark command is missing input joints: " + ", ".join(missing))
    result = np.asarray([command[name] for name in names], dtype=np.float32)
    if not np.all(np.isfinite(result)):
        raise ValueError("benchmark command contains non-finite values")
    return result


def _pd_feature(
    mapping: RuntimeMapping,
    position_error: np.ndarray,
    velocity: np.ndarray,
) -> np.ndarray:
    """Evaluate the mapping's frozen zero-velocity PD observation."""

    baseline = mapping.pd_baseline
    if baseline is None:
        raise ValueError("solver_pd feature requires a mapping PD baseline")
    width = len(mapping.input_joints)
    kp = _stat_vector(baseline.kp, width, label="pd_baseline.kp")
    kd = _stat_vector(baseline.kd, width, label="pd_baseline.kd")
    return kp * position_error - kd * velocity


def _runtime_feature_batch(
    metadata: RuntimeGRUMetadata,
    *,
    state: ReplayState,
    command: Mapping[str, float],
    dynamic_bias: ReplayState | None,
    previous: np.ndarray,
) -> np.ndarray:
    """Build one ``[mapping, feature-major-width]`` physical input tensor."""

    expected_previous = (len(metadata.mappings), metadata.output_size)
    if previous.shape != expected_previous:
        raise ValueError(f"previous torque must have shape {expected_previous}")
    rows: list[np.ndarray] = []
    for mapping_index, mapping in enumerate(metadata.mappings):
        position, velocity = _state_values(
            state,
            mapping.input_joints,
            label=f"neural mapping {mapping_index} state",
        )
        targets = _command_values(command, mapping.input_joints)
        position_error = targets - position
        blocks: list[np.ndarray] = []
        for feature in metadata.input_features:
            if feature.components != ("value",):
                raise ValueError(
                    f"runtime feature {feature.name!r} has unsupported components "
                    f"{feature.components!r}; replay currently provides only 'value'"
                )
            if feature.name == "position":
                values = position
            elif feature.name == "position_error":
                values = position_error
            elif feature.name == "velocity":
                values = velocity
            elif feature.name == "solver_pd":
                values = _pd_feature(mapping, position_error, velocity)
            elif feature.name == "dynamic_bias":
                if dynamic_bias is None:
                    raise ValueError("dynamic_bias feature requires live solver bias values")
                values, _ = _state_values(
                    dynamic_bias,
                    mapping.input_joints,
                    label=f"neural mapping {mapping_index} dynamic bias",
                )
            elif feature.name == "previous_torque":
                values = previous[mapping_index]
            else:  # pragma: no cover - metadata schema rejects unsupported names
                raise ValueError(f"unsupported runtime feature {feature.name!r}")
            flat = np.asarray(values, dtype=np.float32).reshape(-1)
            if flat.shape != (feature.width,):
                raise ValueError(
                    f"runtime feature {feature.name!r} produced width {len(flat)}, "
                    f"expected {feature.width}"
                )
            blocks.append(flat)
        rows.append(np.concatenate(blocks))
    result = np.stack(rows).astype(np.float32, copy=False)
    expected = (len(metadata.mappings), metadata.input_size)
    if result.shape != expected or not np.all(np.isfinite(result)):
        raise ValueError(f"runtime feature batch must contain finite values with shape {expected}")
    return result


def _plain_version(module: Any) -> str:
    """Return a stable module version string for an internal asset snapshot."""

    return str(getattr(module, "__version__", "unknown"))


def _expected_asset_sha(resolved: ResolvedBenchmarkRun) -> str | None:
    """Read a recognized current-asset digest from resolved fingerprints."""

    fingerprint = resolved.fingerprints.get("asset")
    if not isinstance(fingerprint, Mapping):
        return None
    values = [
        fingerprint[key]
        for key in ("asset_sha256", "sha256")
        if key in fingerprint
    ]
    if len(values) != 1 or not isinstance(values[0], str):
        return None
    return values[0]


def _requires_dynamic_bias(candidate: Any) -> bool:
    """Return whether a neural candidate requests the live bias feature."""

    return isinstance(candidate, TrainingBenchmarkCandidate) and any(
        feature.name == "dynamic_bias"
        for feature in candidate.metadata.input_features
    )


def _require_dynamic_bias_capability(solver: Any) -> None:
    """Fail clearly when the active Newton solver lacks the pinned bias API."""

    if not callable(getattr(solver, "eval_dynamic_bias", None)):
        raise RuntimeError(
            "dynamic_bias benchmark replay requires "
            "newton.solvers.SolverMuJoCo.eval_dynamic_bias(state, out=...)"
        )


def _assign_model_values(array: Any, updates: Mapping[int, float]) -> None:
    """Apply indexed host values to one Warp-like Newton model array."""

    if not updates:
        return
    values = np.asarray(array.numpy()).copy()
    for index, value in updates.items():
        values[index] = value
    array.assign(values)


class NewtonMuJoCoReplayAdapter:
    """Direct Newton 1.4 replay that does not use its schema-v1 GRU controller.

    Schema-v3 inference remains in Anchor Bench. This adapter imports the USD,
    applies only candidate-owned model properties, writes commands and neural
    force into Newton's canonical target/force arrays, and advances one
    ``SolverMuJoCo`` instance.
    """

    def __init__(self, resolved: ResolvedBenchmarkRun) -> None:
        """Build a one-world fixed-base Newton model and configure the candidate."""

        try:
            newton = importlib.import_module("newton")
            warp = importlib.import_module("warp")
        except ImportError as error:  # pragma: no cover - full runtime dependency
            raise RuntimeError(
                "benchmark replay requires Newton and Warp from the full runtime"
            ) from error

        from anchor_bench.robots.registry import get_robot_from_root
        from anchor_bench.runtime.assets import resolve_runtime_asset
        from anchor_bench.training.asset_snapshot import build_asset_snapshot

        bundle = get_robot_from_root(resolved.robot, resolved.repository_root)
        asset_path = resolve_runtime_asset(bundle).resolve(strict=True)
        asset_sha256 = sha256_file(asset_path)
        expected_asset_sha256 = _expected_asset_sha(resolved)
        if expected_asset_sha256 is not None and asset_sha256 != expected_asset_sha256:
            raise RuntimeError(
                f"runtime asset changed after benchmark resolution: {asset_path}"
            )
        if (
            isinstance(resolved.candidate, TrainingBenchmarkCandidate)
            and resolved.candidate.metadata.hashes.asset_sha256 != asset_sha256
        ):
            raise RuntimeError("runtime USD does not match the neural metadata asset hash")

        solver_type = getattr(getattr(newton, "solvers", None), "SolverMuJoCo", None)
        if solver_type is None:
            raise RuntimeError("benchmark replay requires newton.solvers.SolverMuJoCo")
        builder = newton.ModelBuilder()
        register = getattr(solver_type, "register_custom_attributes", None)
        if not callable(register):
            raise RuntimeError(
                "benchmark replay requires SolverMuJoCo.register_custom_attributes"
            )
        register(builder)
        builder.add_usd(
            str(asset_path),
            floating=not bundle.spec.fixed_base,
            collapse_fixed_joints=True,
            enable_self_collisions=False,
            load_visual_shapes=False,
            skip_mesh_approximation=True,
        )
        model = builder.finalize()
        snapshot = build_asset_snapshot(
            asset_path=asset_path,
            asset_sha256=asset_sha256,
            model=model,
            template_joint_labels=tuple(builder.joint_label),
            joints_per_world=int(builder.joint_count),
            coordinates_per_world=int(builder.joint_coord_count),
            dofs_per_world=int(builder.joint_dof_count),
            newton=newton,
            versions={"newton": _plain_version(newton), "warp": _plain_version(warp)},
            solver_settings={
                "integrator": resolved.resolved_profile.simulation.integrator,
                "iterations": resolved.resolved_profile.simulation.solver_iterations,
                "ls_iterations": resolved.resolved_profile.simulation.line_search_iterations,
            },
        )
        if snapshot.worlds != 1:
            raise ValueError("benchmark replay expects one Newton world")

        joint_names = tuple(bundle.spec.joint_names)
        joints = {
            name: snapshot.resolve_joint(name, required=True, role="robot joint")
            for name in joint_names
        }
        if any(joint is None for joint in joints.values()):  # pragma: no cover - required invariant
            raise RuntimeError("Newton asset scalar-joint resolution unexpectedly returned None")

        self._newton = newton
        self._warp = warp
        self._model = model
        self._snapshot = snapshot
        self._joint_names = joint_names
        self._joints = joints
        self._apply_candidate_overrides(
            candidate_joint_overrides(resolved.candidate)
        )

        simulation = resolved.resolved_profile.simulation
        if simulation.physics_freq % simulation.control_freq != 0:
            raise ValueError(
                "benchmark simulation requires physics_freq divisible by control_freq"
            )
        self._physics_dt = 1.0 / simulation.physics_freq
        self._control_substeps = simulation.physics_freq // simulation.control_freq
        try:
            self._solver = solver_type(
                model,
                integrator=simulation.integrator,
                iterations=simulation.solver_iterations,
                ls_iterations=simulation.line_search_iterations,
            )
        except Exception as error:  # pragma: no cover - full runtime dependency
            raise RuntimeError("failed to construct Newton SolverMuJoCo for replay") from error
        if _requires_dynamic_bias(resolved.candidate):
            _require_dynamic_bias_capability(self._solver)

        self._state_0 = model.state()
        self._state_1 = model.state()
        self._control = model.control()
        if (
            self._state_0.joint_q is None
            or self._state_0.joint_qd is None
            or self._control.joint_f is None
            or self._control.joint_target_q is None
            or self._control.joint_target_qd is None
        ):
            raise RuntimeError("Newton replay model did not allocate required joint arrays")
        self._dynamic_bias = warp.empty(
            model.joint_dof_count,
            dtype=warp.float32,
            device=model.device,
        )

    @property
    def joint_names(self) -> tuple[str, ...]:
        """Return the canonical configured robot axis."""

        return self._joint_names

    def _apply_candidate_overrides(
        self, overrides: Mapping[str, JointActuationOverride]
    ) -> None:
        """Apply only candidate-owned DOF properties before solver construction."""

        unknown = sorted(set(overrides) - set(self._joint_names))
        if unknown:
            raise ValueError("candidate overrides unknown robot joints: " + ", ".join(unknown))
        kp: dict[int, float] = {}
        kd: dict[int, float] = {}
        mode: dict[int, int] = {}
        dry: dict[int, float] = {}
        viscous: dict[int, float] = {}
        armature: dict[int, float] = {}
        target_mode = self._newton.JointTargetMode
        for name, values in overrides.items():
            joint = self._joints[name]
            assert joint is not None
            dof = joint.dof_index
            kp[dof] = values.kp
            kd[dof] = values.kd
            mode[dof] = int(
                target_mode.EFFORT
                if values.force_only
                else target_mode.from_gains(values.kp, values.kd, has_drive=True)
            )
            if values.dry_friction is not None:
                dry[dof] = values.dry_friction
            if values.viscous_friction is not None:
                viscous[dof] = values.viscous_friction
            if values.armature is not None:
                armature[dof] = values.armature
        _assign_model_values(self._model.joint_target_ke, kp)
        _assign_model_values(self._model.joint_target_kd, kd)
        _assign_model_values(self._model.joint_target_mode, mode)
        _assign_model_values(self._model.joint_friction, dry)
        _assign_model_values(self._model.joint_damping, viscous)
        _assign_model_values(self._model.joint_armature, armature)

    def _state_arrays(
        self,
        positions: Mapping[str, float],
        velocities: Mapping[str, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Overlay recorded scalar joints on model-default generalized state."""

        unknown = sorted((set(positions) | set(velocities)) - set(self._joint_names))
        if unknown:
            raise ValueError("initial state contains unknown robot joints: " + ", ".join(unknown))
        q = np.asarray(self._model.joint_q.numpy(), dtype=np.float32).copy()
        qd = np.zeros(self._model.joint_dof_count, dtype=np.float32)
        for name, value in positions.items():
            joint = self._joints[name]
            assert joint is not None
            q[joint.q_index] = float(value)
        for name, value in velocities.items():
            joint = self._joints[name]
            assert joint is not None
            qd[joint.dof_index] = float(value)
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(qd)):
            raise ValueError("initial Newton state contains non-finite values")
        return q, qd

    def _write_command(self, command: Mapping[str, float]) -> None:
        """Write canonical position targets and zero velocity targets."""

        unknown = sorted(set(command) - set(self._joint_names))
        if unknown:
            raise ValueError("command contains unknown robot joints: " + ", ".join(unknown))
        targets = np.asarray(self._control.joint_target_q.numpy(), dtype=np.float32).copy()
        velocities = np.zeros(self._model.joint_dof_count, dtype=np.float32)
        for name, value in command.items():
            joint = self._joints[name]
            assert joint is not None
            targets[joint.q_index] = float(value)
        self._control.joint_target_q.assign(targets)
        self._control.joint_target_qd.assign(velocities)

    def _write_effort(self, effort: Mapping[str, float]) -> None:
        """Write candidate neural force while zeroing every unowned DOF."""

        unknown = sorted(set(effort) - set(self._joint_names))
        if unknown:
            raise ValueError("effort contains unknown robot joints: " + ", ".join(unknown))
        forces = np.zeros(self._model.joint_dof_count, dtype=np.float32)
        for name, value in effort.items():
            joint = self._joints[name]
            assert joint is not None
            forces[joint.dof_index] = float(value)
        if not np.all(np.isfinite(forces)):
            raise ValueError("neural effort contains non-finite values")
        self._control.joint_f.assign(forces)

    def _advance_substeps(self, count: int) -> None:
        """Advance a fixed number of physical substeps with current controls."""

        for _ in range(count):
            self._solver.step(
                self._state_0,
                self._state_1,
                self._control,
                None,
                self._physics_dt,
            )
            self._state_0, self._state_1 = self._state_1, self._state_0

    def _assign_live_state(self, q: np.ndarray, qd: np.ndarray) -> None:
        """Clear solver internals and install one exact generalized state."""

        self._solver.reset(self._state_0)
        self._state_0.joint_q.assign(q)
        self._state_0.joint_qd.assign(qd)
        self._state_1.joint_q.assign(q)
        self._state_1.joint_qd.assign(qd)

    def reset(
        self,
        *,
        initial_position: Mapping[str, float],
        initial_velocity: Mapping[str, float],
        initial_command: Mapping[str, float],
        buffer_time_s: float,
    ) -> ReplayState:
        """Settle at the first command, then restore the exact measured row zero."""

        if not math.isfinite(buffer_time_s) or buffer_time_s < 0.0:
            raise ValueError("benchmark buffer_time_s must be finite and non-negative")
        q, qd = self._state_arrays(initial_position, initial_velocity)
        self._assign_live_state(q, qd)
        self._write_command(initial_command)
        self._write_effort({})
        buffer_controls = int(round(buffer_time_s / (self._physics_dt * self._control_substeps)))
        self._advance_substeps(buffer_controls * self._control_substeps)
        self._assign_live_state(q, qd)
        return self._read_state()

    def _read_state(self) -> ReplayState:
        """Gather canonical scalar joint state from Newton generalized arrays."""

        q = np.asarray(self._state_0.joint_q.numpy(), dtype=np.float64)
        qd = np.asarray(self._state_0.joint_qd.numpy(), dtype=np.float64)
        position = []
        velocity = []
        for name in self._joint_names:
            joint = self._joints[name]
            assert joint is not None
            position.append(q[joint.q_index])
            velocity.append(qd[joint.dof_index])
        return ReplayState(self._joint_names, position, velocity)

    def step(
        self,
        *,
        command: Mapping[str, float],
        effort: Mapping[str, float],
    ) -> ReplayState:
        """Apply one held command/effort pair for a complete control interval."""

        self._write_command(command)
        self._write_effort(effort)
        self._advance_substeps(self._control_substeps)
        return self._read_state()

    def eval_dynamic_bias(self) -> ReplayState:
        """Evaluate and gather qfrc_bias without changing simulation time."""

        _require_dynamic_bias_capability(self._solver)
        result = self._solver.eval_dynamic_bias(
            self._state_0, out=self._dynamic_bias
        )
        if result is not self._dynamic_bias:
            raise RuntimeError("SolverMuJoCo.eval_dynamic_bias did not reuse its output")
        values = np.asarray(result.numpy(), dtype=np.float64)
        selected = []
        for name in self._joint_names:
            joint = self._joints[name]
            assert joint is not None
            selected.append(values[joint.dof_index])
        return ReplayState(self._joint_names, selected, np.zeros(len(selected)))

    def close(self) -> None:
        """Release strong references to the solver-owned state."""

        self._solver = None


def _default_adapter_factory(resolved: ResolvedBenchmarkRun) -> BenchmarkReplayAdapter:
    """Construct the direct Newton 1.4 adapter lazily."""

    return NewtonMuJoCoReplayAdapter(resolved)


def _initial_rows(
    motion: PreparedReplayMotion,
    *,
    sync_initial_state: bool,
) -> tuple[dict[str, float], dict[str, float]]:
    """Choose exact measured row zero or command/zero fallback for reset."""

    if sync_initial_state:
        position = _row_mapping(motion.joint_names, motion.position[0])
        if motion.velocity is None:
            raise ValueError("initial-state synchronization requires recorded velocity")
        velocity = _row_mapping(motion.joint_names, motion.velocity[0])
        return position, velocity
    return (
        _row_mapping(motion.joint_names, motion.command[0]),
        {name: 0.0 for name in motion.joint_names},
    )


def _validate_adapter_state(
    state: Any, expected_names: Sequence[str]
) -> ReplayState:
    """Require an adapter state covering its declared canonical joint axis."""

    if not isinstance(state, ReplayState):
        raise TypeError("benchmark replay adapter must return ReplayState")
    if state.joint_names != tuple(expected_names):
        raise ValueError("benchmark replay adapter changed its joint axis")
    return state


def _trajectory_csv(
    timestamps: np.ndarray,
    joints: Sequence[str],
    signals: Sequence[str],
    values: Mapping[str, np.ndarray],
) -> str:
    """Serialize one comparison trace with no torque column or placeholder."""

    stream = StringIO(newline="")
    columns = ["timestamp_s"] + [
        f"{signal}/{joint}" for signal in signals for joint in joints
    ]
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(columns)
    for row, timestamp in enumerate(timestamps):
        payload: list[str] = [format(float(timestamp), ".17g")]
        for signal in signals:
            array = values[signal]
            payload.extend(format(float(value), ".17g") for value in array[row])
        writer.writerow(payload)
    return stream.getvalue()


def _signal_metrics(
    real: np.ndarray, simulated: np.ndarray, joints: Sequence[str]
) -> dict[str, Any]:
    """Return finite overall and per-joint RMSE/MAE values."""

    actual = np.asarray(real, dtype=np.float64)
    predicted = np.asarray(simulated, dtype=np.float64)
    if actual.shape != predicted.shape or actual.shape[1:] != (len(joints),):
        raise ValueError("metric signal arrays must share [time, candidate_outputs] shape")
    delta = predicted - actual
    if not np.all(np.isfinite(delta)):
        raise ValueError("benchmark metric inputs contain non-finite values")
    return {
        "rmse": float(np.sqrt(np.mean(delta**2))),
        "mae": float(np.mean(np.abs(delta))),
        "per_joint": {
            joint: {
                "rmse": float(np.sqrt(np.mean(delta[:, index] ** 2))),
                "mae": float(np.mean(np.abs(delta[:, index]))),
            }
            for index, joint in enumerate(joints)
        },
    }


def _replay_motion(
    resolved: ResolvedBenchmarkRun,
    motion: PreparedReplayMotion,
    adapter: BenchmarkReplayAdapter,
    neural: _NeuralReplaySession | None,
) -> tuple[BenchmarkTrajectoryResult, dict[str, Any], dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Replay one independent motion and return CSV plus metric source arrays."""

    outputs = _candidate_outputs(resolved.candidate)
    output_columns = _selected_columns(
        motion.joint_names, outputs, label=f"benchmark motion {motion.name!r}"
    )
    initial_position, initial_velocity = _initial_rows(
        motion, sync_initial_state=resolved.resolved_profile.sync_initial_state
    )
    initial_command = _row_mapping(motion.joint_names, motion.command[0])
    state = _validate_adapter_state(
        adapter.reset(
            initial_position={
                name: value
                for name, value in initial_position.items()
                if name in adapter.joint_names
            },
            initial_velocity={
                name: value
                for name, value in initial_velocity.items()
                if name in adapter.joint_names
            },
            initial_command={
                name: value
                for name, value in initial_command.items()
                if name in adapter.joint_names
            },
            buffer_time_s=float(resolved.resolved_profile.buffer_time),
        ),
        adapter.joint_names,
    )
    if neural is not None:
        neural.reset()

    adapter_outputs = _selected_columns(
        adapter.joint_names, outputs, label="benchmark replay adapter"
    )
    simulated_position = np.empty((len(motion.timestamps), len(outputs)), dtype=np.float64)
    simulated_velocity = np.empty_like(simulated_position)
    simulated_position[0] = state.position[adapter_outputs]
    simulated_velocity[0] = state.velocity[adapter_outputs]
    for row in range(len(motion.timestamps) - 1):
        command = _row_mapping(motion.joint_names, motion.command[row])
        runtime_command = {
            name: value for name, value in command.items() if name in adapter.joint_names
        }
        effort: Mapping[str, float] = {}
        if neural is not None:
            dynamic_bias = adapter.eval_dynamic_bias() if neural.needs_dynamic_bias else None
            if dynamic_bias is not None:
                dynamic_bias = _validate_adapter_state(dynamic_bias, adapter.joint_names)
            effort = neural.effort(
                state=state,
                command=runtime_command,
                dynamic_bias=dynamic_bias,
            )
        state = _validate_adapter_state(
            adapter.step(command=runtime_command, effort=effort),
            adapter.joint_names,
        )
        simulated_position[row + 1] = state.position[adapter_outputs]
        simulated_velocity[row + 1] = state.velocity[adapter_outputs]

    real_values: dict[str, np.ndarray] = {"position": motion.position[:, output_columns]}
    simulated_values: dict[str, np.ndarray] = {"position": simulated_position}
    if "velocity" in resolved.resolved_profile.signals:
        if motion.velocity is None:
            raise ValueError(f"benchmark motion {motion.name!r} has no velocity signal")
        real_values["velocity"] = motion.velocity[:, output_columns]
        simulated_values["velocity"] = simulated_velocity
    signals = tuple(resolved.resolved_profile.signals)
    metrics = {
        signal: _signal_metrics(real_values[signal], simulated_values[signal], outputs)
        for signal in signals
    }
    sources = {
        signal: (real_values[signal], simulated_values[signal])
        for signal in signals
    }
    trajectories = BenchmarkTrajectoryResult(
        real_csv=_trajectory_csv(
            motion.timestamps, outputs, signals, real_values
        ),
        simulated_csv=_trajectory_csv(
            motion.timestamps, outputs, signals, simulated_values
        ),
    )
    return trajectories, metrics, sources


def run_benchmark_replay(
    resolved: ResolvedBenchmarkRun,
    *,
    adapter_factory: ReplayAdapterFactory | None = None,
    neural_runner_factory: NeuralRunnerFactory | None = None,
) -> BenchmarkExecutionResult:
    """Replay an exact SysID or schema-v3 neural candidate on held-out motions."""

    _verify_replay_inputs(resolved)
    sample_dt_s = _candidate_sample_dt(resolved.candidate)
    prepared = tuple(
        _prepare_motion(
            source,
            sample_dt_s=sample_dt_s,
            signals=resolved.resolved_profile.signals,
            sync_initial_state=resolved.resolved_profile.sync_initial_state,
        )
        for source in resolved.motions
    )
    adapter = (adapter_factory or _default_adapter_factory)(resolved)
    neural: _NeuralReplaySession | None = None
    trajectories: dict[str, BenchmarkTrajectoryResult] = {}
    motion_metrics: dict[str, Any] = {}
    aggregate_sources: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        signal: [] for signal in resolved.resolved_profile.signals
    }
    try:
        if not isinstance(adapter.joint_names, tuple):
            raise TypeError("benchmark replay adapter joint_names must be a tuple")
        _ordered_names(adapter.joint_names, label="benchmark replay adapter joint_names")
        if isinstance(resolved.candidate, TrainingBenchmarkCandidate):
            runner = (neural_runner_factory or _default_neural_runner_factory)(
                resolved.candidate
            )
            try:
                neural = _NeuralReplaySession(resolved.candidate, runner)
            except Exception:
                runner.close()
                raise
        for motion in prepared:
            trajectory, metrics, sources = _replay_motion(
                resolved, motion, adapter, neural
            )
            trajectories[motion.name] = trajectory
            motion_metrics[motion.name] = metrics
            for signal, values in sources.items():
                aggregate_sources[signal].append(values)
    finally:
        if neural is not None:
            neural.close()
        adapter.close()

    outputs = _candidate_outputs(resolved.candidate)
    aggregate: dict[str, Any] = {}
    for signal, pairs in aggregate_sources.items():
        if not pairs:
            raise ValueError(f"benchmark produced no {signal!r} metric samples")
        aggregate[signal] = _signal_metrics(
            np.concatenate([real for real, _simulated in pairs], axis=0),
            np.concatenate([simulated for _real, simulated in pairs], axis=0),
            outputs,
        )
    result = BenchmarkExecutionResult(
        metrics={
            "candidate_outputs": list(outputs),
            "signals": list(resolved.resolved_profile.signals),
            "aggregate": aggregate,
            "motions": motion_metrics,
        },
        trajectories=trajectories,
    )
    result.validated_metrics()
    return result


__all__ = [
    "BenchmarkReplayAdapter",
    "JointActuationOverride",
    "NeuralBatchRunner",
    "NewtonMuJoCoReplayAdapter",
    "PreparedReplayMotion",
    "ReplayAdapterFactory",
    "ReplayState",
    "TorchScriptGRURunner",
    "candidate_joint_overrides",
    "run_benchmark_replay",
]
