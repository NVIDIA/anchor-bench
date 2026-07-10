"""Lazy Newton 1.4 dynamic-bias evaluation for actuator training."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from anchor_bench.core.hashing import sha256_file
from anchor_bench.training.asset_snapshot import AssetSnapshot, ScalarJointMapping, build_asset_snapshot

if TYPE_CHECKING:
    from anchor_bench.robots.registry import RobotBundle


_SOLVER_SETTING_NAMES = frozenset(
    {
        "separate_worlds",
        "njmax",
        "nconmax",
        "iterations",
        "ls_iterations",
        "ccd_iterations",
        "sdf_iterations",
        "sdf_initpoints",
        "solver",
        "integrator",
        "cone",
        "jacobian",
        "impratio",
        "tolerance",
        "ls_tolerance",
        "ccd_tolerance",
        "density",
        "viscosity",
        "wind",
        "magnetic",
        "use_mujoco_cpu",
        "enable_multiccd",
        "disable_contacts",
        "update_data_interval",
        "save_to_mjcf",
        "ls_parallel",
        "use_mujoco_contacts",
        "include_sites",
        "skip_visual_only_geoms",
    }
)


@dataclass(frozen=True)
class DynamicBiasResult:
    """Bias values for requested joints plus reproducibility metadata."""

    values: np.ndarray
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError("dynamic-bias values must be a 2-D array")
        if not np.all(np.isfinite(values)):
            raise ValueError("dynamic-bias values contain non-finite values")
        readonly = np.array(values, dtype=np.float32, copy=True)
        readonly.setflags(write=False)
        object.__setattr__(self, "values", readonly)
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))


class DynamicBiasProvider(Protocol):
    """Training-side interface for evaluating one synchronized episode."""

    def compute(
        self,
        *,
        joint_names: Sequence[str],
        positions: np.ndarray,
        velocities: np.ndarray,
        requested_joint_names: Sequence[str],
        cache_key: str | None = None,
    ) -> DynamicBiasResult: ...


@dataclass(frozen=True)
class _RuntimeModules:
    newton: Any
    warp: Any
    mujoco: Any
    mujoco_warp: Any


@dataclass(frozen=True)
class _MappedColumn:
    column: int
    recorded_name: str
    joint: ScalarJointMapping


@dataclass(frozen=True)
class _TrajectoryCacheEntry:
    fingerprint: str
    mapped_names: tuple[str, ...]
    values: np.ndarray


def resolve_runtime_asset(bundle: RobotBundle) -> Path:
    """Resolve the configured USD without importing runtime helpers eagerly."""

    from anchor_bench.runtime.assets import resolve_runtime_asset as resolve

    return resolve(bundle)


def _import_newton_runtime() -> _RuntimeModules:
    """Import the complete Newton/MuJoCo stack only on first real use."""

    try:
        newton = importlib.import_module("newton")
        warp = importlib.import_module("warp")
        mujoco = importlib.import_module("mujoco")
        mujoco_warp = importlib.import_module("mujoco_warp")
    except ImportError as exc:  # pragma: no cover - requires the full simulator environment
        raise RuntimeError(
            "dynamic_bias training requires Newton, Warp, MuJoCo, and MuJoCo Warp "
            "from the full runtime environment"
        ) from exc
    return _RuntimeModules(newton, warp, mujoco, mujoco_warp)


def _module_version(module: Any, distribution: str) -> str:
    value = getattr(module, "__version__", None)
    if value is not None:
        return str(value)
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _plain_setting(value: Any, *, label: str) -> Any:
    """Make a solver setting finite and safe for canonical provenance."""

    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"solver setting {label!r} must be finite")
        return result
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (tuple, list)):
        return [_plain_setting(item, label=label) for item in value]
    raise ValueError(f"solver setting {label!r} has unsupported value type {type(value).__name__}")


def _resolved_solver_settings(values: Mapping[str, Any] | None) -> dict[str, Any]:
    settings = dict(values or {})
    unknown = sorted(set(settings) - _SOLVER_SETTING_NAMES)
    if unknown:
        raise ValueError("unknown SolverMuJoCo settings: " + ", ".join(unknown))
    if settings.get("use_mujoco_cpu", False) is not False:
        raise ValueError("dynamic_bias uses the MuJoCo Warp backend; use_mujoco_cpu must be false")
    if settings.get("disable_contacts", True) is not True:
        raise ValueError("dynamic_bias requires disable_contacts=true")
    settings["use_mujoco_cpu"] = False
    settings["disable_contacts"] = True
    return {key: _plain_setting(value, label=key) for key, value in settings.items()}


def _ordered_names(values: Sequence[str], *, label: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"dynamic_bias {label} must be an ordered sequence")
    names = tuple(str(value).strip() for value in values)
    if not names or any(not name for name in names):
        raise ValueError(f"dynamic_bias {label} must contain non-empty names")
    duplicates = sorted({name for index, name in enumerate(names) if name in names[:index]})
    if duplicates:
        raise ValueError(f"dynamic_bias {label} must be unique: {', '.join(duplicates)}")
    return names


def _hash_text(digest: Any, value: str) -> None:
    encoded = value.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "little"))
    digest.update(encoded)


def _trajectory_fingerprint(
    *,
    asset_sha256: str,
    trajectory_names: tuple[str, ...],
    positions: np.ndarray,
    velocities: np.ndarray,
    mapped: Sequence[_MappedColumn],
) -> str:
    """Hash the actual computation inputs, never just their source path/key."""

    digest = hashlib.sha256()
    _hash_text(digest, asset_sha256)
    for name in trajectory_names:
        _hash_text(digest, name)
    digest.update(np.asarray(positions.shape, dtype="<i8").tobytes())
    digest.update(np.ascontiguousarray(positions, dtype="<f4").tobytes())
    digest.update(np.ascontiguousarray(velocities, dtype="<f4").tobytes())
    for item in mapped:
        _hash_text(digest, item.recorded_name)
        digest.update(np.asarray((item.column, item.joint.q_index, item.joint.dof_index), dtype="<i8").tobytes())
    return digest.hexdigest()


class NewtonMuJoCoDynamicBiasProvider:
    """Evaluate Newton's MuJoCo dynamic bias without advancing simulation.

    A single replicated Newton model, state, and output buffer are allocated on
    first use and then reused for every episode and batch.  Instances are not
    intended for concurrent calls to :meth:`compute`.
    """

    def __init__(
        self,
        bundle: RobotBundle,
        *,
        batch_size: int = 64,
        solver_settings: Mapping[str, Any] | None = None,
        device: str | None = None,
    ) -> None:
        if isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("dynamic-bias batch_size must be >= 1")
        self._bundle = bundle
        self._batch_size = int(batch_size)
        self._solver_settings = _resolved_solver_settings(solver_settings)
        self._device = device

        self._initialized = False
        self._snapshot: AssetSnapshot | None = None
        self._solver: Any = None
        self._state: Any = None
        self._bias: Any = None
        self._default_q = np.empty(0, dtype=np.float32)
        self._q_staging = np.empty(0, dtype=np.float32)
        self._qd_staging = np.empty(0, dtype=np.float32)
        self._trajectory_cache: dict[str, _TrajectoryCacheEntry] = {}

    @property
    def asset_snapshot(self) -> AssetSnapshot:
        """Initialize lazily and return the immutable training asset snapshot."""

        self._initialize()
        assert self._snapshot is not None
        return self._snapshot

    @property
    def snapshot(self) -> AssetSnapshot:
        """Short alias for :attr:`asset_snapshot`."""

        return self.asset_snapshot

    def effort_limits(self, joint_names: Sequence[str]) -> tuple[float, ...]:
        """Return model effort limits in requested output-joint order."""

        return self.asset_snapshot.effort_limits(joint_names)

    def _initialize(self) -> None:
        if self._initialized:
            return

        runtime = _import_newton_runtime()
        solver_namespace = getattr(runtime.newton, "solvers", None)
        solver_type = getattr(solver_namespace, "SolverMuJoCo", None)
        if solver_type is None or not callable(getattr(solver_type, "eval_dynamic_bias", None)):
            raise RuntimeError(
                "dynamic_bias training requires "
                "newton.solvers.SolverMuJoCo.eval_dynamic_bias(state, out=...)"
            )

        asset_path = resolve_runtime_asset(self._bundle).resolve(strict=True)
        asset_sha256 = sha256_file(asset_path)

        template = runtime.newton.ModelBuilder()
        register = getattr(solver_type, "register_custom_attributes", None)
        if not callable(register):
            raise RuntimeError("Newton SolverMuJoCo.register_custom_attributes is unavailable")
        register(template)
        template.add_usd(
            str(asset_path),
            floating=not self._bundle.spec.fixed_base,
            collapse_fixed_joints=True,
            enable_self_collisions=False,
            load_visual_shapes=False,
            skip_mesh_approximation=True,
        )

        scene = runtime.newton.ModelBuilder()
        scene.replicate(template, self._batch_size)
        finalize_kwargs = {} if self._device is None else {"device": self._device}
        model = scene.finalize(**finalize_kwargs)
        solver = solver_type(model, **self._solver_settings)

        if sha256_file(asset_path) != asset_sha256:
            raise RuntimeError(f"training asset changed while Newton was loading it: {asset_path}")

        versions = {
            "newton": _module_version(runtime.newton, "newton"),
            "warp": _module_version(runtime.warp, "warp-lang"),
            "mujoco": _module_version(runtime.mujoco, "mujoco"),
            "mujoco_warp": _module_version(runtime.mujoco_warp, "mujoco-warp"),
        }
        snapshot = build_asset_snapshot(
            asset_path=asset_path,
            asset_sha256=asset_sha256,
            model=model,
            template_joint_labels=tuple(template.joint_label),
            joints_per_world=int(template.joint_count),
            coordinates_per_world=int(template.joint_coord_count),
            dofs_per_world=int(template.joint_dof_count),
            newton=runtime.newton,
            versions=versions,
            solver_settings=self._solver_settings,
        )
        if snapshot.worlds != self._batch_size:
            raise ValueError(
                f"replicated Newton model has {snapshot.worlds} worlds, expected {self._batch_size}"
            )

        state = model.state()
        if state.joint_q is None or state.joint_qd is None:
            raise RuntimeError("Newton model state must allocate joint_q and joint_qd")
        bias = runtime.warp.empty(
            model.joint_dof_count,
            dtype=runtime.warp.float32,
            device=model.device,
        )

        self._snapshot = snapshot
        self._solver = solver
        self._state = state
        self._bias = bias
        self._default_q = np.array(snapshot.default_q, dtype=np.float32, copy=True)
        self._q_staging = np.array(self._default_q, dtype=np.float32, copy=True)
        self._qd_staging = np.zeros(self._batch_size * snapshot.dofs_per_world, dtype=np.float32)
        self._initialized = True

    def _mapped_columns(
        self,
        trajectory_names: tuple[str, ...],
        requested_names: tuple[str, ...],
    ) -> tuple[tuple[_MappedColumn, ...], tuple[str, ...]]:
        snapshot = self.asset_snapshot
        requested_set = set(requested_names)
        mapped: list[_MappedColumn] = []
        unmapped: list[str] = []
        owner_by_joint: dict[int, str] = {}
        for column, name in enumerate(trajectory_names):
            joint = snapshot.resolve_joint(
                name,
                required=name in requested_set,
                role="input joint",
            )
            if joint is None:
                unmapped.append(name)
                continue
            previous = owner_by_joint.get(joint.joint_index)
            if previous is not None:
                raise ValueError(
                    f"trajectory joints {previous!r} and {name!r} map to the same Newton scalar joint/DOF"
                )
            owner_by_joint[joint.joint_index] = name
            mapped.append(_MappedColumn(column, name, joint))
        return tuple(mapped), tuple(unmapped)

    def _evaluate_table(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        mapped: tuple[_MappedColumn, ...],
    ) -> np.ndarray:
        assert self._snapshot is not None
        assert self._state is not None
        assert self._solver is not None
        assert self._bias is not None

        snapshot = self._snapshot
        mapped_columns = np.asarray([item.column for item in mapped], dtype=np.int64)
        mapped_q = np.asarray([item.joint.q_index for item in mapped], dtype=np.int64)
        mapped_qd = np.asarray([item.joint.dof_index for item in mapped], dtype=np.int64)
        table = np.empty((len(positions), len(mapped)), dtype=np.float32)

        q = self._q_staging.reshape(self._batch_size, snapshot.coordinates_per_world)
        qd = self._qd_staging.reshape(self._batch_size, snapshot.dofs_per_world)
        for start in range(0, len(positions), self._batch_size):
            end = min(start + self._batch_size, len(positions))
            active = end - start
            np.copyto(self._q_staging, self._default_q)
            self._qd_staging.fill(0.0)
            q[:active, mapped_q] = positions[start:end, mapped_columns]
            qd[:active, mapped_qd] = velocities[start:end, mapped_columns]

            self._state.joint_q.assign(self._q_staging)
            self._state.joint_qd.assign(self._qd_staging)
            returned = self._solver.eval_dynamic_bias(self._state, out=self._bias)
            if returned is not self._bias:
                raise RuntimeError("SolverMuJoCo.eval_dynamic_bias did not return the supplied out buffer")
            bias = np.asarray(self._bias.numpy(), dtype=np.float32)
            expected = self._batch_size * snapshot.dofs_per_world
            if bias.shape != (expected,):
                raise RuntimeError(f"SolverMuJoCo dynamic bias has shape {bias.shape}, expected {(expected,)}")
            table[start:end] = bias.reshape(self._batch_size, snapshot.dofs_per_world)[:active, mapped_qd]

        if not np.all(np.isfinite(table)):
            raise ValueError("SolverMuJoCo produced non-finite dynamic_bias values")
        return table

    def _cached_table(
        self,
        *,
        cache_key: str | None,
        trajectory_names: tuple[str, ...],
        positions: np.ndarray,
        velocities: np.ndarray,
        mapped: tuple[_MappedColumn, ...],
    ) -> np.ndarray:
        assert self._snapshot is not None
        fingerprint = _trajectory_fingerprint(
            asset_sha256=self._snapshot.sha256,
            trajectory_names=trajectory_names,
            positions=positions,
            velocities=velocities,
            mapped=mapped,
        )
        mapped_names = tuple(item.recorded_name for item in mapped)
        if cache_key is not None:
            entry = self._trajectory_cache.get(cache_key)
            if entry is not None and entry.fingerprint == fingerprint and entry.mapped_names == mapped_names:
                return entry.values

        table = self._evaluate_table(positions, velocities, mapped)
        if cache_key is not None:
            cached = np.array(table, dtype=np.float32, copy=True)
            cached.setflags(write=False)
            self._trajectory_cache[cache_key] = _TrajectoryCacheEntry(fingerprint, mapped_names, cached)
            return cached
        return table

    def compute(
        self,
        *,
        joint_names: Sequence[str],
        positions: np.ndarray,
        velocities: np.ndarray,
        requested_joint_names: Sequence[str],
        cache_key: str | None = None,
    ) -> DynamicBiasResult:
        trajectory_names = _ordered_names(joint_names, label="joint_names")
        requested_names = _ordered_names(requested_joint_names, label="requested_joint_names")
        missing = [name for name in requested_names if name not in trajectory_names]
        if missing:
            raise ValueError("dynamic_bias trajectory is missing input joints: " + ", ".join(missing))
        if cache_key is not None and not isinstance(cache_key, str):
            raise ValueError("dynamic_bias cache_key must be a string or None")

        position_values = np.asarray(positions, dtype=np.float32)
        velocity_values = np.asarray(velocities, dtype=np.float32)
        expected_columns = len(trajectory_names)
        if position_values.ndim != 2 or position_values.shape[1:] != (expected_columns,):
            raise ValueError(f"dynamic_bias positions must have shape [samples, {expected_columns}]")
        if velocity_values.shape != position_values.shape:
            raise ValueError("dynamic_bias velocities must have the same shape as positions")
        if not len(position_values):
            raise ValueError("dynamic_bias requires at least one trajectory sample")
        if not np.all(np.isfinite(position_values)) or not np.all(np.isfinite(velocity_values)):
            raise ValueError("dynamic_bias input positions and velocities must be finite")

        self._initialize()
        assert self._snapshot is not None
        mapped, unmapped = self._mapped_columns(trajectory_names, requested_names)
        table = self._cached_table(
            cache_key=cache_key,
            trajectory_names=trajectory_names,
            positions=position_values,
            velocities=velocity_values,
            mapped=mapped,
        )
        table_column = {item.recorded_name: index for index, item in enumerate(mapped)}
        requested_columns = np.asarray([table_column[name] for name in requested_names], dtype=np.int64)
        result = np.array(table[:, requested_columns], dtype=np.float32, copy=True)

        mapped_indices = {item.joint.joint_index for item in mapped}
        defaulted = [
            joint.name
            for joint in self._snapshot.scalar_joints
            if joint.joint_index not in mapped_indices
        ]
        provenance = {
            **dict(self._snapshot.provenance),
            "source": "newton.solvers.SolverMuJoCo.eval_dynamic_bias",
            "quantity": "qfrc_bias",
            "batch_size": self._batch_size,
            "contacts": "disabled",
            "use_mujoco_cpu": False,
            "trajectory_joint_names": list(trajectory_names),
            "requested_joint_names": list(requested_names),
            "mapped_joint_names": [item.recorded_name for item in mapped],
            "mapped_asset_joint_names": [item.joint.name for item in mapped],
            "unmapped_trajectory_joint_names": list(unmapped),
            "defaulted_joint_names": defaulted,
            "unrecorded_positions": "model_defaults",
            "unrecorded_velocities": "zero",
        }
        return DynamicBiasResult(result, provenance)


__all__ = [
    "DynamicBiasProvider",
    "DynamicBiasResult",
    "NewtonMuJoCoDynamicBiasProvider",
    "resolve_runtime_asset",
]
