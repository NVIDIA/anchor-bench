from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np

if TYPE_CHECKING:
    from anchor_bench.data.motion import MotionData
    from anchor_bench.robots.registry import RobotBundle


@dataclass(frozen=True)
class DynamicBiasResult:
    """Dynamic-bias samples and the provenance needed to reproduce them."""

    values: np.ndarray
    provenance: Mapping[str, Any]


def _import_newton_runtime() -> Any:
    """Import Newton only when dynamic bias is evaluated."""
    try:
        import newton
    except ImportError as exc:  # pragma: no cover - full-runtime dependency
        raise RuntimeError("dynamic_bias training requires Newton 1.4") from exc
    return newton


def resolve_runtime_asset(bundle: RobotBundle) -> Path:
    """Resolve the robot USD without importing runtime helpers at module load."""
    from anchor_bench.runtime.assets import resolve_runtime_asset as resolve

    return resolve(bundle)


def _normalized_joint_name(value: str) -> str:
    """Normalize common USD joint labels for recorded-name matching."""
    return str(value).rsplit("/", 1)[-1].removesuffix("_joint")


def _file_sha256(path: Path) -> str:
    """Hash the exact asset bytes used to construct the solver."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _evaluate_dynamic_bias(solver: Any, state: Any) -> np.ndarray:
    """Use release-1.4's private CPU sync surface without integrating state."""
    required = ("model", "mj_model", "mj_data", "_mujoco", "_update_mjc_data")
    missing = [name for name in required if getattr(solver, name, None) is None]
    sync = getattr(solver, "_update_mjc_data", None)
    forward = getattr(getattr(solver, "_mujoco", None), "mj_forward", None)
    if missing or not callable(sync) or not callable(forward):
        details = ", ".join(missing) or "sync/forward helpers must be callable"
        raise RuntimeError(
            "Newton release-1.4 SolverMuJoCo dynamic-bias compatibility surface is unavailable: " + details
        )
    if getattr(solver, "use_mujoco_cpu", None) is not True:
        raise RuntimeError("dynamic_bias requires SolverMuJoCo(use_mujoco_cpu=True)")
    sync(solver.mj_data, solver.model, state)
    forward(solver.mj_model, solver.mj_data)
    raw = getattr(solver.mj_data, "qfrc_bias", None)
    if raw is None:
        raise RuntimeError("Newton release-1.4 SolverMuJoCo is missing mj_data.qfrc_bias")
    result = np.asarray(raw, dtype=np.float32)
    if result.ndim != 1 or not np.all(np.isfinite(result)):
        raise RuntimeError("SolverMuJoCo produced invalid MuJoCo qfrc_bias")
    return result


class NewtonMuJoCoDynamicBiasProvider:
    """Evaluate Newton MuJoCo dynamic bias without advancing simulation."""

    def __init__(self, bundle: RobotBundle) -> None:
        """Create a lazy provider for one robot asset."""
        self._bundle = bundle
        self._initialized = False

    def _initialize(self) -> None:
        """Load one Newton world and allocate reusable state and output arrays."""
        if self._initialized:
            return
        newton = _import_newton_runtime()
        solver_type = getattr(getattr(newton, "solvers", None), "SolverMuJoCo", None)
        if solver_type is None:
            raise RuntimeError("dynamic_bias training requires newton.solvers.SolverMuJoCo")
        register = getattr(solver_type, "register_custom_attributes", None)
        if not callable(register):
            raise RuntimeError("SolverMuJoCo.register_custom_attributes is unavailable")
        asset = resolve_runtime_asset(self._bundle).resolve(strict=True)
        asset_sha256 = _file_sha256(asset)
        builder = newton.ModelBuilder()
        register(builder)
        builder.add_usd(
            str(asset),
            floating=not self._bundle.spec.fixed_base,
            collapse_fixed_joints=True,
            enable_self_collisions=False,
            load_visual_shapes=False,
            skip_mesh_approximation=True,
        )
        model = builder.finalize()
        if int(model.world_count) != 1:
            raise RuntimeError("dynamic_bias requires a single-world Newton model")
        solver_settings = {"use_mujoco_cpu": True, "disable_contacts": True}
        solver = solver_type(model, **solver_settings)
        state = model.state()
        if state.joint_q is None or state.joint_qd is None:
            raise RuntimeError("Newton model state must allocate joint_q and joint_qd")
        if getattr(solver, "mj_qd_start", None) is None:
            raise RuntimeError("SolverMuJoCo is missing the mj_qd_start mapping")
        mj_qd_start = np.asarray(solver.mj_qd_start.numpy(), dtype=np.int64)
        if mj_qd_start.shape != (len(builder.joint_label),):
            raise RuntimeError("SolverMuJoCo mj_qd_start does not match the Newton joint layout")
        if _file_sha256(asset) != asset_sha256:
            raise RuntimeError("robot asset changed while constructing the dynamic-bias solver")
        self._newton, self._builder, self._model = newton, builder, model
        self._solver, self._state = solver, state
        self._default_q = np.asarray(model.joint_q.numpy(), dtype=np.float32).copy()
        self._mj_qd_start = mj_qd_start
        self._provenance = {
            "source": "mujoco.mj_forward.qfrc_bias",
            "quantity": "mujoco.MjData.qfrc_bias",
            "state_sync": "newton.solvers.SolverMuJoCo._update_mjc_data",
            "asset": asset.name,
            "asset_sha256": asset_sha256,
            "newton_version": str(getattr(newton, "__version__", "unknown")),
            "solver_settings": solver_settings,
            "unrecorded_positions": "model_defaults",
            "unrecorded_velocities": "zero",
        }
        self._initialized = True

    def _resolve_joint(self, name: str) -> int | None:
        """Resolve one recorded name to a unique Newton joint index."""
        labels = [str(label) for label in self._builder.joint_label]
        matches = [i for i, label in enumerate(labels) if label == name or label.endswith(f"/{name}")]
        if not matches:
            normalized = _normalized_joint_name(name)
            matches = [i for i, label in enumerate(labels) if _normalized_joint_name(label) == normalized]
        if len(matches) > 1:
            raise ValueError(f"trajectory joint {name!r} matches multiple Newton joints")
        return matches[0] if matches else None

    def _scalar_starts(self, index: int) -> tuple[int, int] | None:
        """Return coordinate and DOF starts for a scalar movable joint."""
        joint_type = self._builder.joint_type[index]
        if sum(self._builder.joint_dof_dim[index]) != 1 or joint_type not in (
            self._newton.JointType.REVOLUTE,
            self._newton.JointType.PRISMATIC,
        ):
            return None
        q_index, qd_index = int(self._builder.joint_q_start[index]), int(self._builder.joint_qd_start[index])
        return (q_index, qd_index) if q_index >= 0 and qd_index >= 0 else None

    def compute(self, motion: MotionData, requested_joints: Sequence[str]) -> DynamicBiasResult:
        """Evaluate ``qfrc_bias`` on the motion state grid for requested joints."""
        requested = tuple(str(name) for name in requested_joints)
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("dynamic_bias requested_joints must be non-empty and unique")
        trajectory = tuple(motion.joints.names)
        missing = [name for name in requested if name not in trajectory]
        if missing:
            raise ValueError("dynamic_bias trajectory is missing requested joints: " + ", ".join(missing))
        if motion.velocity is None:
            raise ValueError(f"{motion.source} is missing measured velocity required for dynamic_bias")
        positions = np.asarray(motion.position.values, dtype=np.float32)
        velocities = np.asarray(motion.velocity.interpolate(motion.position.time).values, dtype=np.float32)
        if not len(positions):
            raise ValueError("dynamic_bias requires at least one trajectory sample")
        self._initialize()
        mapped: list[tuple[int, str, int, int, int]] = []
        owner: dict[int, str] = {}
        for column, name in enumerate(trajectory):
            joint_index = self._resolve_joint(name)
            if joint_index is None:
                if name in requested:
                    raise ValueError(f"requested joint {name!r} was not found in the Newton model")
                continue
            starts = self._scalar_starts(joint_index)
            mj_qd = int(self._mj_qd_start[joint_index])
            if starts is None or mj_qd < 0:
                if name in requested:
                    raise ValueError(f"requested joint {name!r} is not a scalar joint in the Newton model")
                continue
            if joint_index in owner:
                raise ValueError(
                    f"trajectory joints {owner[joint_index]!r} and {name!r} map to the same Newton joint"
                )
            owner[joint_index] = name
            mapped.append((column, name, joint_index, starts[0], starts[1]))
        by_name = {name: int(self._mj_qd_start[joint]) for _, name, joint, _, _ in mapped}
        absent = [name for name in requested if name not in by_name]
        if absent:
            raise ValueError("requested joints not found in the Newton model: " + ", ".join(absent))
        columns = np.asarray([item[0] for item in mapped], dtype=np.int64)
        q_indices = np.asarray([item[3] for item in mapped], dtype=np.int64)
        qd_indices = np.asarray([item[4] for item in mapped], dtype=np.int64)
        requested_qd = np.asarray([by_name[name] for name in requested], dtype=np.int64)
        q = self._default_q.copy()
        qd = np.zeros(int(self._model.joint_dof_count), dtype=np.float32)
        values = np.empty((len(positions), len(requested)), dtype=np.float32)
        for row in range(len(positions)):
            q[q_indices] = positions[row, columns]
            qd[qd_indices] = velocities[row, columns]
            self._state.joint_q.assign(q)
            self._state.joint_qd.assign(qd)
            values[row] = _evaluate_dynamic_bias(self._solver, self._state)[requested_qd]

        mapped_indices = {item[2] for item in mapped}
        defaulted = [
            _normalized_joint_name(label)
            for index, label in enumerate(self._builder.joint_label)
            if index not in mapped_indices and self._scalar_starts(index) is not None
        ]
        provenance = {
            **self._provenance,
            "mapped_joint_names": [item[1] for item in mapped],
            "mapped_asset_joint_names": [
                _normalized_joint_name(self._builder.joint_label[item[2]]) for item in mapped
            ],
            "defaulted_joint_names": defaulted,
        }
        return DynamicBiasResult(values, provenance)
