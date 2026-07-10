from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

if TYPE_CHECKING:
    from anchor_bench.data.motion import MotionData
    from anchor_bench.robots.registry import RobotBundle


def _import_newton_runtime() -> Any:
    """Import Newton only when dynamic bias is evaluated."""
    try:
        import newton
    except ImportError as exc:  # pragma: no cover - full-runtime dependency
        raise RuntimeError("dynamic_bias training requires Newton 1.4") from exc
    return newton


def _normalized_joint_name(value: str) -> str:
    """Normalize common USD joint labels for recorded-name matching."""
    return str(value).rsplit("/", 1)[-1].removesuffix("_joint")


def _evaluate_dynamic_bias(solver: Any, state: Any) -> np.ndarray:
    """Evaluate bias through the pinned Newton release's MuJoCo bridge."""
    solver._update_mjc_data(solver.mj_data, solver.model, state)
    solver._mujoco.mj_forward(solver.mj_model, solver.mj_data)
    return np.asarray(solver.mj_data.qfrc_bias, dtype=np.float32)


class NewtonMuJoCoDynamicBiasProvider:
    """Evaluate Newton MuJoCo dynamic bias without advancing simulation."""

    def __init__(self, bundle: RobotBundle) -> None:
        """Create a lazy provider for one robot asset."""
        self._bundle = bundle
        self._initialized = False

    def _initialize(self) -> None:
        """Load one Newton world and allocate reusable state arrays."""
        if self._initialized:
            return
        newton = _import_newton_runtime()
        solver_type = newton.solvers.SolverMuJoCo

        from anchor_bench.runtime.assets import resolve_runtime_asset

        asset = resolve_runtime_asset(self._bundle).resolve(strict=True)
        builder = newton.ModelBuilder()
        solver_type.register_custom_attributes(builder)
        builder.add_usd(
            str(asset),
            floating=not self._bundle.spec.fixed_base,
            collapse_fixed_joints=True,
            enable_self_collisions=False,
            load_visual_shapes=False,
            skip_mesh_approximation=True,
        )
        model = builder.finalize()
        solver = solver_type(model, use_mujoco_cpu=True, disable_contacts=True)
        self._newton, self._builder, self._model = newton, builder, model
        self._solver, self._state = solver, model.state()
        self._default_q = np.asarray(model.joint_q.numpy(), dtype=np.float32).copy()
        self._mj_qd_start = np.asarray(solver.mj_qd_start.numpy(), dtype=np.int64)
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

    def compute(self, motion: MotionData, requested_joints: Sequence[str]) -> np.ndarray:
        """Evaluate ``qfrc_bias`` on the motion state grid for requested joints."""
        requested = tuple(str(name) for name in requested_joints)
        if not requested:
            raise ValueError("dynamic_bias requested_joints must be non-empty")
        motion.joints.indices(requested)
        if motion.velocity is None:
            raise ValueError(f"{motion.source} is missing measured velocity required for dynamic_bias")
        positions = np.asarray(motion.position.values, dtype=np.float32)
        velocities = np.asarray(motion.velocity.interpolate(motion.position.time).values, dtype=np.float32)
        if not len(positions):
            raise ValueError("dynamic_bias requires at least one trajectory sample")
        self._initialize()

        mapped: list[tuple[int, str, int, int]] = []
        output_index: dict[str, int] = {}
        owner: dict[int, str] = {}
        for column, name in enumerate(motion.joints.names):
            joint_index = self._resolve_joint(name)
            if joint_index is None:
                continue
            starts = self._scalar_starts(joint_index)
            mj_qd = int(self._mj_qd_start[joint_index])
            if starts is None or mj_qd < 0:
                if name in requested:
                    raise ValueError(f"requested joint {name!r} is not a scalar joint in the Newton model")
                continue
            if joint_index in owner:
                raise ValueError(f"trajectory joints {owner[joint_index]!r} and {name!r} map to the same Newton joint")
            owner[joint_index] = name
            mapped.append((column, name, starts[0], starts[1]))
            output_index[name] = mj_qd

        absent = [name for name in requested if name not in output_index]
        if absent:
            raise ValueError("requested joints not found in the Newton model: " + ", ".join(absent))
        columns = np.asarray([item[0] for item in mapped], dtype=np.int64)
        q_indices = np.asarray([item[2] for item in mapped], dtype=np.int64)
        qd_indices = np.asarray([item[3] for item in mapped], dtype=np.int64)
        requested_qd = np.asarray([output_index[name] for name in requested], dtype=np.int64)
        q = self._default_q.copy()
        qd = np.zeros(int(self._model.joint_dof_count), dtype=np.float32)
        values = np.empty((len(positions), len(requested)), dtype=np.float32)
        for row in range(len(positions)):
            q[q_indices] = positions[row, columns]
            qd[qd_indices] = velocities[row, columns]
            self._state.joint_q.assign(q)
            self._state.joint_qd.assign(qd)
            values[row] = _evaluate_dynamic_bias(self._solver, self._state)[requested_qd]
        return values
