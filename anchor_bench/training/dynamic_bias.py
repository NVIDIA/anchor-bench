from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np

from anchor_bench.runtime.assets import resolve_runtime_asset

if TYPE_CHECKING:
    from anchor_bench.robots.registry import RobotBundle


def _normalized_joint_name(value: str) -> str:
    name = value.rsplit("/", 1)[-1]
    return name.removesuffix("_joint")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class NewtonMuJoCoDynamicBiasProvider:
    """Evaluate MuJoCo qfrc_bias for recorded states without advancing simulation."""

    def __init__(self, bundle: RobotBundle, *, batch_size: int = 64):
        if batch_size < 1:
            raise ValueError("dynamic-bias batch_size must be >= 1")

        try:
            import newton
            import warp as wp
        except ImportError as exc:  # pragma: no cover - full-runtime dependency
            raise RuntimeError(
                "dynamic_bias training requires Newton, Warp, MuJoCo, and MuJoCo-Warp "
                "from the full runtime environment"
            ) from exc

        solver_type = newton.solvers.SolverMuJoCo
        if not hasattr(solver_type, "eval_dynamic_bias"):
            raise RuntimeError(
                "dynamic_bias training requires a Newton SolverMuJoCo with "
                "eval_dynamic_bias(state, out); install the matching Newton feature branch or release"
            )

        asset_path = resolve_runtime_asset(bundle)
        template = newton.ModelBuilder()
        solver_type.register_custom_attributes(template)
        template.add_usd(
            str(asset_path),
            floating=not bundle.spec.fixed_base,
            collapse_fixed_joints=True,
            enable_self_collisions=False,
            load_visual_shapes=False,
            skip_mesh_approximation=True,
        )

        scene = newton.ModelBuilder()
        scene.replicate(template, batch_size)
        model = scene.finalize()
        solver = solver_type(
            model,
            use_mujoco_cpu=False,
            disable_contacts=True,
        )

        self._newton = newton
        self._template = template
        self._model = model
        self._solver = solver
        self._state = model.state()
        self._bias = wp.empty(
            model.joint_dof_count,
            dtype=wp.float32,
            device=model.device,
        )
        self._batch_size = batch_size
        self._required_joint_names = tuple(
            _normalized_joint_name(name) for name in bundle.spec.joint_names
        )
        self._coords_per_world = template.joint_coord_count
        self._dofs_per_world = template.joint_dof_count
        self._default_q = model.joint_q.numpy().astype(np.float32, copy=True)
        # A trajectory only supplies velocities for its recorded joints. Keep all
        # other model DOFs stationary while preserving their configured positions.
        self._default_qd = np.zeros(model.joint_dof_count, dtype=np.float32)
        self._mapping_cache: dict[tuple[str, ...], tuple[np.ndarray, np.ndarray]] = {}
        self.provenance = {
            "source": "newton.solvers.SolverMuJoCo.qfrc_bias",
            "asset_sha256": _file_sha256(asset_path),
            "asset": asset_path.name,
            "batch_size": batch_size,
            "contacts": "disabled",
            "unrecorded_positions": "model_defaults",
            "unrecorded_velocities": "zero",
            "newton_version": str(getattr(newton, "__version__", "unknown")),
        }

    def _resolve_template_joint(self, requested_name: str) -> int:
        exact = [
            index
            for index, label in enumerate(self._template.joint_label)
            if label == requested_name or label.endswith(f"/{requested_name}")
        ]
        if len(exact) == 1:
            return exact[0]

        normalized = _normalized_joint_name(requested_name)
        matches = [
            index
            for index, label in enumerate(self._template.joint_label)
            if _normalized_joint_name(label) == normalized
        ]
        if len(matches) != 1:
            labels = ", ".join(self._template.joint_label)
            if not matches:
                raise ValueError(
                    f"trajectory joint {requested_name!r} was not found in the Newton model; "
                    f"available joints: {labels}"
                )
            raise ValueError(
                f"trajectory joint {requested_name!r} matches multiple Newton joints: "
                + ", ".join(self._template.joint_label[index] for index in matches)
            )
        return matches[0]

    def _indices_for(self, joint_names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        key = tuple(str(name) for name in joint_names)
        cached = self._mapping_cache.get(key)
        if cached is not None:
            return cached

        q_indices: list[int] = []
        qd_indices: list[int] = []
        for name in key:
            joint_index = self._resolve_template_joint(name)
            dof_count = sum(self._template.joint_dof_dim[joint_index])
            joint_type = self._template.joint_type[joint_index]
            if dof_count != 1 or joint_type not in (
                self._newton.JointType.REVOLUTE,
                self._newton.JointType.PRISMATIC,
            ):
                raise ValueError(
                    f"trajectory joint {name!r} must map to one revolute or prismatic Newton DOF"
                )
            q_indices.append(int(self._template.joint_q_start[joint_index]))
            qd_indices.append(int(self._template.joint_qd_start[joint_index]))

        resolved = (
            np.asarray(q_indices, dtype=np.int64),
            np.asarray(qd_indices, dtype=np.int64),
        )
        self._mapping_cache[key] = resolved
        return resolved

    def compute(
        self,
        *,
        joint_names: Sequence[str],
        positions: np.ndarray,
        velocities: np.ndarray,
    ) -> np.ndarray:
        position_values = np.asarray(positions, dtype=np.float32)
        velocity_values = np.asarray(velocities, dtype=np.float32)
        if position_values.ndim != 2:
            raise ValueError(
                f"dynamic_bias positions must have shape [samples, {len(joint_names)}]"
            )
        expected_shape = (position_values.shape[0], len(joint_names))
        if position_values.shape != expected_shape:
            raise ValueError(
                f"dynamic_bias positions must have shape [samples, {len(joint_names)}]"
            )
        if velocity_values.shape != position_values.shape:
            raise ValueError(
                "dynamic_bias velocities must have the same shape as positions"
            )

        normalized_names = tuple(
            _normalized_joint_name(str(name)) for name in joint_names
        )
        if len(set(normalized_names)) != len(normalized_names):
            raise ValueError("dynamic_bias joint_names must be unique")
        available = set(normalized_names)
        missing = [name for name in self._required_joint_names if name not in available]
        if missing:
            raise ValueError(
                "dynamic_bias trajectory is missing configured robot joints: "
                + ", ".join(missing)
            )

        if (
            not np.isfinite(position_values).all()
            or not np.isfinite(velocity_values).all()
        ):
            raise ValueError(
                "dynamic_bias input positions and velocities must be finite"
            )
        if position_values.shape[0] == 0:
            raise ValueError("dynamic_bias requires at least one trajectory sample")

        q_indices, qd_indices = self._indices_for(joint_names)
        result = np.empty(position_values.shape, dtype=np.float32)

        for start in range(0, len(position_values), self._batch_size):
            end = min(start + self._batch_size, len(position_values))
            active = end - start
            q_values = self._default_q.copy()
            qd_values = self._default_qd.copy()
            for world in range(active):
                q_base = world * self._coords_per_world
                qd_base = world * self._dofs_per_world
                q_values[q_base + q_indices] = position_values[start + world]
                qd_values[qd_base + qd_indices] = velocity_values[start + world]

            self._state.joint_q.assign(q_values)
            self._state.joint_qd.assign(qd_values)
            self._solver.eval_dynamic_bias(self._state, self._bias)
            bias = self._bias.numpy().reshape(self._batch_size, self._dofs_per_world)
            result[start:end] = bias[:active, qd_indices]

        if not np.isfinite(result).all():
            raise ValueError("SolverMuJoCo produced non-finite dynamic_bias values")
        return result
