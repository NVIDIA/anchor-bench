from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping, Protocol, Sequence

import numpy as np

if TYPE_CHECKING:
    from anchor_bench.robots.registry import RobotBundle


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
        object.__setattr__(
            self,
            "provenance",
            MappingProxyType(dict(self.provenance)),
        )


class DynamicBiasProvider(Protocol):
    """Training-side interface for evaluating bias on a synchronized episode."""

    def compute(
        self,
        *,
        joint_names: Sequence[str],
        positions: np.ndarray,
        velocities: np.ndarray,
        requested_joint_names: Sequence[str],
    ) -> DynamicBiasResult: ...


def resolve_runtime_asset(bundle: RobotBundle) -> Path:
    """Resolve the runtime asset without importing runtime helpers at module load."""

    from anchor_bench.runtime.assets import resolve_runtime_asset as resolve

    return resolve(bundle)


def _import_newton_runtime() -> tuple[Any, Any]:
    try:
        import newton
        import warp as wp
    except ImportError as exc:  # pragma: no cover - full-runtime dependency
        raise RuntimeError(
            "dynamic_bias training requires Newton, Warp, MuJoCo, and MuJoCo-Warp from the full runtime environment"
        ) from exc
    return newton, wp


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
    """Evaluate MuJoCo qfrc_bias without advancing the simulation."""

    def __init__(self, bundle: RobotBundle, *, batch_size: int = 64):
        if batch_size < 1:
            raise ValueError("dynamic-bias batch_size must be >= 1")
        self._bundle = bundle
        self._batch_size = int(batch_size)
        self._initialized = False
        self._newton: Any = None
        self._wp: Any = None
        self._template: Any = None
        self._model: Any = None
        self._solver: Any = None
        self._state: Any = None
        self._bias: Any = None
        self._coords_per_world = 0
        self._dofs_per_world = 0
        self._default_q = np.empty(0, dtype=np.float32)
        self._default_qd = np.empty(0, dtype=np.float32)
        self._mapping_cache: dict[str, int | None] = {}
        self._provenance: dict[str, Any] = {
            "source": "newton.solvers.SolverMuJoCo.eval_dynamic_bias",
            "quantity": "qfrc_bias",
            "batch_size": self._batch_size,
            "contacts": "disabled",
            "use_mujoco_cpu": False,
            "unrecorded_positions": "model_defaults",
            "unrecorded_velocities": "zero",
        }

    @property
    def provenance(self) -> dict[str, Any]:
        return dict(self._provenance)

    def _initialize(self) -> None:
        if self._initialized:
            return

        newton, wp = _import_newton_runtime()
        solver_namespace = getattr(newton, "solvers", None)
        solver_type = getattr(solver_namespace, "SolverMuJoCo", None)
        if solver_type is None or not hasattr(solver_type, "eval_dynamic_bias"):
            raise RuntimeError("dynamic_bias training requires Newton SolverMuJoCo.eval_dynamic_bias(state, out)")

        asset_path = resolve_runtime_asset(self._bundle)
        template = newton.ModelBuilder()
        solver_type.register_custom_attributes(template)
        template.add_usd(
            str(asset_path),
            floating=not self._bundle.spec.fixed_base,
            collapse_fixed_joints=True,
            enable_self_collisions=False,
            load_visual_shapes=False,
            skip_mesh_approximation=True,
        )

        scene = newton.ModelBuilder()
        scene.replicate(template, self._batch_size)
        model = scene.finalize()
        solver = solver_type(
            model,
            use_mujoco_cpu=False,
            disable_contacts=True,
        )

        self._newton = newton
        self._wp = wp
        self._template = template
        self._model = model
        self._solver = solver
        self._state = model.state()
        self._bias = wp.empty(
            model.joint_dof_count,
            dtype=wp.float32,
            device=model.device,
        )
        self._coords_per_world = int(template.joint_coord_count)
        self._dofs_per_world = int(template.joint_dof_count)
        self._default_q = model.joint_q.numpy().astype(np.float32, copy=True)
        self._default_qd = np.zeros(model.joint_dof_count, dtype=np.float32)
        self._provenance.update(
            {
                "asset": asset_path.name,
                "asset_sha256": _file_sha256(asset_path),
                "newton_version": str(getattr(newton, "__version__", "unknown")),
            }
        )
        self._initialized = True

    def _resolve_template_joint(self, requested_name: str) -> int | None:
        cached = self._mapping_cache.get(requested_name)
        if requested_name in self._mapping_cache:
            return cached
        assert self._template is not None

        exact = [
            index
            for index, label in enumerate(self._template.joint_label)
            if label == requested_name or label.endswith(f"/{requested_name}")
        ]
        if len(exact) > 1:
            raise ValueError(f"trajectory joint {requested_name!r} matches multiple Newton joints")
        if len(exact) == 1:
            resolved = exact[0]
        else:
            normalized = _normalized_joint_name(requested_name)
            matches = [
                index
                for index, label in enumerate(self._template.joint_label)
                if _normalized_joint_name(label) == normalized
            ]
            if len(matches) > 1:
                raise ValueError(f"trajectory joint {requested_name!r} matches multiple Newton joints")
            resolved = matches[0] if matches else None

        self._mapping_cache[requested_name] = resolved
        return resolved

    def _scalar_indices(
        self,
        name: str,
        *,
        required: bool,
    ) -> tuple[int, int] | None:
        joint_index = self._resolve_template_joint(name)
        if joint_index is None:
            if required:
                raise ValueError(f"input joint {name!r} was not found in the Newton model")
            return None
        assert self._template is not None
        assert self._newton is not None
        dof_count = sum(self._template.joint_dof_dim[joint_index])
        joint_type = self._template.joint_type[joint_index]
        if dof_count != 1 or joint_type not in (
            self._newton.JointType.REVOLUTE,
            self._newton.JointType.PRISMATIC,
        ):
            if required:
                raise ValueError(f"input joint {name!r} must map to one revolute or prismatic Newton DOF")
            return None
        return (
            int(self._template.joint_q_start[joint_index]),
            int(self._template.joint_qd_start[joint_index]),
        )

    def _defaulted_model_joints(self, mapped_indices: set[int]) -> list[str]:
        assert self._template is not None
        assert self._newton is not None
        defaulted: list[str] = []
        for index, label in enumerate(self._template.joint_label):
            dof_count = sum(self._template.joint_dof_dim[index])
            if (
                index not in mapped_indices
                and dof_count == 1
                and self._template.joint_type[index]
                in (self._newton.JointType.REVOLUTE, self._newton.JointType.PRISMATIC)
            ):
                defaulted.append(_normalized_joint_name(label))
        return defaulted

    @staticmethod
    def _record_unique_model_joint(
        mapped_model_names: dict[int, str],
        joint_index: int,
        name: str,
    ) -> None:
        existing_name = mapped_model_names.get(joint_index)
        if existing_name is not None:
            raise ValueError(
                f"trajectory joints {existing_name!r} and {name!r} map to the same Newton scalar joint/DOF"
            )
        mapped_model_names[joint_index] = name

    def compute(
        self,
        *,
        joint_names: Sequence[str],
        positions: np.ndarray,
        velocities: np.ndarray,
        requested_joint_names: Sequence[str],
    ) -> DynamicBiasResult:
        trajectory_names = tuple(str(name) for name in joint_names)
        requested_names = tuple(str(name) for name in requested_joint_names)
        if not trajectory_names:
            raise ValueError("dynamic_bias joint_names must not be empty")
        if len(set(trajectory_names)) != len(trajectory_names):
            raise ValueError("dynamic_bias joint_names must be unique")
        if not requested_names:
            raise ValueError("dynamic_bias requested_joint_names must not be empty")
        if len(set(requested_names)) != len(requested_names):
            raise ValueError("dynamic_bias requested_joint_names must be unique")
        missing = [name for name in requested_names if name not in trajectory_names]
        if missing:
            raise ValueError("dynamic_bias trajectory is missing input joints: " + ", ".join(missing))

        position_values = np.asarray(positions, dtype=np.float32)
        velocity_values = np.asarray(velocities, dtype=np.float32)
        expected_shape = (
            position_values.shape[0] if position_values.ndim == 2 else 0,
            len(trajectory_names),
        )
        if position_values.ndim != 2 or position_values.shape != expected_shape:
            raise ValueError(f"dynamic_bias positions must have shape [samples, {len(trajectory_names)}]")
        if velocity_values.shape != position_values.shape:
            raise ValueError("dynamic_bias velocities must have the same shape as positions")
        if not len(position_values):
            raise ValueError("dynamic_bias requires at least one trajectory sample")
        if not np.all(np.isfinite(position_values)) or not np.all(np.isfinite(velocity_values)):
            raise ValueError("dynamic_bias input positions and velocities must be finite")

        self._initialize()
        requested_set = set(requested_names)
        mapped: list[tuple[int, str, int, int]] = []
        mapped_model_indices: set[int] = set()
        mapped_model_names: dict[int, str] = {}
        unmapped_names: list[str] = []
        for column, name in enumerate(trajectory_names):
            indices = self._scalar_indices(name, required=name in requested_set)
            if indices is None:
                unmapped_names.append(name)
                continue
            joint_index = self._resolve_template_joint(name)
            assert joint_index is not None
            self._record_unique_model_joint(
                mapped_model_names,
                joint_index,
                name,
            )
            mapped_model_indices.add(joint_index)
            mapped.append((column, name, indices[0], indices[1]))

        q_indices_by_name = {name: (q_index, qd_index) for _, name, q_index, qd_index in mapped}
        requested_qd_indices = np.asarray(
            [q_indices_by_name[name][1] for name in requested_names],
            dtype=np.int64,
        )
        result = np.empty(
            (len(position_values), len(requested_names)),
            dtype=np.float32,
        )

        mapped_columns = np.asarray([item[0] for item in mapped], dtype=np.int64)
        mapped_q_indices = np.asarray([item[2] for item in mapped], dtype=np.int64)
        mapped_qd_indices = np.asarray([item[3] for item in mapped], dtype=np.int64)
        for start in range(0, len(position_values), self._batch_size):
            end = min(start + self._batch_size, len(position_values))
            active = end - start
            q_values = self._default_q.copy()
            qd_values = self._default_qd.copy()
            for world in range(active):
                q_base = world * self._coords_per_world
                qd_base = world * self._dofs_per_world
                q_values[q_base + mapped_q_indices] = position_values[
                    start + world,
                    mapped_columns,
                ]
                qd_values[qd_base + mapped_qd_indices] = velocity_values[
                    start + world,
                    mapped_columns,
                ]

            self._state.joint_q.assign(q_values)
            self._state.joint_qd.assign(qd_values)
            self._solver.eval_dynamic_bias(self._state, self._bias)
            bias = self._bias.numpy().reshape(
                self._batch_size,
                self._dofs_per_world,
            )
            result[start:end] = bias[:active, requested_qd_indices]

        if not np.all(np.isfinite(result)):
            raise ValueError("SolverMuJoCo produced non-finite dynamic_bias values")

        provenance = {
            **self._provenance,
            "trajectory_joint_names": list(trajectory_names),
            "requested_joint_names": list(requested_names),
            "mapped_joint_names": [item[1] for item in mapped],
            "unmapped_trajectory_joint_names": unmapped_names,
            "defaulted_joint_names": self._defaulted_model_joints(mapped_model_indices),
        }
        self._provenance = dict(provenance)
        return DynamicBiasResult(result, provenance)
