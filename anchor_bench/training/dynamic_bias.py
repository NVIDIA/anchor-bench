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
        cache_key: str | None = None,
    ) -> DynamicBiasResult: ...


def resolve_runtime_asset(bundle: RobotBundle) -> Path:
    """Resolve the runtime asset without importing runtime helpers at module load."""

    from anchor_bench.runtime.assets import resolve_runtime_asset as resolve

    return resolve(bundle)


def _import_newton_runtime() -> tuple[Any, Any]:
    try:
        import mujoco
        import newton
    except ImportError as exc:  # pragma: no cover - full-runtime dependency
        raise RuntimeError(
            "dynamic_bias training requires Newton and MuJoCo from the full runtime environment"
        ) from exc
    return newton, mujoco


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

    def __init__(self, bundle: RobotBundle):
        self._bundle = bundle
        self._initialized = False
        self._newton: Any = None
        self._mujoco: Any = None
        self._template: Any = None
        self._solver: Any = None
        self._mj_q_start = np.empty(0, dtype=np.int64)
        self._mj_qd_start = np.empty(0, dtype=np.int64)
        self._dof_ref = np.empty(0, dtype=np.float64)
        self._default_qpos = np.empty(0, dtype=np.float64)
        self._mapping_cache: dict[str, int | None] = {}
        self._trajectory_cache: dict[str, tuple[tuple[str, ...], tuple[int, ...], np.ndarray]] = {}
        self._provenance: dict[str, Any] = {
            "source": "mujoco.mj_forward.qfrc_bias",
            "model_source": "newton.solvers.SolverMuJoCo(use_mujoco_cpu=True)",
            "quantity": "qfrc_bias",
            "contacts": "disabled",
            "use_mujoco_cpu": True,
            "unrecorded_positions": "model_defaults",
            "unrecorded_velocities": "zero",
        }

    def _initialize(self) -> None:
        if self._initialized:
            return

        newton, mujoco = _import_newton_runtime()
        solver_namespace = getattr(newton, "solvers", None)
        solver_type = getattr(solver_namespace, "SolverMuJoCo", None)
        if solver_type is None:
            raise RuntimeError("dynamic_bias training requires newton.solvers.SolverMuJoCo")

        asset_path = resolve_runtime_asset(self._bundle)
        template = newton.ModelBuilder()
        register = getattr(solver_type, "register_custom_attributes", None)
        if callable(register):
            register(template)
        template.add_usd(
            str(asset_path),
            floating=not self._bundle.spec.fixed_base,
            collapse_fixed_joints=True,
            enable_self_collisions=False,
            load_visual_shapes=False,
            skip_mesh_approximation=True,
        )
        model = template.finalize()
        solver = solver_type(
            model,
            use_mujoco_cpu=True,
            disable_contacts=True,
        )
        missing = [
            name
            for name in ("mj_model", "mj_data", "mj_q_start", "mj_qd_start")
            if getattr(solver, name, None) is None
        ]
        if missing:
            raise RuntimeError(
                "dynamic_bias training requires SolverMuJoCo MuJoCo model access; missing: "
                + ", ".join(missing)
            )

        mujoco_attrs = getattr(model, "mujoco", None)
        dof_ref = getattr(mujoco_attrs, "dof_ref", None) if mujoco_attrs is not None else None

        self._newton = newton
        self._mujoco = mujoco
        self._template = template
        self._solver = solver
        self._mj_q_start = np.asarray(solver.mj_q_start.numpy(), dtype=np.int64)
        self._mj_qd_start = np.asarray(solver.mj_qd_start.numpy(), dtype=np.int64)
        self._dof_ref = (
            np.zeros(int(model.joint_dof_count), dtype=np.float64)
            if dof_ref is None
            else np.asarray(dof_ref.numpy(), dtype=np.float64)
        )
        self._default_qpos = np.array(solver.mj_data.qpos, dtype=np.float64, copy=True)
        self._provenance.update(
            {
                "asset": asset_path.name,
                "asset_sha256": _file_sha256(asset_path),
                "newton_version": str(getattr(newton, "__version__", "unknown")),
                "mujoco_version": str(getattr(mujoco, "__version__", "unknown")),
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
    ) -> tuple[int, int, int] | None:
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
        if int(self._mj_q_start[joint_index]) < 0:
            # Loop-closure joints carry no MuJoCo coordinates; treat as unmapped.
            if required:
                raise ValueError(f"input joint {name!r} has no MuJoCo coordinates in the compiled model")
            return None
        return (
            int(self._mj_q_start[joint_index]),
            int(self._mj_qd_start[joint_index]),
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

    def _trajectory_bias_table(
        self,
        cache_key: str | None,
        trajectory_names: tuple[str, ...],
        position_values: np.ndarray,
        velocity_values: np.ndarray,
        *,
        mapped_columns: np.ndarray,
        mapped_q_adr: np.ndarray,
        mapped_qd_adr: np.ndarray,
        mapped_dof_ref: np.ndarray,
    ) -> np.ndarray:
        """Evaluate qfrc_bias for every mapped trajectory joint, once per key."""

        shape = tuple(int(dim) for dim in position_values.shape)
        if cache_key is not None:
            cached = self._trajectory_cache.get(cache_key)
            if cached is not None and cached[0] == trajectory_names and cached[1] == shape:
                return cached[2]

        mj_model = self._solver.mj_model
        mj_data = self._solver.mj_data
        mj_data.qpos[:] = self._default_qpos
        mj_data.qvel[:] = 0.0
        table = np.empty((len(position_values), len(mapped_columns)), dtype=np.float32)
        for row in range(len(position_values)):
            mj_data.qpos[mapped_q_adr] = position_values[row, mapped_columns] + mapped_dof_ref
            mj_data.qvel[mapped_qd_adr] = velocity_values[row, mapped_columns]
            self._mujoco.mj_forward(mj_model, mj_data)
            table[row] = mj_data.qfrc_bias[mapped_qd_adr]

        if not np.all(np.isfinite(table)):
            raise ValueError("SolverMuJoCo produced non-finite dynamic_bias values")

        if cache_key is not None:
            table.setflags(write=False)
            self._trajectory_cache[cache_key] = (trajectory_names, shape, table)
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
        mapped: list[tuple[int, str, int, int, int]] = []
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
            existing_name = mapped_model_names.get(joint_index)
            if existing_name is not None:
                raise ValueError(
                    f"trajectory joints {existing_name!r} and {name!r} map to the same Newton scalar joint/DOF"
                )
            mapped_model_names[joint_index] = name
            mapped_model_indices.add(joint_index)
            mapped.append((column, name, indices[0], indices[1], indices[2]))

        # The full trajectory sweep depends only on the source's mapped joints,
        # so evaluate every mapped joint once and slice the requested columns.
        # requested_names are all required and therefore always present in mapped.
        mapped_names = [item[1] for item in mapped]
        table_column_by_name = {name: index for index, name in enumerate(mapped_names)}
        requested_columns_in_table = np.asarray(
            [table_column_by_name[name] for name in requested_names],
            dtype=np.int64,
        )

        table = self._trajectory_bias_table(
            cache_key,
            trajectory_names,
            position_values,
            velocity_values,
            mapped_columns=np.asarray([item[0] for item in mapped], dtype=np.int64),
            mapped_q_adr=np.asarray([item[2] for item in mapped], dtype=np.int64),
            mapped_qd_adr=np.asarray([item[3] for item in mapped], dtype=np.int64),
            mapped_dof_ref=self._dof_ref[np.asarray([item[4] for item in mapped], dtype=np.int64)],
        )
        result = np.array(table[:, requested_columns_in_table], dtype=np.float32)

        provenance = {
            **self._provenance,
            "trajectory_joint_names": list(trajectory_names),
            "requested_joint_names": list(requested_names),
            "mapped_joint_names": [item[1] for item in mapped],
            "unmapped_trajectory_joint_names": unmapped_names,
            "defaulted_joint_names": self._defaulted_model_joints(mapped_model_indices),
        }
        return DynamicBiasResult(result, provenance)
