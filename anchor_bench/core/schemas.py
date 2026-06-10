
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


CONTROLLERS = {"pd", "pid", "mlp", "lstm", "gru", "fmu", "torque_passthrough"}
GRU_INPUT_COLUMNS = {"position", "position_error", "velocity", "solver_pd"}
CLAMPS = {"effort", "velocity", "dc_motor", "position", "none"}
EXPERIMENT_ACTUATOR_TYPES = CONTROLLERS | {"implicit"}
EXPERIMENT_BENCHMARK_PARAMS = {"initial", "fitted"}

_CONTROLLER_KNOWN_KEYS = {"type", "parameters", "source", "network_file"}
_RESIDUAL_KNOWN_KEYS = {"type", "network_file", "stats_file", "input_columns", "output_column", "source", "parameters"}
_DELAY_KNOWN_KEYS = {"duration_ms"}
_CLAMP_KNOWN_KEYS = {"type", "parameters"}
_ACTUATOR_SPEC_KNOWN_KEYS = {
    "name", "joints", "controller", "delay", "clamps",
    "physical_parameters", "residual",
}
_BOUNDS_RANGE_KNOWN_KEYS = {"lower", "upper"}


def _require(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ValueError(f"{context} missing required key '{key}'")
    return mapping[key]


def _parse_column_list(value: Any, context: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        columns = [item.strip() for item in value.split(",")]
    elif isinstance(value, (list, tuple)):
        columns = [str(item).strip() for item in value]
    else:
        raise ValueError(f"{context} must be a string or list of strings")
    columns = [column for column in columns if column]
    if not columns:
        raise ValueError(f"{context} must contain at least one column")
    return columns


def _validate_gru_columns(columns: Any, context: str) -> list[str]:
    parsed = _parse_column_list(columns, context)
    unsupported = sorted(set(parsed) - GRU_INPUT_COLUMNS)
    if unsupported:
        raise ValueError(f"unsupported GRU input columns: {', '.join(unsupported)}")
    return parsed


@dataclass(frozen=True)
class ControllerSpec:
    type: str
    parameters: dict[str, Any] = field(default_factory=dict)
    source: str | None = None
    network_file: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any], context: str) -> "ControllerSpec":
        unknown = sorted(set(data) - _CONTROLLER_KNOWN_KEYS)
        if unknown:
            raise ValueError(f"{context} has unknown keys: {', '.join(unknown)}")
        ctype = _require(data, "type", context)
        if ctype not in CONTROLLERS:
            raise ValueError(f"{context} has unsupported controller '{ctype}'")
        if ctype == "gru" and not data.get("network_file"):
            raise ValueError(f"{context}.type 'gru' requires network_file")
        parameters = dict(data.get("parameters", {}))
        if ctype == "gru" and "input_columns" in parameters:
            parameters["input_columns"] = _validate_gru_columns(parameters["input_columns"], f"{context}.parameters.input_columns")
        return cls(
            type=ctype,
            parameters=parameters,
            source=data.get("source"),
            network_file=data.get("network_file"),
        )


@dataclass(frozen=True)
class ResidualSpec:
    type: str
    network_file: str
    stats_file: str | None = None
    input_columns: list[str] = field(default_factory=lambda: ["position", "position_error", "velocity"])
    output_column: str = "torque_residual"
    source: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any], context: str) -> "ResidualSpec":
        unknown = sorted(set(data) - _RESIDUAL_KNOWN_KEYS)
        if unknown:
            raise ValueError(f"{context} has unknown keys: {', '.join(unknown)}")
        rtype = _require(data, "type", context)
        if rtype != "gru":
            raise ValueError(f"{context}.type {rtype!r} must be 'gru'")
        network_file = _require(data, "network_file", context)
        columns = _validate_gru_columns(
            data.get("input_columns", ["position", "position_error", "velocity"]),
            f"{context}.input_columns",
        )
        output_column = str(data.get("output_column", "torque_residual"))
        return cls(
            type=str(rtype),
            network_file=str(network_file),
            stats_file=data.get("stats_file"),
            input_columns=columns,
            output_column=output_column,
            source=data.get("source"),
            parameters=dict(data.get("parameters", {})),
        )


@dataclass(frozen=True)
class DelaySpec:
    duration_ms: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None, context: str = "delay") -> "DelaySpec":
        if not data:
            return cls()
        unknown = sorted(set(data) - _DELAY_KNOWN_KEYS)
        if unknown:
            raise ValueError(f"{context} has unknown keys: {', '.join(unknown)}")
        return cls(duration_ms=float(data.get("duration_ms", 0.0)))


@dataclass(frozen=True)
class ClampingSpec:
    type: str
    parameters: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any], context: str) -> "ClampingSpec":
        unknown = sorted(set(data) - _CLAMP_KNOWN_KEYS)
        if unknown:
            raise ValueError(f"{context} has unknown keys: {', '.join(unknown)}")
        ctype = _require(data, "type", context)
        if ctype not in CLAMPS:
            raise ValueError(f"{context} has unsupported clamp '{ctype}'")
        return cls(type=ctype, parameters=dict(data.get("parameters", {})))


@dataclass(frozen=True)
class ActuatorSpec:
    name: str
    joints: list[str]
    controller: ControllerSpec
    delay: DelaySpec = field(default_factory=DelaySpec)
    clamps: list[ClampingSpec] = field(default_factory=list)
    physical_parameters: dict[str, Any] = field(default_factory=dict)
    residual: ResidualSpec | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any], context: str) -> "ActuatorSpec":
        if "model_type" in data:
            raise ValueError(f"{context} uses legacy model_type")
        unknown = sorted(set(data) - _ACTUATOR_SPEC_KNOWN_KEYS)
        if unknown:
            raise ValueError(f"{context} has unknown keys: {', '.join(unknown)}")
        return cls(
            name=_require(data, "name", context),
            joints=list(_require(data, "joints", context)),
            controller=ControllerSpec.from_dict(_require(data, "controller", context), f"{context}.controller"),
            delay=DelaySpec.from_dict(data.get("delay"), f"{context}.delay"),
            clamps=[
                ClampingSpec.from_dict(item, f"{context}.clamps[{idx}]")
                for idx, item in enumerate(data.get("clamps", []))
            ],
            physical_parameters=dict(data.get("physical_parameters", {})),
            residual=ResidualSpec.from_dict(data["residual"], f"{context}.residual") if data.get("residual") else None,
        )


@dataclass(frozen=True)
class ActuatorFile:
    schema_version: int
    backend: str
    actuators: list[ActuatorSpec]

    @classmethod
    def from_dict(cls, data: dict[str, Any], source: str | Path) -> "ActuatorFile":
        text_context = str(source)
        if "model_type" in data:
            raise ValueError(f"{text_context} uses legacy model_type")
        schema_version = int(_require(data, "schema_version", text_context))
        if schema_version < 2:
            raise ValueError(f"{text_context} schema_version must be >= 2")
        backend = _require(data, "backend", text_context)
        if backend != "newton":
            raise ValueError(f"{text_context} backend must be 'newton'")
        actuators = [
            ActuatorSpec.from_dict(item, f"{text_context}.actuators[{idx}]")
            for idx, item in enumerate(_require(data, "actuators", text_context))
        ]
        return cls(schema_version=schema_version, backend=backend, actuators=actuators)


_ROBOT_KNOWN_KEYS = {
    "schema_version", "name", "display_name", "fixed_base", "asset_path",
    "joint_names", "joint_aliases", "actuator_file", "bounds_file",
    "valid_joints_file", "ee_frame", "runtime_usd",
}


@dataclass(frozen=True)
class RobotSpec:
    name: str
    display_name: str
    fixed_base: bool
    asset_path: str
    joint_names: list[str]
    joint_aliases: dict[str, str]
    actuator_file: str
    bounds_file: str
    valid_joints_file: str
    ee_frame: str | None = None
    runtime_usd: str | None = None
    schema_version: int = 1

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RobotSpec":
        if "model_type" in data:
            raise ValueError("robot uses legacy model_type")
        unknown = sorted(set(data) - _ROBOT_KNOWN_KEYS)
        if unknown:
            raise ValueError(f"robot has unknown keys: {', '.join(unknown)}")
        schema_version = int(_require(data, "schema_version", "robot"))
        if schema_version < 1:
            raise ValueError("robot schema_version must be >= 1")
        joints = list(_require(data, "joint_names", "robot"))
        return cls(
            name=_require(data, "name", "robot"),
            display_name=data.get("display_name", data["name"]),
            fixed_base=bool(data.get("fixed_base", True)),
            asset_path=_require(data, "asset_path", "robot"),
            joint_names=joints,
            joint_aliases=dict(data.get("joint_aliases", {})),
            actuator_file=data.get("actuator_file", "actuators.yaml"),
            bounds_file=data.get("bounds_file", "bounds.yaml"),
            valid_joints_file=data.get("valid_joints_file", "valid_joints.txt"),
            ee_frame=data.get("ee_frame"),
            runtime_usd=data.get("runtime_usd"),
            schema_version=schema_version,
        )


@dataclass(frozen=True)
class BoundsRange:
    lower: float
    upper: float

    @classmethod
    def from_dict(cls, data: dict[str, Any], context: str) -> "BoundsRange":
        unknown = sorted(set(data) - _BOUNDS_RANGE_KNOWN_KEYS)
        if unknown:
            raise ValueError(f"{context} has unknown keys: {', '.join(unknown)}")
        lower = float(_require(data, "lower", context))
        upper = float(_require(data, "upper", context))
        if upper < lower:
            raise ValueError(f"{context}: upper ({upper}) must be >= lower ({lower})")
        return cls(lower=lower, upper=upper)


_CMAES_KNOWN_KEYS = {"sigma", "max_iterations", "epsilon"}


@dataclass(frozen=True)
class CMAESConfig:
    sigma: float = 0.5
    max_iterations: int = 200
    epsilon: float | None = 0.01

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None, context: str) -> "CMAESConfig":
        if not data:
            return cls()
        unknown = sorted(set(data) - _CMAES_KNOWN_KEYS)
        if unknown:
            raise ValueError(f"{context} has unknown keys: {', '.join(unknown)}")
        return cls(
            sigma=float(data.get("sigma", 0.5)),
            max_iterations=int(data.get("max_iterations", 200)),
            epsilon=float(data["epsilon"]) if "epsilon" in data else 0.01,
        )


_BOUNDS_KNOWN_KEYS = {"schema_version", "mirror", "joint_types", "parameters", "fixed"}


@dataclass(frozen=True)
class SysIdBoundsFile:
    schema_version: int
    mirror: bool
    joint_types: list[str]
    parameters: dict[str, BoundsRange]
    fixed: dict[str, float]

    @classmethod
    def from_dict(cls, data: dict[str, Any], source: str | Path) -> "SysIdBoundsFile":
        text_context = str(source)
        if "model_type" in data:
            raise ValueError(f"{text_context} uses legacy model_type")
        unknown = sorted(set(data) - _BOUNDS_KNOWN_KEYS)
        if unknown:
            raise ValueError(f"{text_context} has unknown keys: {', '.join(unknown)}")
        schema_version = int(_require(data, "schema_version", text_context))
        if schema_version < 2:
            raise ValueError(f"{text_context} schema_version must be >= 2")
        parameters_raw = _require(data, "parameters", text_context)
        if not isinstance(parameters_raw, dict) or not parameters_raw:
            raise ValueError(f"{text_context}.parameters must be a non-empty mapping")
        parameters = {
            str(name): BoundsRange.from_dict(value, f"{text_context}.parameters.{name}")
            for name, value in parameters_raw.items()
        }
        joint_types = list(_require(data, "joint_types", text_context))
        if not joint_types:
            raise ValueError(f"{text_context}.joint_types must be non-empty")
        return cls(
            schema_version=schema_version,
            mirror=bool(data.get("mirror", False)),
            joint_types=joint_types,
            parameters=parameters,
            fixed={str(k): float(v) for k, v in (data.get("fixed") or {}).items()},
        )


_RUN_KNOWN_KEYS = {"simulation", "benchmark", "analysis", "sysid", "training"}
_RUN_SIMULATION_KEYS = {
    "physics_freq", "control_freq", "render_freq", "integrator",
    "solver_iterations", "line_search_iterations", "ls_iterations",
}
_RUN_BENCHMARK_KEYS = {
    "dataset", "fix_root", "buffer_time", "motion_files", "motion_name",
    "output_folder", "real_init_pose_sync",
}
_RUN_ANALYSIS_KEYS = {"output_dir", "sample_dt", "motion_names"}
_RUN_SYSID_KEYS = {
    "output_dir", "real_data_dir", "joints", "num_envs", "buffer_time", "cmaes",
}
_RUN_SYSID_FLAT_CMAES_KEYS = ("sigma", "epsilon", "max_iter")
_TRAINING_MODELS = {"gru", "hybrid_residual"}
_RUN_TRAINING_KEYS = {
    "model", "real_data_dir", "output_dir", "joint_type", "skip_optuna",
    "trials", "trial_epochs", "final_epochs", "n_streams", "noise_std", "seed",
    "input_columns", "target_column",
}


def _optional_mapping_section(
    data: dict[str, Any],
    section: str,
    source: str,
    known_keys: set[str],
) -> dict[str, Any]:
    if section not in data:
        return {}
    value = data.get(section) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{source}.{section} must be a mapping")
    unknown = sorted(set(value) - known_keys)
    if unknown:
        raise ValueError(f"{source}.{section} has unknown keys: {', '.join(unknown)}")
    return dict(value)


def _validate_training_section(training: dict[str, Any], source: str) -> dict[str, Any]:
    if not training:
        return training
    resolved = dict(training)
    model = str(resolved.get("model", "gru"))
    if model not in _TRAINING_MODELS:
        raise ValueError(f"{source}.training.model {model!r} must be one of {sorted(_TRAINING_MODELS)}")
    resolved["model"] = model
    if "skip_optuna" in resolved and not isinstance(resolved["skip_optuna"], bool):
        raise ValueError(f"{source}.training.skip_optuna must be a boolean")
    for key in ("trials", "trial_epochs", "final_epochs", "n_streams"):
        if key in resolved:
            value = int(resolved[key])
            if value < 1:
                raise ValueError(f"{source}.training.{key} must be >= 1")
            resolved[key] = value
    if "noise_std" in resolved:
        noise_std = float(resolved["noise_std"])
        if noise_std < 0.0:
            raise ValueError(f"{source}.training.noise_std must be >= 0")
        resolved["noise_std"] = noise_std
    if "seed" in resolved:
        resolved["seed"] = int(resolved["seed"])
    if "input_columns" in resolved:
        resolved["input_columns"] = _validate_gru_columns(resolved["input_columns"], f"{source}.training.input_columns")
    if "target_column" in resolved:
        resolved["target_column"] = str(resolved["target_column"])
    return resolved


@dataclass(frozen=True)
class RobotRunFile:
    simulation: dict[str, Any] = field(default_factory=dict)
    benchmark: dict[str, Any] = field(default_factory=dict)
    analysis: dict[str, Any] = field(default_factory=dict)
    sysid: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any], source: str | Path) -> "RobotRunFile":
        text_context = str(source)
        if not isinstance(data, dict):
            raise ValueError(f"{text_context} must be a mapping")
        unknown = sorted(set(data) - _RUN_KNOWN_KEYS)
        if unknown:
            raise ValueError(f"{text_context} has unknown keys: {', '.join(unknown)}")

        simulation = _optional_mapping_section(data, "simulation", text_context, _RUN_SIMULATION_KEYS)
        benchmark = _optional_mapping_section(data, "benchmark", text_context, _RUN_BENCHMARK_KEYS)
        analysis = _optional_mapping_section(data, "analysis", text_context, _RUN_ANALYSIS_KEYS)
        sysid = _optional_mapping_section(data, "sysid", text_context, _RUN_SYSID_KEYS | set(_RUN_SYSID_FLAT_CMAES_KEYS))
        training = _optional_mapping_section(data, "training", text_context, _RUN_TRAINING_KEYS)
        training = _validate_training_section(training, text_context)

        for key in _RUN_SYSID_FLAT_CMAES_KEYS:
            if key in sysid:
                raise ValueError(f"{text_context}.sysid uses unsupported flat CMA-ES key '{key}'")

        if sysid:
            cmaes = CMAESConfig.from_dict(sysid.get("cmaes"), f"{text_context}.sysid.cmaes")
            sysid["cmaes"] = {
                "sigma": cmaes.sigma,
                "max_iterations": cmaes.max_iterations,
                "epsilon": cmaes.epsilon,
            }

        return cls(simulation=simulation, benchmark=benchmark, analysis=analysis, sysid=sysid, training=training)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key in ("simulation", "benchmark", "analysis", "sysid", "training"):
            value = getattr(self, key)
            if value:
                result[key] = dict(value)
        return result


_EXPERIMENT_KNOWN_KEYS = {
    "schema_version", "robot", "outputs", "seeds", "benchmark",
    "datasets", "actuator_models", "analysis",
}
_EXPERIMENT_OUTPUTS_KEYS = {"benchmark", "analysis"}
_EXPERIMENT_BENCHMARK_KEYS = {"num_envs", "headless"}
_EXPERIMENT_DATASET_KEYS = {"real_root", "motions"}
_EXPERIMENT_ACTUATOR_MODEL_KEYS = {
    "type", "initial_params", "fitted_params", "benchmark_params",
}
_EXPERIMENT_ANALYSIS_KEYS = {
    "seeds", "datasets", "motions", "actuator_models", "plots",
}
_EXPERIMENT_ANALYSIS_PLOT_KEYS = {"joints", "end_effector", "require_end_effector"}


@dataclass(frozen=True)
class ExperimentActuatorModel:
    type: str
    initial_params: str | None = None
    fitted_params: str | None = None
    benchmark_params: str = "fitted"

    @classmethod
    def from_dict(cls, data: dict[str, Any], context: str) -> "ExperimentActuatorModel":
        if "model_type" in data:
            raise ValueError(f"{context} uses legacy model_type")
        unknown = sorted(set(data) - _EXPERIMENT_ACTUATOR_MODEL_KEYS)
        if unknown:
            raise ValueError(f"{context} has unknown keys: {', '.join(unknown)}")
        actuator_type = _require(data, "type", context)
        if actuator_type not in EXPERIMENT_ACTUATOR_TYPES:
            raise ValueError(
                f"{context}.type {actuator_type!r} must be one of "
                f"{sorted(EXPERIMENT_ACTUATOR_TYPES)}"
            )
        benchmark_params = str(data.get("benchmark_params", "fitted"))
        if benchmark_params not in EXPERIMENT_BENCHMARK_PARAMS:
            raise ValueError(
                f"{context}.benchmark_params {benchmark_params!r} must be one of "
                f"{sorted(EXPERIMENT_BENCHMARK_PARAMS)}"
            )
        return cls(
            type=actuator_type,
            initial_params=data.get("initial_params"),
            fitted_params=data.get("fitted_params"),
            benchmark_params=benchmark_params,
        )


@dataclass(frozen=True)
class ExperimentFile:
    schema_version: int
    robot: str

    @classmethod
    def from_dict(cls, data: dict[str, Any], source: str | Path) -> "ExperimentFile":
        text_context = str(source)
        if "model_type" in data:
            raise ValueError(f"{text_context} uses legacy model_type")
        unknown = sorted(set(data) - _EXPERIMENT_KNOWN_KEYS)
        if unknown:
            raise ValueError(f"{text_context} has unknown keys: {', '.join(unknown)}")
        schema_version = int(_require(data, "schema_version", text_context))
        if schema_version < 1:
            raise ValueError(f"{text_context} schema_version must be >= 1")
        robot = str(_require(data, "robot", text_context))

        outputs = data.get("outputs")
        if outputs is not None:
            if not isinstance(outputs, dict):
                raise ValueError(f"{text_context}.outputs must be a mapping")
            outputs_unknown = sorted(set(outputs) - _EXPERIMENT_OUTPUTS_KEYS)
            if outputs_unknown:
                raise ValueError(
                    f"{text_context}.outputs has unknown keys: "
                    f"{', '.join(outputs_unknown)}"
                )

        benchmark = data.get("benchmark")
        if benchmark is not None:
            if not isinstance(benchmark, dict):
                raise ValueError(f"{text_context}.benchmark must be a mapping")
            benchmark_unknown = sorted(set(benchmark) - _EXPERIMENT_BENCHMARK_KEYS)
            if benchmark_unknown:
                raise ValueError(
                    f"{text_context}.benchmark has unknown keys: "
                    f"{', '.join(benchmark_unknown)}"
                )
            if "num_envs" in benchmark and int(benchmark["num_envs"]) < 1:
                raise ValueError(f"{text_context}.benchmark.num_envs must be >= 1")
            if "headless" in benchmark and not isinstance(benchmark["headless"], bool):
                raise ValueError(f"{text_context}.benchmark.headless must be a boolean")

        datasets = data.get("datasets") or {}
        if not isinstance(datasets, dict) or not datasets:
            raise ValueError(
                f"{text_context}.datasets must be a non-empty mapping"
            )
        for name, dataset in datasets.items():
            ctx = f"{text_context}.datasets.{name}"
            if not isinstance(dataset, dict):
                raise ValueError(f"{ctx} must be a mapping")
            ds_unknown = sorted(set(dataset) - _EXPERIMENT_DATASET_KEYS)
            if ds_unknown:
                raise ValueError(f"{ctx} has unknown keys: {', '.join(ds_unknown)}")
            _require(dataset, "real_root", ctx)

        actuator_models = data.get("actuator_models") or {}
        if not isinstance(actuator_models, dict) or not actuator_models:
            raise ValueError(
                f"{text_context}.actuator_models must be a non-empty mapping"
            )
        for name, model in actuator_models.items():
            ExperimentActuatorModel.from_dict(
                model or {}, f"{text_context}.actuator_models.{name}"
            )

        analysis = data.get("analysis")
        if analysis is not None:
            if not isinstance(analysis, dict):
                raise ValueError(f"{text_context}.analysis must be a mapping")
            analysis_unknown = sorted(set(analysis) - _EXPERIMENT_ANALYSIS_KEYS)
            if analysis_unknown:
                raise ValueError(
                    f"{text_context}.analysis has unknown keys: "
                    f"{', '.join(analysis_unknown)}"
                )
            plots = analysis.get("plots")
            if plots is not None:
                if not isinstance(plots, dict):
                    raise ValueError(f"{text_context}.analysis.plots must be a mapping")
                plots_unknown = sorted(set(plots) - _EXPERIMENT_ANALYSIS_PLOT_KEYS)
                if plots_unknown:
                    raise ValueError(
                        f"{text_context}.analysis.plots has unknown keys: "
                        f"{', '.join(plots_unknown)}"
                    )
                for key, value in plots.items():
                    if not isinstance(value, bool):
                        raise ValueError(f"{text_context}.analysis.plots.{key} must be a boolean")

        return cls(schema_version=schema_version, robot=robot)


@dataclass(frozen=True)
class SysIdRunConfig:
    robot: str
    data: str | None = None
    output: str | None = None
    dry_run: bool = True
    headless: bool = False
    num_envs: int | None = None
    max_iter: int | None = None
    max_trajectory_len: int | None = None
    seed: int | None = None


@dataclass(frozen=True)
class TrainActuatorRunConfig:
    robot: str
    model: str | None = None
    data: str | None = None
    output: str | None = None
    joint_type: str | None = None
    input_columns: list[str] | None = None
    target_column: str | None = None
    dry_run: bool = True
    skip_optuna: bool | None = None
    trials: int | None = None
    trial_epochs: int | None = None
    final_epochs: int | None = None
    n_streams: int | None = None
    noise_std: float | None = None
    seed: int | None = None


@dataclass(frozen=True)
class BenchmarkRunConfig:
    robot: str
    motion: str | None = None
    output: str | None = None
    dry_run: bool = True
    headless: bool = False
    motion_name: str | None = None
    actuator_params: str | None = None
    dataset: str | None = None
    actuator_model: str | None = None
    seed: int | None = None
