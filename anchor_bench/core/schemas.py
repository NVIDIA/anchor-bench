from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import Field, field_validator, model_validator

from anchor_bench.core.config_model import (
    ConfigModel,
    JointSelection,
    RUNTIME_GRU_INPUT_COLUMNS,
    parse_gru_columns,
    parse_joint_selection,
)
from anchor_bench.core.experiment_schema import (
    ExperimentActuatorModel,
    ExperimentAnalysis,
    ExperimentAnalysisPlots,
    ExperimentBenchmark,
    ExperimentDataset,
    ExperimentFile,
    ExperimentOutputs,
)
from anchor_bench.core.run_schema import (
    AnalysisDefaults,
    BenchmarkDefaults,
    CMAESConfig,
    RobotRunFile,
    SimulationDefaults,
    SysIdDefaults,
    TrainingDefaults,
    TrainingFrictionBaseline,
    TrainingGroup,
    TrainingJointMapping,
    TrainingPDBaseline,
)


CONTROLLERS = {"pd", "pid", "mlp", "lstm", "gru", "fmu", "torque_passthrough"}
CLAMPS = {"effort", "velocity", "dc_motor", "position", "none"}


class ControllerSpec(ConfigModel):
    type: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    source: str | None = None
    network_file: str | None = None

    @field_validator("type")
    @classmethod
    def _validate_type(cls, value: str) -> str:
        if value not in CONTROLLERS:
            raise ValueError(f"has unsupported controller {value!r}")
        return value

    @field_validator("parameters", mode="before")
    @classmethod
    def _validate_parameters(cls, value: Any) -> dict[str, Any]:
        parameters = dict(value or {})
        if "input_columns" in parameters:
            parameters["input_columns"] = parse_gru_columns(
                parameters["input_columns"], allowed=RUNTIME_GRU_INPUT_COLUMNS
            )
        return parameters

    @model_validator(mode="after")
    def _require_gru_network(self) -> "ControllerSpec":
        if self.type == "gru" and not self.network_file:
            raise ValueError("type 'gru' requires network_file")
        return self


class ResidualSpec(ConfigModel):
    type: str
    network_file: str
    stats_file: str | None = None
    input_columns: list[str] = Field(default_factory=lambda: ["position", "position_error", "velocity"])
    output_column: str = "torque_residual"
    source: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("type")
    @classmethod
    def _validate_type(cls, value: str) -> str:
        if value != "gru":
            raise ValueError(f"{value!r} must be 'gru'")
        return value

    @field_validator("input_columns", mode="before")
    @classmethod
    def _validate_input_columns(cls, value: Any) -> list[str]:
        return parse_gru_columns(value, allowed=RUNTIME_GRU_INPUT_COLUMNS)


class DelaySpec(ConfigModel):
    duration_ms: float = Field(default=0.0, ge=0.0)


class ClampingSpec(ConfigModel):
    type: str
    parameters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("type")
    @classmethod
    def _validate_type(cls, value: str) -> str:
        if value not in CLAMPS:
            raise ValueError(f"has unsupported clamp {value!r}")
        return value


class ActuatorSpec(ConfigModel):
    name: str
    joints: list[str]
    controller: ControllerSpec
    delay: DelaySpec = Field(default_factory=DelaySpec)
    clamps: list[ClampingSpec] = Field(default_factory=list)
    physical_parameters: dict[str, Any] = Field(default_factory=dict)
    residual: ResidualSpec | None = None


class ActuatorFile(ConfigModel):
    schema_version: int = Field(ge=2)
    backend: str
    actuators: list[ActuatorSpec]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("backend")
    @classmethod
    def _validate_backend(cls, value: str) -> str:
        if value != "newton":
            raise ValueError("must be 'newton'")
        return value


class RobotSpec(ConfigModel):
    name: str
    display_name: str
    fixed_base: bool = True
    asset_path: str
    joint_names: list[str]
    joint_aliases: dict[str, str] = Field(default_factory=dict)
    actuator_file: str = "actuators.yaml"
    bounds_file: str = "bounds.yaml"
    ee_frame: str | None = None
    runtime_usd: str | None = None
    schema_version: int = Field(default=1, ge=1)

    @model_validator(mode="before")
    @classmethod
    def _default_display_name(cls, data: Any) -> Any:
        if isinstance(data, dict) and "display_name" not in data and "name" in data:
            return {**data, "display_name": data["name"]}
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RobotSpec":
        return super().from_dict(data, "robot")


class BoundsRange(ConfigModel):
    lower: float
    upper: float

    @model_validator(mode="after")
    def _validate_order(self) -> "BoundsRange":
        if self.upper < self.lower:
            raise ValueError(f"upper ({self.upper}) must be >= lower ({self.lower})")
        return self


class SysIdBoundsFile(ConfigModel):
    schema_version: int = Field(ge=2)
    mirror: bool = False
    joint_types: list[str] = Field(min_length=1)
    parameters: dict[str, BoundsRange] = Field(min_length=1)
    fixed: dict[str, float] = Field(default_factory=dict)


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
    groups: list[TrainingGroup] | None = None
    input_joints: JointSelection | str | None = None
    output_joints: JointSelection | str | None = None
    pd_kp: float | None = None
    pd_kd: float | None = None
    friction_dry: float | None = None
    friction_viscous: float | None = None
    input_columns: list[str] | str | None = None
    dry_run: bool = True
    skip_optuna: bool | None = None
    trials: int | None = None
    trial_epochs: int | None = None
    final_epochs: int | None = None
    n_streams: int | None = None
    noise_std: float | None = None
    seed: int | None = None
    sample_dt_s: float | None = None

    def __post_init__(self) -> None:
        input_joints = None if self.input_joints is None else parse_joint_selection(self.input_joints)
        output_joints = None if self.output_joints is None else parse_joint_selection(self.output_joints)
        selection = TrainingDefaults(
            groups=self.groups,
            input_joints=input_joints,
            output_joints=output_joints,
        )
        object.__setattr__(self, "groups", selection.groups)
        object.__setattr__(self, "input_joints", selection.input_joints)
        object.__setattr__(self, "output_joints", selection.output_joints)
        if self.input_columns is not None:
            object.__setattr__(
                self,
                "input_columns",
                parse_gru_columns(self.input_columns),
            )

    @property
    def resolved_groups(self) -> list[TrainingGroup]:
        input_joints = None if self.input_joints is None else parse_joint_selection(self.input_joints)
        output_joints = None if self.output_joints is None else parse_joint_selection(self.output_joints)
        return TrainingDefaults(
            groups=self.groups,
            input_joints=input_joints,
            output_joints=output_joints,
        ).resolved_groups


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


__all__ = [
    "ActuatorFile",
    "ActuatorSpec",
    "AnalysisDefaults",
    "BenchmarkDefaults",
    "BenchmarkRunConfig",
    "BoundsRange",
    "CMAESConfig",
    "ClampingSpec",
    "ControllerSpec",
    "DelaySpec",
    "ExperimentActuatorModel",
    "ExperimentAnalysis",
    "ExperimentAnalysisPlots",
    "ExperimentBenchmark",
    "ExperimentDataset",
    "ExperimentFile",
    "ExperimentOutputs",
    "ResidualSpec",
    "RobotRunFile",
    "RobotSpec",
    "SimulationDefaults",
    "SysIdBoundsFile",
    "SysIdDefaults",
    "SysIdRunConfig",
    "TrainingDefaults",
    "TrainingFrictionBaseline",
    "TrainingGroup",
    "TrainingJointMapping",
    "TrainingPDBaseline",
    "TrainActuatorRunConfig",
]
