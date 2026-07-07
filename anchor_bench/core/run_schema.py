from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import Field, StrictBool, field_validator, model_validator

from anchor_bench.core.config_model import (
    ConfigModel,
    parse_gru_training_columns,
    parse_joint_names,
)


class CMAESConfig(ConfigModel):
    sigma: float = Field(default=0.5, gt=0.0)
    max_iterations: int = Field(default=200, ge=1)
    epsilon: float | None = Field(default=0.01, ge=0.0)


class SimulationDefaults(ConfigModel):
    physics_freq: int | None = Field(default=None, ge=1)
    control_freq: int | None = Field(default=None, ge=1)
    render_freq: int | None = Field(default=None, ge=1)
    integrator: str | None = None
    solver_iterations: int | None = Field(default=None, ge=1)
    line_search_iterations: int | None = Field(default=None, ge=1)


class BenchmarkDefaults(ConfigModel):
    dataset: str | None = None
    fix_root: StrictBool | None = None
    buffer_time: float | None = Field(default=None, ge=0.0)
    motion_files: str | None = None
    motion_name: str | None = None
    output_folder: str | None = None
    real_init_pose_sync: StrictBool | None = None


class AnalysisDefaults(ConfigModel):
    output_dir: str | None = None
    sample_dt: float | None = Field(default=None, ge=0.0)
    motion_names: str | list[str] | None = None

    @field_validator("motion_names", mode="before")
    @classmethod
    def _validate_motion_names(cls, value: Any) -> Any:
        if value is None or isinstance(value, str):
            return value
        if not isinstance(value, list):
            raise ValueError("must be a string or list")
        return [str(item) for item in value]


class SysIdDefaults(ConfigModel):
    output_dir: str | None = None
    real_data_dir: str | None = None
    joints: str | list[str] | None = None
    num_envs: int | None = Field(default=None, ge=1)
    buffer_time: float | None = Field(default=None, ge=0.0)
    cmaes: CMAESConfig | None = None

    @field_validator("joints", mode="before")
    @classmethod
    def _validate_joints(cls, value: Any) -> Any:
        if value is None or isinstance(value, str):
            return value
        if not isinstance(value, list):
            raise ValueError("must be a string or list")
        return [str(item) for item in value]


class TrainingPDBaseline(ConfigModel):
    kp: float = Field(ge=0.0)
    kd: float = Field(ge=0.0)


class TrainingFrictionBaseline(ConfigModel):
    dry: float = Field(ge=0.0)
    viscous: float = Field(ge=0.0)


class TrainingDefaults(ConfigModel):
    model: Literal["gru", "hybrid_residual"] | None = None
    real_data_dir: str | None = None
    output_dir: str | None = None
    joint_type: str | None = None
    input_joints: list[str] | None = None
    output_joints: list[str] | None = None
    pd_baseline: TrainingPDBaseline | None = None
    friction_baseline: TrainingFrictionBaseline | None = None
    skip_optuna: StrictBool | None = None
    trials: int | None = Field(default=None, ge=1)
    trial_epochs: int | None = Field(default=None, ge=1)
    final_epochs: int | None = Field(default=None, ge=1)
    n_streams: int | None = Field(default=None, ge=1)
    noise_std: float | None = Field(default=None, ge=0.0)
    seed: int | None = None
    input_columns: list[str] | None = None
    target_column: str | None = None

    @field_validator("input_columns", mode="before")
    @classmethod
    def _validate_input_columns(cls, value: Any) -> list[str] | None:
        return None if value is None else parse_gru_training_columns(value)

    @field_validator("input_joints", "output_joints", mode="before")
    @classmethod
    def _validate_joint_names(cls, value: Any) -> list[str] | None:
        return None if value is None else parse_joint_names(value)

    @model_validator(mode="after")
    def _validate_joint_selection(self) -> "TrainingDefaults":
        if (
            self.joint_type is not None
            and (self.input_joints is not None or self.output_joints is not None)
        ):
            raise ValueError("joint_type cannot be combined with input_joints or output_joints")
        if self.output_joints is not None and self.input_joints is None:
            raise ValueError("input_joints is required when output_joints is configured")
        return self


class RobotRunFile(ConfigModel):
    simulation: SimulationDefaults = Field(default_factory=SimulationDefaults)
    benchmark: BenchmarkDefaults = Field(default_factory=BenchmarkDefaults)
    analysis: AnalysisDefaults = Field(default_factory=AnalysisDefaults)
    sysid: SysIdDefaults = Field(default_factory=SysIdDefaults)
    training: TrainingDefaults = Field(default_factory=TrainingDefaults)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None, source: str | Path) -> "RobotRunFile":
        return super().from_dict(data, source)

    def to_dict(self) -> dict[str, Any]:
        sections = {
            "simulation": self.simulation,
            "benchmark": self.benchmark,
            "analysis": self.analysis,
            "sysid": self.sysid,
            "training": self.training,
        }
        return {
            name: payload
            for name, section in sections.items()
            if (payload := section.to_dict())
        }
