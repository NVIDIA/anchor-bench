from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import Field, StrictBool, field_validator, model_validator

from anchor_bench.core.config_model import (
    ConfigModel,
    JointSelection,
    parse_gru_columns,
    parse_joint_selection,
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


class TrainingJointMapping(ConfigModel):
    input_joints: JointSelection
    output_joints: JointSelection | None = None

    @field_validator("input_joints", "output_joints", mode="before")
    @classmethod
    def _parse_joint_selection(cls, value: Any) -> JointSelection | None:
        return None if value is None else parse_joint_selection(value)


class TrainingGroup(ConfigModel):
    name: str
    mappings: list[TrainingJointMapping] = Field(min_length=1)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        name = value.strip()
        if not name:
            raise ValueError("must not be empty")
        return name

    @model_validator(mode="after")
    def _reject_duplicate_mappings(self) -> "TrainingGroup":
        seen: dict[tuple[tuple[str, ...], tuple[str, ...]], int] = {}
        for index, mapping in enumerate(self.mappings):
            inputs = ("all",) if mapping.input_joints == "all" else tuple(mapping.input_joints)
            outputs = mapping.output_joints or mapping.input_joints
            output_key = ("all",) if outputs == "all" else tuple(outputs)
            key = (inputs, output_key)
            if key in seen:
                raise ValueError(f"contains duplicate mappings at indices {seen[key]} and {index}")
            seen[key] = index
        return self


class TrainingDefaults(ConfigModel):
    model: Literal["gru", "hybrid_residual"] | None = None
    real_data_dir: str | None = None
    output_dir: str | None = None
    groups: list[TrainingGroup] | None = Field(default=None, min_length=1)
    input_joints: JointSelection | None = None
    output_joints: JointSelection | None = None
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
    sample_dt_s: float | None = Field(default=None, gt=0.0)

    _REMOVED_KEY_HINTS: ClassVar[dict[str, str]] = {
        "joint_type": "declare training.groups or training.input_joints/output_joints instead",
        "target_column": "the training target is always measured torque",
    }

    @model_validator(mode="before")
    @classmethod
    def _reject_removed_keys(cls, data: Any) -> Any:
        if isinstance(data, Mapping):
            removed = [key for key in cls._REMOVED_KEY_HINTS if key in data]
            if removed:
                key = removed[0]
                raise ValueError(
                    f"'{key}' is no longer supported; {cls._REMOVED_KEY_HINTS[key]}"
                )
        return data

    @field_validator("input_columns", mode="before")
    @classmethod
    def _validate_input_columns(cls, value: Any) -> list[str] | None:
        return None if value is None else parse_gru_columns(value)

    @field_validator("input_joints", "output_joints", mode="before")
    @classmethod
    def _parse_joint_selection(cls, value: Any) -> JointSelection | None:
        return None if value is None else parse_joint_selection(value)

    @model_validator(mode="after")
    def _validate_joint_selection(self) -> "TrainingDefaults":
        if self.groups is not None and (self.input_joints is not None or self.output_joints is not None):
            raise ValueError("groups cannot be combined with input_joints or output_joints")
        if self.output_joints is not None and self.input_joints is None:
            raise ValueError("input_joints is required when output_joints is configured")
        if self.groups is not None:
            names = [group.name for group in self.groups]
            duplicates = sorted({name for name in names if names.count(name) > 1})
            if duplicates:
                raise ValueError(f"duplicate training group names: {', '.join(duplicates)}")
        return self

    @property
    def resolved_groups(self) -> list[TrainingGroup]:
        if self.groups is not None:
            return self.groups
        if self.input_joints is None:
            return []
        return [
            TrainingGroup(
                name="default",
                mappings=[
                    TrainingJointMapping(
                        input_joints=self.input_joints,
                        output_joints=self.output_joints,
                    )
                ],
            )
        ]


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
        return {name: payload for name, section in sections.items() if (payload := section.to_dict())}
