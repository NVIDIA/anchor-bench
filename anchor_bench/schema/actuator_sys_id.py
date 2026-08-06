"""Pydantic model for ``actuator_sys_id_schema.yaml``."""

from __future__ import annotations

from math import isclose
from typing import Annotated, List, Literal, Optional, Union

from pydantic import Field, StrictFloat, StrictInt, field_validator, model_validator
from typing_extensions import TypeAlias

from anchor_bench.schema.base import BaseConfig, StrictBaseModel
from anchor_bench.schema.dataset import ParquetDatasetConfig

JsonNumber: TypeAlias = Union[StrictInt, StrictFloat]
NonEmptyString: TypeAlias = Annotated[str, Field(min_length=1)]
PositiveInteger: TypeAlias = Annotated[StrictInt, Field(ge=1)]
CmaesPopulationSize: TypeAlias = Annotated[StrictInt, Field(ge=2)]
NonNegativeInteger: TypeAlias = Annotated[StrictInt, Field(ge=0)]
PositiveNumber: TypeAlias = Annotated[JsonNumber, Field(gt=0.0)]
NonNegativeNumber: TypeAlias = Annotated[JsonNumber, Field(ge=0.0)]


class CmaesOptimizerConfig(StrictBaseModel):
    type: Literal["python-cmaes"]
    sigma: PositiveNumber
    max_iterations: PositiveInteger
    epsilon: PositiveNumber


class ControllerClamps(StrictBaseModel):
    effort_limit: NonNegativeNumber
    velocity_limit: NonNegativeNumber


class Bound(StrictBaseModel):
    lower: JsonNumber
    upper: JsonNumber
    initial: Optional[JsonNumber] = None

    @model_validator(mode="after")
    def validate_interval(self) -> Bound:
        if self.lower > self.upper:
            raise ValueError("lower must be less than or equal to upper")
        if self.initial is not None and not self.lower <= self.initial <= self.upper:
            raise ValueError("initial must lie between lower and upper")
        return self


class PdControllerConfig(StrictBaseModel):
    type: Literal["pd"]
    delay_duration_ms: NonNegativeInteger = 0
    clamps: ControllerClamps
    stiffness: Bound
    damping: Bound


class JointParameters(StrictBaseModel):
    dry_friction: Bound
    viscous_friction: Bound
    armature: Bound


class JointParameterGroup(StrictBaseModel):
    names: Annotated[List[NonEmptyString], Field(min_length=1)]
    controller: PdControllerConfig
    parameters: JointParameters

    @field_validator("names")
    @classmethod
    def names_must_be_unique(cls, names: List[str]) -> List[str]:
        if len(names) != len(set(names)):
            raise ValueError("joint names must be unique within a parameter group")
        return names

    @model_validator(mode="after")
    def physical_parameter_bounds_must_be_nonnegative(self) -> JointParameterGroup:
        bounds = {
            "stiffness": self.controller.stiffness,
            "damping": self.controller.damping,
            "armature": self.parameters.armature,
            "dry_friction": self.parameters.dry_friction,
            "viscous_friction": self.parameters.viscous_friction,
        }
        negative = [name for name, bound in bounds.items() if bound.lower < 0]
        if negative:
            raise ValueError(
                "actuator parameter bounds must be nonnegative: " + ", ".join(negative)
            )
        return self


class WarpMujocoSolverConfig(StrictBaseModel):
    type: Literal["warp_mujoco"]
    integrator: NonEmptyString = "implicitfast"
    solver_iterations: PositiveInteger
    line_search_iterations: NonNegativeInteger


class NewtonSimulationConfig(StrictBaseModel):
    type: Literal["newton"]
    usd_path: NonEmptyString
    physics_freq: PositiveInteger
    control_freq: PositiveInteger
    render_freq: PositiveInteger
    solver: WarpMujocoSolverConfig

    @model_validator(mode="after")
    def validate_frequencies(self) -> NewtonSimulationConfig:
        if self.physics_freq % self.control_freq != 0:
            raise ValueError("physics_freq must be divisible by control_freq")
        return self


class ActuatorSysIdConfig(BaseConfig):
    type: Literal["actuator_sys_id"]  # pyright: ignore[reportIncompatibleVariableOverride]
    joint_parameters: Annotated[List[JointParameterGroup], Field(min_length=1)]
    optimizer: CmaesOptimizerConfig
    simulation: NewtonSimulationConfig
    num_envs: CmaesPopulationSize = 64
    buffer_time: NonNegativeNumber = 0.0
    seed: NonNegativeInteger = 0
    output_dir: NonEmptyString = "outputs/actuator_sys_id"
    dataset: ParquetDatasetConfig

    @model_validator(mode="after")
    def validate_task_invariants(self) -> ActuatorSysIdConfig:
        names = [
            name
            for parameter_group in self.joint_parameters
            for name in parameter_group.names
        ]
        if len(names) != len(set(names)):
            raise ValueError("joint names must be unique across parameter groups")
        dataset_dt = float(self.dataset.resampling.dt)
        control_dt = 1.0 / self.simulation.control_freq
        if not isclose(dataset_dt, control_dt, rel_tol=1.0e-9, abs_tol=1.0e-12):
            raise ValueError(
                "dataset resampling dt must match the simulation control timestep: "
                f"{dataset_dt} != {control_dt}"
            )
        return self
