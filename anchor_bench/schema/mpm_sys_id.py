# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validated MPM experiment, material, and search configuration (SI units)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Dict, List, Literal, Optional, Union

from pydantic import Field, model_validator

from anchor_bench.schema.base import BaseConfig, StrictBaseModel
from anchor_bench.schema.dataset import JsonNumber, NonEmptyString, PositiveNumber
from anchor_bench.schema.mlops import MLOpsConfig

NonNegative = Annotated[JsonNumber, Field(ge=0)]
PositiveInt = Annotated[int, Field(gt=0)]
FIXED = {
    "young_modulus": 9576000.0,
    "poisson_ratio": 0.25,
    "hardening": 0.0,
    "hardening_rate": 1.0,
    "softening_rate": 1.0,
    "dilatancy": 0.0,
    "viscosity": 0.0,
}

DEFAULT_BOUNDS: Dict[str, tuple[float, float, Literal["linear", "log"], float]] = {
    "damping": (0.0, 0.1, "log", 0.0001),
    "friction": (0.1, 1.5, "linear", 0.0),
    "yield_pressure": (1000.0, 1e12, "log", 0.0),
    "tensile_yield_ratio": (0.0, 1.0, "linear", 0.0),
    "yield_stress": (0.0, 1e5, "log", 100.0),
}


class Bound(StrictBaseModel):
    lower: NonNegative
    upper: NonNegative
    scale: Literal["linear", "log"] = "linear"
    offset: NonNegative = 0.0

    @model_validator(mode="after")
    def validate_bounds(self) -> Bound:
        if self.lower >= self.upper:
            raise ValueError("lower must be less than upper")
        if self.scale == "log" and self.lower + self.offset <= 0:
            raise ValueError("log bounds require lower + offset > 0")
        return self


class Material(StrictBaseModel):
    fixed: Dict[str, JsonNumber] = Field(default_factory=lambda: dict(FIXED))
    optimized: Dict[str, Bound] = Field(
        default_factory=lambda: {
            name: Bound(lower=lo, upper=hi, scale=scale, offset=offset)
            for name, (lo, hi, scale, offset) in DEFAULT_BOUNDS.items()
        }
    )

    @model_validator(mode="after")
    def validate_material(self) -> Material:
        if self.fixed != FIXED:
            raise ValueError(
                "Fixed elastic and plasticity parameters must match the supported sand model"
            )
        if set(self.optimized) != set(DEFAULT_BOUNDS):
            raise ValueError(
                "Optimize exactly damping, friction, yield_pressure, tensile_yield_ratio, yield_stress"
            )
        if (
            self.optimized["yield_pressure"].lower <= 0
            or self.optimized["tensile_yield_ratio"].upper > 1
        ):
            raise ValueError(
                "yield_pressure must be positive and tensile_yield_ratio at most 1"
            )
        return self


class CsvInput(StrictBaseModel):
    type: Literal["csv"] = "csv"
    source: NonEmptyString


class SimulationSettings(StrictBaseModel):
    drop_height_m: PositiveNumber
    density: PositiveNumber = 1600.0
    voxel_size: PositiveNumber = 0.02
    fps: PositiveInt = 60
    substeps: PositiveInt = 8
    num_frames: PositiveInt = 720
    ground_friction: NonNegative = 0.6


class SandPour(SimulationSettings):
    type: Literal["sand_pour"]
    sand_volume_m3: PositiveNumber = 0.01892705892
    bucket_radius: PositiveNumber = 0.15
    funnel_height: PositiveNumber = 0.20
    outlet_diameter: PositiveNumber = 0.10
    funnel_friction: NonNegative = 0.25

    @model_validator(mode="after")
    def resolved_outlet(self) -> SandPour:
        if not 4 * self.voxel_size <= self.outlet_diameter < 2 * self.bucket_radius:
            raise ValueError(
                "Outlet must span four voxels and be smaller than the bucket"
            )
        return self


class BottleDrop(SimulationSettings):
    type: Literal["bottle_drop"]
    water_volume_m3: PositiveNumber
    voxel_size: PositiveNumber = 0.015
    bed_width: PositiveNumber = 0.45
    bed_length: PositiveNumber = 0.45
    bed_depth: PositiveNumber = 0.30
    bottle_radius: PositiveNumber = 0.04
    bottle_height: PositiveNumber = 0.25
    cap_radius: PositiveNumber = 0.015
    cap_height: PositiveNumber = 0.015
    bottle_friction: NonNegative = 0.3
    pre_settle_seconds: PositiveNumber = 0.5
    impact_settle_seconds: PositiveNumber = 3.0
    lift_speed: PositiveNumber = 0.25
    lift_distance: PositiveNumber = 0.6
    crater_threshold: PositiveNumber = 0.003

    @model_validator(mode="after")
    def resolved_geometry(self) -> BottleDrop:
        if (
            self.bed_depth < max(0.30, 4 * self.voxel_size)
            or min(self.bed_width, self.bed_length) < 0.4
        ):
            raise ValueError(
                "Use a bed at least 0.30 m deep, four voxels deep, and 0.4 m wide"
            )
        if self.voxel_size > self.cap_radius:
            raise ValueError(
                "Resolve the cap with at least two voxels across its diameter"
            )
        if (
            self.bottle_height <= self.cap_height + 0.04
            or self.bottle_radius <= self.cap_radius
        ):
            raise ValueError(
                "Bottle must be taller and wider than its cap and shoulder"
            )
        return self


class Experiment(StrictBaseModel):
    id: Annotated[str, Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")]
    enabled: bool = True
    weight: PositiveNumber = 1.0
    input_data: CsvInput
    simulation: Annotated[Union[SandPour, BottleDrop], Field(discriminator="type")]


class Execution(StrictBaseModel):
    device: Annotated[str, Field(pattern=r"^cuda:[0-9]+$")] = "cuda:0"
    seed: Annotated[int, Field(ge=0)] = 48
    cuda_graphs: bool = True
    worker_timeout_s: PositiveNumber = 1800.0


class Stopping(StrictBaseModel):
    mode: Literal["continuous", "bounded"] = "bounded"
    max_evaluations: Optional[PositiveInt] = None
    target_error: Optional[NonNegative] = None

    @model_validator(mode="after")
    def limits(self) -> Stopping:
        if self.mode == "bounded" and self.max_evaluations is None:
            raise ValueError("Bounded search requires max_evaluations")
        if self.mode == "continuous" and (
            self.max_evaluations is not None or self.target_error is not None
        ):
            raise ValueError("Continuous search does not accept stopping limits")
        return self


class Optimizer(StrictBaseModel):
    type: Literal["random_forest"] = "random_forest"
    initial_samples: Annotated[int, Field(ge=2)] = 16
    pool_size: Annotated[int, Field(ge=16)] = 4096
    exploration: NonNegative = 1.0
    stopping: Stopping = Field(default_factory=lambda: Stopping(max_evaluations=64))


class Objective(StrictBaseModel):
    type: Literal["relative_rms"] = "relative_rms"
    target_reduction: Literal["mean"] = "mean"


class MpmSysIdConfig(BaseConfig):
    # Narrow the task discriminator, matching the other task schema models.
    type: Literal["mpm_sys_id"]  # pyright: ignore[reportIncompatibleVariableOverride]
    schema_version: Literal[1] = 1
    output_dir: Optional[NonEmptyString] = None
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    mlops: Optional[MLOpsConfig] = None
    execution: Execution = Field(default_factory=Execution)
    material: Material = Field(default_factory=Material)
    optimizer: Optimizer = Field(default_factory=Optimizer)
    objective: Objective = Field(default_factory=Objective)
    experiments: Annotated[List[Experiment], Field(min_length=1)]

    @model_validator(mode="after")
    def selected_experiments(self) -> MpmSysIdConfig:
        if len({e.id for e in self.experiments}) != len(self.experiments):
            raise ValueError("Experiment IDs must be unique")
        if not any(e.enabled for e in self.experiments):
            raise ValueError("Enable at least one experiment")
        return self

    def resolve_sources(self, directory: Path) -> None:
        for experiment in self.experiments:
            source = Path(experiment.input_data.source).expanduser()
            if not source.is_absolute():
                source = directory / source
            experiment.input_data.source = str(source.resolve())
