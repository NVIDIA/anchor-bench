# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend-neutral system-identification simulation configuration."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, JsonValue, StrictInt, model_validator

from anchor_bench.schema.base import StrictBaseModel

__all__ = [
    "JsonValue",
    "NewtonSimulationConfig",
    "SysIdSimulationConfig",
    "WarpMujocoSolverConfig",
]

PositiveInteger = Annotated[StrictInt, Field(ge=1)]
NonNegativeInteger = Annotated[StrictInt, Field(ge=0)]
NonEmptyString = Annotated[str, Field(min_length=1)]


class SysIdSimulationConfig(StrictBaseModel):
    """Timing shared by all registered system-identification backends."""

    type: NonEmptyString
    physics_freq: PositiveInteger
    control_freq: PositiveInteger
    render_freq: PositiveInteger

    @model_validator(mode="after")
    def validate_frequencies(self) -> SysIdSimulationConfig:
        if self.physics_freq % self.control_freq != 0:
            raise ValueError("physics_freq must be divisible by control_freq")
        return self


class WarpMujocoSolverConfig(StrictBaseModel):
    type: Literal["warp_mujoco"]
    integrator: NonEmptyString = "implicitfast"
    solver_iterations: PositiveInteger
    line_search_iterations: NonNegativeInteger


class NewtonSimulationConfig(SysIdSimulationConfig):
    type: Literal["newton"]  # pyright: ignore[reportIncompatibleVariableOverride]
    usd_path: NonEmptyString
    solver: WarpMujocoSolverConfig
