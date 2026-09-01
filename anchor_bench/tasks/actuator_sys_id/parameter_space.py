# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic actuator parameter-vector encoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, cast

import numpy as np

from anchor_bench.schema.actuator_sys_id import ActuatorSysIdConfig, Bound
from anchor_bench.schema.base import BaseConfig
from anchor_bench.schema.helpers import _config_type
from anchor_bench.tasks.actuator_sys_id.model import Candidate


@dataclass(frozen=True)
class Parameter:
    """One scalar physical parameter in the optimizer vector."""

    target: str
    name: str
    lower: float
    upper: float
    initial: float


class ParameterSpace:
    """Map bounded physical parameters to the optimizer's ``[-1, 1]`` space."""

    def __init__(self, parameters: List[Parameter]) -> None:
        if not parameters:
            raise ValueError("actuator system identification requires parameters")
        keys = [(item.target, item.name) for item in parameters]
        if len(keys) != len(set(keys)):
            raise ValueError("parameter space contains duplicate target parameters")
        for item in parameters:
            values = np.asarray(
                [item.lower, item.upper, item.initial],
                dtype=np.float64,
            )
            if not np.isfinite(values).all():
                raise ValueError("parameter bounds and initials must be finite")
            if item.lower > item.upper:
                raise ValueError("parameter lower bound must not exceed upper bound")
            if not item.lower <= item.initial <= item.upper:
                raise ValueError("parameter initial value must lie within its bounds")
        self.parameters = list(parameters)
        self.lower = np.asarray([item.lower for item in parameters], dtype=np.float64)
        self.upper = np.asarray([item.upper for item in parameters], dtype=np.float64)
        self.initial = np.asarray(
            [item.initial for item in parameters], dtype=np.float64
        )

    @property
    def dimension(self) -> int:
        return len(self.parameters)

    @property
    def initial_normalized(self) -> np.ndarray:
        return self.normalize(self.initial)

    def normalize(self, physical: np.ndarray) -> np.ndarray:
        values = np.asarray(physical, dtype=np.float64)
        self._validate_width(values)
        if not np.isfinite(values).all():
            raise ValueError("physical parameters must be finite")
        if np.any(values < self.lower) or np.any(values > self.upper):
            raise ValueError("physical parameters must lie within their bounds")
        span = self.upper - self.lower
        unit = np.divide(
            values - self.lower,
            span,
            out=np.full_like(values, 0.5),
            where=span != 0.0,
        )
        return unit * 2.0 - 1.0

    def denormalize(self, normalized: np.ndarray) -> np.ndarray:
        values = np.asarray(normalized, dtype=np.float64)
        self._validate_width(values)
        if not np.isfinite(values).all():
            raise ValueError("optimizer candidates must be finite")
        if np.any(values < -1.0) or np.any(values > 1.0):
            raise ValueError("optimizer candidates must lie within [-1, 1]")
        return self.lower + ((values + 1.0) / 2.0) * (self.upper - self.lower)

    def decode(self, normalized: np.ndarray) -> Candidate:
        physical = self.denormalize(normalized)
        if physical.ndim != 1:
            raise ValueError("decode expects one parameter vector")
        candidate: Candidate = {}
        for parameter, value in zip(self.parameters, physical):
            candidate.setdefault(parameter.target, {})[parameter.name] = float(value)
        return candidate

    def decode_population(self, normalized: np.ndarray) -> List[Candidate]:
        population = np.asarray(normalized, dtype=np.float64)
        if population.ndim != 2:
            raise ValueError("population must be a two-dimensional matrix")
        return [self.decode(row) for row in population]

    def _validate_width(self, values: np.ndarray) -> None:
        if values.ndim == 0:
            raise ValueError("parameter values must have at least one dimension")
        if values.shape[-1] != self.dimension:
            raise ValueError(
                f"expected parameter width {self.dimension}, got {values.shape[-1]}"
            )


def create_parameter_space(config: BaseConfig) -> ParameterSpace:
    """Create the parameter space selected by the config's string discriminator."""

    if config.type == _config_type(ActuatorSysIdConfig):
        actuator_config = cast(ActuatorSysIdConfig, config)
        parameters: List[Parameter] = []
        for group in actuator_config.joint_parameters:
            bounds = {
                "stiffness": group.controller.stiffness,
                "damping": group.controller.damping,
                "armature": group.parameters.armature,
                "dry_friction": group.parameters.dry_friction,
                "viscous_friction": group.parameters.viscous_friction,
            }
            for joint in group.names:
                for name, bound in bounds.items():
                    parameters.append(_create_parameter(joint, name, bound))
        for body in actuator_config.rigid_body_parameters:
            parameters.append(_create_parameter(body.name, "mass", body.mass))
        return ParameterSpace(parameters)
    raise ValueError(f"Unsupported parameter-space config type: {config.type!r}")


def _create_parameter(target: str, name: str, bound: Bound) -> Parameter:
    lower = float(bound.lower)
    upper = float(bound.upper)
    initial = (
        float(bound.initial)
        if bound.initial is not None
        else lower + (upper - lower) / 2.0
    )
    return Parameter(target, name, lower, upper, initial)
