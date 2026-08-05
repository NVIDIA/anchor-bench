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
    """One scalar actuator parameter in the optimizer vector."""

    joint: str
    name: str
    lower: float
    upper: float
    initial: float


class ParameterSpace:
    """Map bounded physical parameters to the optimizer's ``[-1, 1]`` space."""

    def __init__(self, parameters: List[Parameter]) -> None:
        if not parameters:
            raise ValueError("actuator system identification requires parameters")
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
        if values.shape[-1] != self.dimension:
            raise ValueError(
                f"expected parameter width {self.dimension}, got {values.shape[-1]}"
            )
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
        if values.shape[-1] != self.dimension:
            raise ValueError(
                f"expected parameter width {self.dimension}, got {values.shape[-1]}"
            )
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
            candidate.setdefault(parameter.joint, {})[parameter.name] = float(value)
        return candidate

    def decode_population(self, normalized: np.ndarray) -> List[Candidate]:
        population = np.asarray(normalized, dtype=np.float64)
        if population.ndim != 2:
            raise ValueError("population must be a two-dimensional matrix")
        return [self.decode(row) for row in population]


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
        return ParameterSpace(parameters)
    raise ValueError(f"Unsupported parameter-space config type: {config.type!r}")


def _create_parameter(joint: str, name: str, bound: Bound) -> Parameter:
    lower = float(bound.lower)
    upper = float(bound.upper)
    initial = (
        float(bound.initial)
        if bound.initial is not None
        else lower + (upper - lower) / 2.0
    )
    return Parameter(joint, name, lower, upper, initial)
