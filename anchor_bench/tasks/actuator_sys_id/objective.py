"""Objective that scores normalized actuator candidates."""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

from anchor_bench.tasks.actuator_sys_id.model import Candidate, Trajectory
from anchor_bench.tasks.actuator_sys_id.optimizer import FloatArray
from anchor_bench.tasks.actuator_sys_id.parameter_space import ParameterSpace


class PopulationEvaluator(Protocol):
    """Score decoded actuator candidates against recorded trajectories."""

    def evaluate(
        self,
        population: Sequence[Candidate],
        trajectories: Sequence[Trajectory],
    ) -> FloatArray: ...


class ActuatorSysIdObjective:
    """Decode optimizer vectors and return trajectory errors."""

    def __init__(
        self,
        parameter_space: ParameterSpace,
        evaluator: PopulationEvaluator,
        trajectories: Sequence[Trajectory],
    ) -> None:
        if not trajectories:
            raise ValueError("system identification requires trajectories")
        self._parameter_space = parameter_space
        self._evaluator = evaluator
        self._trajectories = tuple(trajectories)

    def evaluate(self, population: FloatArray) -> FloatArray:
        candidates = self._parameter_space.decode_population(population)
        return np.asarray(
            self._evaluator.evaluate(candidates, self._trajectories),
            dtype=np.float64,
        )
