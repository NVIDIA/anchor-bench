# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Objective-driven optimizer interface and CMA-ES implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import cmaes
import numpy as np
from numpy.typing import NDArray

from anchor_bench.schema.actuator_sys_id import (
    ActuatorSysIdConfig,
    CmaesOptimizerConfig,
)
from anchor_bench.schema.helpers import _config_type
from anchor_bench.tasks.actuator_sys_id.parameter_space import ParameterSpace

MINIMUM_CONVERGENCE_GENERATIONS = 5
IMPROVEMENT_WINDOW = 5

FloatArray = NDArray[np.float64]


class Objective(Protocol):
    """Evaluate a population of optimizer candidates."""

    def evaluate(self, population: FloatArray) -> FloatArray: ...


@dataclass(frozen=True)
class OptimizationStep:
    """Framework-neutral metrics produced by one optimizer iteration."""

    iteration: int
    scores: FloatArray
    best_score: float
    converged: bool

    @property
    def iteration_best_score(self) -> float:
        return float(np.min(self.scores))

    @property
    def mean_score(self) -> float:
        return float(np.mean(self.scores))


class Optimizer(Protocol):
    @property
    def converged(self) -> bool: ...

    @property
    def best_candidate(self) -> FloatArray: ...

    @property
    def best_score(self) -> float: ...

    def step(self, objective: Objective) -> OptimizationStep: ...


class CmaEsOptimizer:
    """Generate bounded candidate populations and track the best candidate."""

    def __init__(
        self,
        parameter_space: ParameterSpace,
        config: CmaesOptimizerConfig,
        *,
        population_size: int,
        seed: int,
    ) -> None:
        self.parameter_space = parameter_space
        self.population_size = population_size
        self.epsilon = float(config.epsilon)
        self.max_iterations = int(config.max_iterations)
        self._optimizer = cmaes.CMA(
            mean=parameter_space.initial_normalized,
            sigma=float(config.sigma),
            bounds=np.asarray(
                [[-1.0, 1.0]] * parameter_space.dimension,
                dtype=np.float64,
            ),
            population_size=population_size,
            seed=seed,
        )
        self._completed_iterations = 0
        self._converged = False
        self._best_score = float("inf")
        self._best_candidate: Optional[FloatArray] = None
        self._best_score_history: list[float] = []

    @property
    def converged(self) -> bool:
        return self._converged

    @property
    def best_candidate(self) -> FloatArray:
        if self._best_candidate is None:
            raise RuntimeError("optimizer has not evaluated a candidate")
        return np.array(self._best_candidate, copy=True)

    @property
    def best_score(self) -> float:
        return self._best_score

    def step(self, objective: Objective) -> OptimizationStep:
        """Evaluate one population and update the CMA-ES search state."""

        if self.converged:
            raise RuntimeError("optimizer has already converged")

        iteration = self._completed_iterations
        population = self._ask_population()
        scores = objective.evaluate(np.array(population, copy=True))
        values = self._tell_population(population, scores)
        self._converged = self._should_stop()

        return OptimizationStep(
            iteration=iteration,
            scores=np.array(values, copy=True),
            best_score=self.best_score,
            converged=self.converged,
        )

    def _ask_population(self) -> FloatArray:
        population = np.asarray(
            [self._optimizer.ask() for _ in range(self.population_size)],
            dtype=np.float64,
        )
        if population.shape != (
            self.population_size,
            self.parameter_space.dimension,
        ):
            raise RuntimeError("CMA-ES returned an invalid population shape")
        if not np.isfinite(population).all():
            raise RuntimeError("CMA-ES returned a non-finite candidate")
        if np.any(population < -1.0) or np.any(population > 1.0):
            raise RuntimeError("CMA-ES returned a candidate outside its bounds")
        return population

    def _tell_population(
        self,
        population: FloatArray,
        scores: FloatArray,
    ) -> FloatArray:
        values = np.asarray(scores, dtype=np.float64)
        if values.shape != (self.population_size,):
            raise ValueError(
                f"expected {self.population_size} scores, got {values.shape}"
            )
        if not np.isfinite(values).all():
            raise RuntimeError("objective returned a non-finite candidate score")

        self._optimizer.tell(
            [
                (population[index], float(values[index]))
                for index in range(self.population_size)
            ]
        )
        best_index = int(values.argmin())
        best_score = float(values[best_index])
        if best_score < self.best_score:
            self._best_score = best_score
            self._best_candidate = np.array(population[best_index], copy=True)
        self._best_score_history.append(self.best_score)
        self._completed_iterations += 1
        return values

    def _should_stop(self) -> bool:
        should_stop = getattr(self._optimizer, "should_stop", None)
        if callable(should_stop) and should_stop():
            return True
        if self._completed_iterations >= self.max_iterations:
            return True
        if self._completed_iterations <= MINIMUM_CONVERGENCE_GENERATIONS:
            return False
        previous = self._best_score_history[-IMPROVEMENT_WINDOW - 1]
        current = self._best_score_history[-1]
        denominator = max(abs(previous), np.finfo(np.float64).eps)
        relative_improvement = (previous - current) / denominator
        return relative_improvement < self.epsilon


def create_optimizer(
    config: ActuatorSysIdConfig,
    parameter_space: ParameterSpace,
) -> Optimizer:
    """Create the optimizer selected by the config's string discriminator."""

    optimizer_config = config.optimizer
    if optimizer_config.type == _config_type(CmaesOptimizerConfig):
        return CmaEsOptimizer(
            parameter_space,
            optimizer_config,
            population_size=config.num_envs,
            seed=config.seed,
        )
    raise ValueError(f"Unsupported optimizer type: {optimizer_config.type!r}")
