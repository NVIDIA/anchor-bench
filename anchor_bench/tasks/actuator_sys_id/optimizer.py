"""CMA-ES adapter for actuator system identification."""

from __future__ import annotations

from typing import Optional, Protocol

import cmaes
import numpy as np

from anchor_bench.schema.actuator_sys_id import (
    ActuatorSysIdConfig,
    CmaesOptimizerConfig,
)
from anchor_bench.schema.helpers import _config_type
from anchor_bench.tasks.actuator_sys_id.parameter_space import ParameterSpace

MINIMUM_CONVERGENCE_GENERATIONS = 5
IMPROVEMENT_WINDOW = 5


class Optimizer(Protocol):
    generation: int
    best_score: float
    best_normalized: Optional[np.ndarray]

    def ask(self) -> np.ndarray: ...

    def tell(self, scores: np.ndarray) -> bool: ...


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
        self._last_population: Optional[np.ndarray] = None
        self.generation = 0
        self.best_score = float("inf")
        self.best_normalized: Optional[np.ndarray] = None
        self._best_score_history: list[float] = []

    def ask(self) -> np.ndarray:
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
        self._last_population = population
        return np.array(population, copy=True)

    def tell(self, scores: np.ndarray) -> bool:
        if self._last_population is None:
            raise RuntimeError("ask must be called before tell")
        values = np.asarray(scores, dtype=np.float64)
        if values.shape != (self.population_size,):
            raise ValueError(
                f"expected {self.population_size} scores, got {values.shape}"
            )
        if not np.isfinite(values).all():
            raise RuntimeError("simulation returned a non-finite candidate score")

        self._optimizer.tell(
            [
                (self._last_population[index], float(values[index]))
                for index in range(self.population_size)
            ]
        )
        best_index = int(values.argmin())
        best_score = float(values[best_index])
        if best_score < self.best_score:
            self.best_score = best_score
            self.best_normalized = np.array(
                self._last_population[best_index], copy=True
            )
        self._best_score_history.append(self.best_score)
        self.generation += 1

        should_stop = getattr(self._optimizer, "should_stop", None)
        if callable(should_stop) and should_stop():
            return True
        if self.generation <= MINIMUM_CONVERGENCE_GENERATIONS:
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
