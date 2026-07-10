"""CMA-ES population adapter for explicit SysID parameter layouts."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from anchor_bench.sysid.schema import (
    ParameterLayoutEntry,
    SysIdOptimizerConfig,
    SysIdProfile,
    parameter_layout,
)


_MIN_CONVERGENCE_GENERATIONS = 5
_IMPROVEMENT_WINDOW = 5


@dataclass
class CMAESOptimizer:
    """Adapt CMA-ES samples to canonical per-joint physical parameters.

    Callers pass one resolved SysID profile and its exact parent parameter
    state. Actual runs require cmaes; host tests may explicitly use
    deterministic sampling.
    """

    profile: SysIdProfile
    num_envs: int
    cmaes: SysIdOptimizerConfig
    seed: int = 0
    allow_random_fallback: bool = False
    parent_parameters: Mapping[str, Mapping[str, float]] | None = None

    def __post_init__(self) -> None:
        """Build the explicit variable layout and initialize the optimizer."""

        if self.num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        self._layout: tuple[ParameterLayoutEntry, ...] = parameter_layout(
            self.profile
        )
        self.param_names = [entry.name for entry in self._layout]
        self._lower = np.asarray(
            [entry.bounds.lower for entry in self._layout], dtype=np.float64
        )
        self._upper = np.asarray(
            [entry.bounds.upper for entry in self._layout], dtype=np.float64
        )
        self._normalized_mean = self._parent_normalized_mean()

        self._rng = np.random.default_rng(self.seed)
        self._cma = None
        self.uses_random_fallback = False
        self._last_norm: np.ndarray | None = None
        self._last_matrix: np.ndarray | None = None
        self.generation = 0
        self.best_score = float("inf")
        self.best_params: dict[str, dict[str, float]] | None = None
        self._best_score_history: list[float] = []

        if self.allow_random_fallback:
            self.uses_random_fallback = True
            return
        try:
            import cmaes
        except ImportError as error:
            raise RuntimeError(
                "cmaes package is required for actual SysID optimization; "
                "install the full runtime or pass allow_random_fallback=True "
                "for host-only tests."
            ) from error
        try:
            self._cma = cmaes.CMA(
                mean=self._normalized_mean.copy(),
                sigma=self.cmaes.sigma,
                bounds=np.asarray([[-1.0, 1.0]] * len(self.param_names)),
                population_size=self.num_envs,
                seed=self.seed,
            )
        except Exception as error:
            raise RuntimeError(f"failed to initialize cmaes optimizer: {error}") from error

    @property
    def normalized_mean(self) -> list[float]:
        """Return the bounded starting mean in optimizer coordinates."""

        return self._normalized_mean.tolist()

    def _parent_normalized_mean(self) -> np.ndarray:
        """Map selected parent cells into the tied-variable layout."""

        if self.parent_parameters is None:
            raise ValueError("parent_parameters is required with a SysIdProfile")
        physical: list[float] = []
        for entry in self._layout:
            parameter_map = self.parent_parameters.get(entry.parameter)
            if parameter_map is None:
                raise ValueError(
                    f"parent state is missing parameter {entry.parameter!r}"
                )
            values: list[float] = []
            for joint in entry.joints:
                if joint not in parameter_map:
                    raise ValueError(
                        f"parent state is missing cell {entry.parameter}/{joint}"
                    )
                value = float(parameter_map[joint])
                if not math.isfinite(value):
                    raise ValueError(
                        f"parent state cell {entry.parameter}/{joint} must be finite"
                    )
                values.append(value)
            # A shared variable can start from independently accepted cells.
            # Their average is the symmetric initialization for the new tie.
            physical.append(sum(values) / len(values))

        values = np.asarray(physical, dtype=np.float64)
        spans = self._upper - self._lower
        normalized = np.zeros_like(values)
        nonzero = spans > 0.0
        normalized[nonzero] = (
            2.0 * (values[nonzero] - self._lower[nonzero]) / spans[nonzero] - 1.0
        )
        return np.clip(normalized, -1.0, 1.0)

    def _denormalize(self, values: np.ndarray) -> np.ndarray:
        """Convert bounded optimizer coordinates to physical values."""

        unit = (values + 1.0) / 2.0
        return self._lower + unit * (self._upper - self._lower)

    def sample_matrix(self) -> list[list[float]]:
        """Sample one physical row for every parallel environment."""

        if self._cma is not None:
            raw = np.asarray(
                [self._cma.ask() for _ in range(self.num_envs)],
                dtype=np.float64,
            )
        else:
            raw = self._rng.uniform(
                -1.0,
                1.0,
                size=(self.num_envs, len(self.param_names)),
            )
        raw = np.clip(raw, -1.0, 1.0)
        self._last_norm = raw
        self._last_matrix = self._denormalize(raw)
        return [
            [round(float(value), 9) for value in row]
            for row in self._last_matrix
        ]

    def sample_population(self) -> list[dict[str, dict[str, float]]]:
        """Expand sampled rows into canonical per-joint maps."""

        return [self._row_to_candidate(row) for row in self.sample_matrix()]

    def tell(self, scores: list[float] | np.ndarray) -> bool:
        """Report scores, track the best patch, and return convergence."""

        score_array = np.asarray(scores, dtype=np.float64)
        if score_array.shape != (self.num_envs,):
            raise ValueError(f"Expected {self.num_envs} scores, got {score_array.shape[0]}")
        if not np.all(np.isfinite(score_array)):
            raise ValueError("scores must be finite")
        if self._last_norm is None or self._last_matrix is None:
            raise RuntimeError("sample_population must be called before tell")

        if self._cma is not None:
            self._cma.tell(
                [
                    (self._last_norm[index].tolist(), float(score_array[index]))
                    for index in range(self.num_envs)
                ]
            )

        best_index = int(score_array.argmin())
        best_score = float(score_array[best_index])
        if best_score < self.best_score:
            self.best_score = best_score
            self.best_params = self._row_to_candidate(self._last_matrix[best_index])
        self._best_score_history.append(self.best_score)
        self.generation += 1

        if self._cma_should_stop():
            return True
        epsilon = self.cmaes.epsilon
        if epsilon is None or self.generation <= _MIN_CONVERGENCE_GENERATIONS:
            return False
        if len(self._best_score_history) <= _IMPROVEMENT_WINDOW:
            return False
        previous = float(self._best_score_history[-_IMPROVEMENT_WINDOW - 1])
        current = float(self._best_score_history[-1])
        denominator = previous if previous > 0.0 else 1.0
        return ((previous - current) / denominator) < float(epsilon)

    def _cma_should_stop(self) -> bool:
        """Read the optional cmaes-native termination signal."""

        if self._cma is None:
            return False
        should_stop = getattr(self._cma, "should_stop", None)
        return bool(should_stop()) if callable(should_stop) else False

    def _row_to_candidate(
        self, row: np.ndarray | list[float]
    ) -> dict[str, dict[str, float]]:
        """Expand one physical vector according to the active layout."""

        candidate: dict[str, dict[str, float]] = {}
        for entry, raw_value in zip(self._layout, row, strict=True):
            joint_map = candidate.setdefault(entry.parameter, {})
            value = round(float(raw_value), 9)
            for joint in entry.joints:
                joint_map[joint] = value
        return candidate
