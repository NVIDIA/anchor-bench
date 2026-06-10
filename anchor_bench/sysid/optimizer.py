from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from anchor_bench.core.schemas import CMAESConfig, SysIdBoundsFile


_MIN_CONVERGENCE_GENERATIONS = 5
_IMPROVEMENT_WINDOW = 5


@dataclass
class CMAESOptimizer:
    """CMA-ES population adapter for actuator parameter search.

    Actual SysID runs require the `cmaes` package. Host-only tests may opt into
    deterministic bounded sampling by setting `allow_random_fallback=True`.
    """

    bounds: SysIdBoundsFile
    num_envs: int
    cmaes: CMAESConfig
    seed: int = 0
    allow_random_fallback: bool = False

    def __post_init__(self) -> None:
        self.joint_types = list(self.bounds.joint_types)
        self.parameter_names = list(self.bounds.parameters)
        self.param_names = [
            f"{param}/{joint}"
            for param in self.parameter_names
            for joint in self.joint_types
        ]
        self._lower = np.array(
            [
                self.bounds.parameters[param].lower
                for param in self.parameter_names
                for _ in self.joint_types
            ],
            dtype=np.float64,
        )
        self._upper = np.array(
            [
                self.bounds.parameters[param].upper
                for param in self.parameter_names
                for _ in self.joint_types
            ],
            dtype=np.float64,
        )
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
        except ImportError as exc:
            raise RuntimeError(
                "cmaes package is required for actual SysID optimization; "
                "install the full runtime or pass allow_random_fallback=True for host-only tests."
            ) from exc

        try:
            self._cma = cmaes.CMA(
                mean=np.zeros(len(self.param_names)),
                sigma=self.cmaes.sigma,
                bounds=np.array([[-1.0, 1.0]] * len(self.param_names)),
                population_size=self.num_envs,
                seed=self.seed,
            )
        except Exception as exc:
            raise RuntimeError(f"failed to initialize cmaes optimizer: {exc}") from exc

    def _denormalize(self, values: np.ndarray) -> np.ndarray:
        unit = (values + 1.0) / 2.0
        return self._lower + unit * (self._upper - self._lower)

    def sample_matrix(self) -> list[list[float]]:
        if self._cma is not None:
            raw = np.array([self._cma.ask() for _ in range(self.num_envs)], dtype=np.float64)
        else:
            raw = self._rng.uniform(-1.0, 1.0, size=(self.num_envs, len(self.param_names)))
        raw = np.clip(raw, -1.0, 1.0)
        self._last_norm = raw
        self._last_matrix = self._denormalize(raw)
        return [[round(float(value), 9) for value in row] for row in self._last_matrix]

    def sample_population(self) -> list[dict[str, dict[str, float]]]:
        population: list[dict[str, dict[str, float]]] = []
        for row in self.sample_matrix():
            candidate: dict[str, dict[str, float]] = {}
            index = 0
            for param in self.parameter_names:
                candidate[param] = {}
                for joint in self.joint_types:
                    candidate[param][joint] = row[index]
                    index += 1
            population.append(candidate)
        return population

    def tell(self, scores: list[float] | np.ndarray) -> bool:
        score_array = np.asarray(scores, dtype=np.float64)
        if score_array.shape[0] != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} scores, got {score_array.shape[0]}")
        if self._last_norm is None or self._last_matrix is None:
            raise RuntimeError("sample_population must be called before tell")

        if self._cma is not None:
            solutions = [
                (self._last_norm[index].tolist(), float(score_array[index]))
                for index in range(self.num_envs)
            ]
            self._cma.tell(solutions)

        best_idx = int(score_array.argmin())
        best_score = float(score_array[best_idx])
        if best_score < self.best_score:
            self.best_score = best_score
            self.best_params = self._row_to_candidate(self._last_matrix[best_idx])
        self._best_score_history.append(self.best_score)

        self.generation += 1
        if self._cma_should_stop():
            return True
        epsilon = self.cmaes.epsilon
        if epsilon is None:
            return False
        if self.generation <= _MIN_CONVERGENCE_GENERATIONS:
            return False
        if len(self._best_score_history) <= _IMPROVEMENT_WINDOW:
            return False
        previous = float(self._best_score_history[-_IMPROVEMENT_WINDOW - 1])
        current = float(self._best_score_history[-1])
        improvement = previous - current
        denominator = previous if previous > 0.0 else 1.0
        return (improvement / denominator) < float(epsilon)

    def _cma_should_stop(self) -> bool:
        if self._cma is None:
            return False
        should_stop = getattr(self._cma, "should_stop", None)
        return bool(should_stop()) if callable(should_stop) else False

    def _row_to_candidate(self, row: np.ndarray | list[float]) -> dict[str, dict[str, float]]:
        candidate: dict[str, dict[str, float]] = {}
        index = 0
        for param in self.parameter_names:
            candidate[param] = {}
            for joint in self.joint_types:
                candidate[param][joint] = round(float(row[index]), 9)
                index += 1
        return candidate
