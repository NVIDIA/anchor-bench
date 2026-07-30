"""Extension interfaces for training-dataset preprocessing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol, Tuple

import pandas as pd


class SimulationContext(Protocol):
    """Minimum simulation interface required by a simulation preprocessor."""

    def load(self, usd_path: str) -> Tuple[Any, Any]:
        """Load a USD asset and return its model and initial state."""
        ...


class BasePreprocessor(ABC):
    """Base class for generating features from a resampled dataset."""

    def __init__(self, dataset: pd.DataFrame) -> None:
        self.dataset = dataset

    @abstractmethod
    def process(self) -> pd.DataFrame:
        """Return generated feature columns aligned to the dataset index."""
        raise NotImplementedError


class SimPreprocessor(BasePreprocessor):
    """Base class for preprocessors that derive features through simulation."""

    def __init__(
        self,
        dataset: pd.DataFrame,
        usd_path: str,
        simulation_context: SimulationContext,
    ) -> None:
        super().__init__(dataset)
        self.model, self.state = simulation_context.load(usd_path)
