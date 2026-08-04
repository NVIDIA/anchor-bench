"""Extension interfaces for training-dataset preprocessing."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from anchor_bench.core.simulation import SimulationContext


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

    def __init__(self, dataset: pd.DataFrame) -> None:
        super().__init__(dataset)
        simulation_context = SimulationContext.current()
        self.model = simulation_context.model
        self.state = simulation_context.state
