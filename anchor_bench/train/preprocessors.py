"""Extension interfaces for training-dataset preprocessing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from anchor_bench.core.simulation import SimulationContext


class BasePreprocessor(ABC):
    """Base class for generating features from a resampled dataset."""

    def __init__(self, dataset: pd.DataFrame) -> None:
        self.dataset = dataset
        self.simulation_context = SimulationContext.current()

    @property
    def model(self) -> Any:
        """Return the shared lazily loaded simulation model."""

        return self.simulation_context.model

    @property
    def state(self) -> Any:
        """Return the shared lazily created simulation state."""

        return self.simulation_context.state

    @abstractmethod
    def process(self) -> pd.DataFrame:
        """Return generated feature columns aligned to the dataset index."""
        raise NotImplementedError
