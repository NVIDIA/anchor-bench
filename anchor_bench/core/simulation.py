"""Shared simulation context."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Tuple


class SimulationContext(ABC):
    """Load a simulation model and its initial state."""

    @abstractmethod
    def load(self, usd_path: str) -> Tuple[Any, Any]:
        raise NotImplementedError
