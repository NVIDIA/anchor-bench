"""Shared simulation interfaces."""

from __future__ import annotations

from typing import Any, Protocol, Tuple


class SimulationContext(Protocol):
    """Minimum interface for loading a simulation model."""

    def load(self, usd_path: str) -> Tuple[Any, Any]:
        """Load a USD asset and return its model and initial state."""
        ...
