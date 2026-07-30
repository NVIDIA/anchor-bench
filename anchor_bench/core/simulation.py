"""Shared simulation context."""

from __future__ import annotations

from typing import Any, Tuple


class SimulationContext:
    """Load a simulation model and its initial state."""

    def load(self, usd_path: str) -> Tuple[Any, Any]:
        raise NotImplementedError
