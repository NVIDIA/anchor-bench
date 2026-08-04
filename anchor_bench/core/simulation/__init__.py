"""Shared simulation context."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Optional, Tuple

from typing_extensions import Self


class SimulationContext(ABC):
    """Own the single lazily loaded simulation model and state."""

    _instance: ClassVar[Optional[SimulationContext]] = None
    _usd_path: Path
    _model: Any
    _state: Any

    def __new__(cls, usd_path: str) -> Self:
        """Create or return the process-wide simulation context."""

        source = Path(usd_path).expanduser().resolve()
        current = SimulationContext._instance
        if current is not None:
            if type(current) is not cls:
                raise RuntimeError(
                    "Only one simulation backend can be used per process."
                )
            if current.usd_path != source:
                raise RuntimeError(
                    f"Simulation context already uses {current.usd_path}."
                )
            return current

        if not source.is_file():
            raise FileNotFoundError(f"USD file does not exist: {source}.")

        instance = super().__new__(cls)
        instance._usd_path = source
        instance._model = None
        instance._state = None
        SimulationContext._instance = instance
        return instance

    def __init__(self, usd_path: str) -> None:
        """Accept the mandatory USD path handled during construction."""

    @classmethod
    def current(cls) -> SimulationContext:
        """Return the process-wide simulation context."""

        instance = SimulationContext._instance
        if instance is None:
            raise RuntimeError("Simulation context has not been created.")
        return instance

    @property
    def usd_path(self) -> Path:
        """Return the USD file assigned to this process."""

        return self._usd_path

    def load(self) -> Tuple[Any, Any]:
        """Load and cache the model and state on first use."""

        if self._model is None:
            model, state = self._load(self._usd_path)
            self._model = model
            self._state = state
        return self._model, self._state

    @property
    def model(self) -> Any:
        """Return the lazily loaded simulation model."""

        model, _ = self.load()
        return model

    @property
    def state(self) -> Any:
        """Return the lazily created initial simulation state."""

        _, state = self.load()
        return state

    @abstractmethod
    def _load(self, usd_path: Path) -> Tuple[Any, Any]:
        """Load a backend model and initial state from the assigned USD."""
        raise NotImplementedError
