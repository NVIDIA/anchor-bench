"""Extension interfaces for training validators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal, Mapping

from anchor_bench.core.simulation import SimulationContext

ValidatorType = Literal["offline", "simulation"]


class BaseValidator(ABC):
    """Base class for configured training validators."""

    def __init__(
        self,
        *,
        name: str,
        type: ValidatorType,
        config: Mapping[str, Any],
    ) -> None:
        self.name = name
        self.type = type
        self.config = dict(config)
        self.simulation_context = SimulationContext.current()

    @abstractmethod
    def validate(self, model: Any, validation_dataset: Any) -> object:
        """Validate a trained model against its validation dataset."""
        raise NotImplementedError
