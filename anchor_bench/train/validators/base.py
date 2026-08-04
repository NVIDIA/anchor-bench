"""Extension interfaces for training validators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal, Mapping

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

    @abstractmethod
    def validate(self) -> object:
        """Validate a trained model using the configured strategy."""
        raise NotImplementedError
