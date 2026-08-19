# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extension interfaces for training validators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal, Mapping

from anchor_bench.core.simulation import SimulationContext

ValidatorType = Literal["offline", "simulation"]


@dataclass(frozen=True)
class ValidationResult:
    """Structured metrics produced by one configured validator."""

    name: str
    type: ValidatorType
    metrics: Mapping[str, float]


class BaseValidator(ABC):
    """Base class for configured training validators."""

    def __init__(
        self,
        *,
        name: str,
        type: ValidatorType,
        config: Mapping[str, Any],
    ) -> None:
        self.name: str = name
        self.type: ValidatorType = type
        self.config: dict[str, Any] = dict(config)
        self.simulation_context = SimulationContext.current()

    @abstractmethod
    def validate(self, model: Any, validation_dataset: Any) -> ValidationResult:
        """Validate a trained model against its validation dataset."""
        raise NotImplementedError
