# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared simulation context."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    Optional,
    Tuple,
    TypeVar,
    cast,
)

from typing_extensions import Self

if TYPE_CHECKING:
    from anchor_bench.schema.actuator_sys_id import ActuatorSysIdConfig
    from anchor_bench.tasks.actuator_sys_id.objective import PopulationEvaluator
    from anchor_bench.tasks.actuator_sys_id.parameter_space import ParameterSpace

ModelT = TypeVar("ModelT")
StateT = TypeVar("StateT")


class SimulationContext(ABC, Generic[ModelT, StateT]):
    """Own the single lazily loaded simulation model and state."""

    _instance: ClassVar[Optional[SimulationContext[Any, Any]]] = None
    _usd_path: Path
    _fixed_base: bool
    _model: Optional[ModelT]
    _state: Optional[StateT]

    def __new__(
        cls,
        usd_path: str,
        *,
        fixed_base: bool = False,
    ) -> Self:
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
            if current.fixed_base != fixed_base:
                raise RuntimeError(
                    "Simulation context already uses a different base mode."
                )
            return cast(Self, current)

        if not source.is_file():
            raise FileNotFoundError(f"USD file does not exist: {source}.")

        instance = super().__new__(cls)
        instance._usd_path = source
        instance._fixed_base = fixed_base
        instance._model = None
        instance._state = None
        SimulationContext._instance = instance
        return instance

    def __init__(
        self,
        usd_path: str,
        *,
        fixed_base: bool = False,
    ) -> None:
        """Accept the mandatory USD path handled during construction."""

    @classmethod
    def current(cls) -> Self:
        """Return the process-wide simulation context."""

        instance = SimulationContext._instance
        if instance is None:
            raise RuntimeError("Simulation context has not been created.")
        return cast(Self, instance)

    @property
    def usd_path(self) -> Path:
        """Return the USD file assigned to this process."""

        return self._usd_path

    @property
    def fixed_base(self) -> bool:
        """Return whether the imported robot root is fixed."""

        return self._fixed_base

    def load(self) -> Tuple[ModelT, StateT]:
        """Load and cache the model and state on first use."""

        if self._model is None or self._state is None:
            model, state = self._load(self._usd_path)
            self._model = model
            self._state = state
        return self._model, self._state

    @property
    def model(self) -> ModelT:
        """Return the lazily loaded simulation model."""

        model, _ = self.load()
        return model

    @property
    def state(self) -> StateT:
        """Return the lazily created initial simulation state."""

        _, state = self.load()
        return state

    def create_sysid_evaluator(
        self, config: ActuatorSysIdConfig, parameter_space: ParameterSpace
    ) -> PopulationEvaluator:
        """Create the population evaluator used by actuator system identification."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support system identification."
        )

    @abstractmethod
    def _load(self, usd_path: Path) -> Tuple[ModelT, StateT]:
        """Load a backend model and initial state from the assigned USD."""
        raise NotImplementedError
