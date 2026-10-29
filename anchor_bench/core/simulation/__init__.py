# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared simulation context."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable
from typing import Any, ClassVar, Generic, Optional, Tuple, TypeVar, cast

from typing_extensions import Self

ModelT = TypeVar("ModelT")
StateT = TypeVar("StateT")


class SimulationContext(ABC, Generic[ModelT, StateT]):
    """Own the single lazily loaded simulation model and state."""

    _instance: ClassVar[Optional[SimulationContext[Any, Any]]] = None
    _resource_key: Hashable
    _model: Optional[ModelT]
    _state: Optional[StateT]

    def __new__(cls, *, resource_key: Hashable) -> Self:
        """Create or return the process-wide context for a hashable resource."""

        try:
            hash(resource_key)
        except TypeError as exc:
            raise TypeError("resource_key must be hashable.") from exc

        current = SimulationContext._instance
        if current is not None:
            if type(current) is not cls:
                raise RuntimeError(
                    "Only one simulation backend can be used per process."
                )
            if current.resource_key != resource_key:
                raise RuntimeError(
                    "Simulation context already uses resource key "
                    f"{current.resource_key!r}."
                )
            return cast(Self, current)

        instance = super().__new__(cls)
        instance._resource_key = resource_key
        instance._model = None
        instance._state = None
        SimulationContext._instance = instance
        return instance

    def __init__(self, *, resource_key: Hashable) -> None:
        """Accept the mandatory resource identity handled during construction."""

    @classmethod
    def current(cls) -> Self:
        """Return the process-wide simulation context."""

        instance = SimulationContext._instance
        if instance is None:
            raise RuntimeError("Simulation context has not been created.")
        return cast(Self, instance)

    @property
    def resource_key(self) -> Hashable:
        """Return the backend-defined resource identity assigned to this process."""

        return self._resource_key

    def load(self) -> Tuple[ModelT, StateT]:
        """Load and cache the model and state on first use."""

        if self._model is None or self._state is None:
            model, state = self._load()
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

    @abstractmethod
    def _load(self) -> Tuple[ModelT, StateT]:
        """Load a backend model and initial state for the assigned resource."""
        raise NotImplementedError
