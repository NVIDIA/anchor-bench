"""Base interface implemented by all Anchor Bench tasks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from anchor_bench.schema.base import BaseConfig

ConfigT = TypeVar("ConfigT", bound=BaseConfig)


class BaseTask(ABC, Generic[ConfigT]):
    """A task constructed from a validated configuration model."""

    @abstractmethod
    def __init__(self, config: ConfigT) -> None:
        """Initialize the task with a validated configuration."""

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        """Return the config discriminator handled by this task."""

    @abstractmethod
    def run(self) -> Any:
        """Execute the task."""
