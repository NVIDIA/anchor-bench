"""Base interface implemented by all Anchor Bench tasks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, Optional, Type, TypeVar, get_args, get_origin

from anchor_bench.core.artifacts import get_or_create_output_folder
from anchor_bench.schema.base import BaseConfig
from anchor_bench.schema.helpers import _config_type

ConfigT = TypeVar("ConfigT", bound=BaseConfig)


class BaseTask(ABC, Generic[ConfigT]):
    """A task constructed from a validated configuration model."""

    def __init__(
        self,
        config: ConfigT,
        output_directory: Optional[Path] = None,
    ) -> None:
        """Initialize the task with its configuration and artifact directory."""

        self.config = config
        if output_directory is None:
            output_directory, _ = get_or_create_output_folder(
                config,
                output_root=getattr(config, "output_dir", None),
            )
        self.output_directory = output_directory

    def __init_subclass__(cls) -> None:
        """Require every concrete task to inherit directly and only from BaseTask."""

        super().__init_subclass__()
        if cls.__bases__ != (BaseTask,):
            raise TypeError(f"{cls.__name__} must directly inherit only from BaseTask")

    @classmethod
    def name(cls) -> str:
        """Return the string Literal declared by the task's config type."""

        return _config_type(cls._config_class())

    @classmethod
    def _config_class(cls) -> Type[BaseConfig]:
        """Return the concrete config bound through ``BaseTask[ConfigT]``."""

        for base in getattr(cls, "__orig_bases__", ()):
            if get_origin(base) is not BaseTask:
                continue
            arguments = get_args(base)
            if len(arguments) != 1:
                continue
            config_class = arguments[0]
            if isinstance(config_class, type) and issubclass(config_class, BaseConfig):
                return config_class
        raise TypeError(
            f"{cls.__name__} must bind a concrete config as BaseTask[ConfigType]"
        )

    @abstractmethod
    def run(self) -> Path:
        """Execute the task and return its output directory."""
