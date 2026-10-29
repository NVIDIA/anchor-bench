# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base interface implemented by all Anchor Bench tasks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Generic, Type, TypeVar, get_args, get_origin

from anchor_bench.errors import AnchorBenchError
from anchor_bench.schema.base import BaseConfig
from anchor_bench.schema.helpers import _config_type

ConfigT = TypeVar("ConfigT", bound=BaseConfig)


class BaseTask(ABC, Generic[ConfigT]):
    """A task constructed from a validated configuration model."""

    def __init__(self, config: ConfigT) -> None:
        """Initialize the task with its configuration."""

        self.config = config

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

    @contextmanager
    def output_context(
        self, output: Path, *, force: bool = False, resume: bool = False
    ):
        """Protect and validate artifacts before the runner reads completion state."""
        if resume:
            raise AnchorBenchError(f"{self.name()} does not support --resume")
        yield

    @abstractmethod
    def run(self, output_directory: Path) -> Path:
        """Execute the task in a runner-provided output directory."""
