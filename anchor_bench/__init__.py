# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Anchor Bench task runner."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from anchor_bench.core.task_runner import TaskRunner

__all__ = ["TaskRunner"]


def __getattr__(name: str) -> Any:
    """Load the runner only when explicitly requested."""

    if name != "TaskRunner":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from anchor_bench.core.task_runner import TaskRunner

    globals()[name] = TaskRunner
    return TaskRunner
