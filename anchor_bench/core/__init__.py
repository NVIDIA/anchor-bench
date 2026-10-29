# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Core task-runner and simulation functionality."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from anchor_bench.core.artifacts import (
        get_or_create_output_folder,
        resolve_repo_root,
        write_yaml,
    )
    from anchor_bench.core.simulation import SimulationContext
    from anchor_bench.core.simulation.newton import NewtonSimulationContext
    from anchor_bench.core.task_runner import TaskRunner

_EXPORTS = {
    "NewtonSimulationContext": (
        "anchor_bench.core.simulation.newton",
        "NewtonSimulationContext",
    ),
    "SimulationContext": ("anchor_bench.core.simulation", "SimulationContext"),
    "TaskRunner": ("anchor_bench.core.task_runner", "TaskRunner"),
    "get_or_create_output_folder": (
        "anchor_bench.core.artifacts",
        "get_or_create_output_folder",
    ),
    "resolve_repo_root": ("anchor_bench.core.artifacts", "resolve_repo_root"),
    "write_yaml": ("anchor_bench.core.artifacts", "write_yaml"),
}

__all__ = [
    "NewtonSimulationContext",
    "SimulationContext",
    "TaskRunner",
    "get_or_create_output_folder",
    "resolve_repo_root",
    "write_yaml",
]


def __getattr__(name: str) -> Any:
    """Resolve public exports without importing task or simulator runtimes eagerly."""

    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
