# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Actuator system-identification task and result."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from anchor_bench.tasks.actuator_sys_id.model import SysIdResult
    from anchor_bench.tasks.actuator_sys_id.task import ActuatorSysIdTask

_EXPORTS = {
    "ActuatorSysIdTask": (
        "anchor_bench.tasks.actuator_sys_id.task",
        "ActuatorSysIdTask",
    ),
    "SysIdResult": ("anchor_bench.tasks.actuator_sys_id.model", "SysIdResult"),
}

__all__ = ["ActuatorSysIdTask", "SysIdResult"]


def __getattr__(name: str) -> Any:
    """Resolve public exports without importing task or simulator runtimes eagerly."""

    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
