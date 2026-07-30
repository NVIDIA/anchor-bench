"""Actuator system-identification task."""

from __future__ import annotations

from typing import Any

from anchor_bench.schema.actuator_sys_id import ActuatorSysIdConfig
from anchor_bench.tasks.base_task import BaseTask


class ActuatorSysIdTask(BaseTask[ActuatorSysIdConfig]):
    """Task boundary for actuator system identification.

    The task runner and validated configuration are implemented here. The
    optimizer/simulator execution pipeline will be added behind this boundary.
    """

    def __init__(self, config: ActuatorSysIdConfig) -> None:
        self.config = config

    @classmethod
    def name(cls) -> str:
        return "actuator_sys_id"

    def run(self) -> Any:
        # Execution is intentionally a no-op until the sys-id engine is wired in.
        return None
