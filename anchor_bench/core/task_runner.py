"""Parse task configuration and dispatch it to its implementation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Optional, Type

from anchor_bench.errors import TaskNotFoundError
from anchor_bench.schema.parse import parse
from anchor_bench.tasks.actuator_sys_id import ActuatorSysIdTask
from anchor_bench.tasks.base_task import BaseTask

TASKS: Dict[str, Type[BaseTask[Any]]] = {
    ActuatorSysIdTask.name(): ActuatorSysIdTask,
}


class TaskRunner:
    def __init__(
        self,
        task_registry: Optional[Mapping[str, Type[BaseTask[Any]]]] = None,
    ) -> None:
        self._tasks = dict(TASKS if task_registry is None else task_registry)

    def run(self, config_path: str) -> Path:
        config = parse(config_path)
        task_type = config.type
        task_class = self._tasks.get(task_type)

        if task_class is None:
            known_types = ", ".join(sorted(self._tasks)) or "none"
            raise TaskNotFoundError(
                f"No task is registered for config type {task_type!r}. "
                f"Registered task types: {known_types}."
            )

        return task_class(config).run()
