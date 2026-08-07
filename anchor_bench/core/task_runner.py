"""Parse task configuration and dispatch it to its implementation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Optional, Type, Union

from anchor_bench.core.artifacts import (
    COMPLETION_MARKER,
    get_or_create_output_folder,
    write_yaml,
)
from anchor_bench.errors import TaskNotFoundError
from anchor_bench.schema.parse import parse
from anchor_bench.tasks.actuator_sys_id import ActuatorSysIdTask
from anchor_bench.tasks.base_task import BaseTask
from anchor_bench.tasks.train.task import TrainingTask

TASKS: Dict[str, Type[BaseTask[Any]]] = {
    ActuatorSysIdTask.name(): ActuatorSysIdTask,
    TrainingTask.name(): TrainingTask,
}


class TaskRunner:
    def __init__(
        self,
        task_registry: Optional[Mapping[str, Type[BaseTask[Any]]]] = None,
    ) -> None:
        self._tasks = dict(TASKS if task_registry is None else task_registry)

    def run(
        self,
        config_path: str,
        *,
        force: bool = False,
        output: Optional[Union[str, Path]] = None,
    ) -> Path:
        config = parse(config_path)
        task_type = config.type
        task_class = self._tasks.get(task_type)

        if task_class is None:
            known_types = ", ".join(sorted(self._tasks)) or "none"
            raise TaskNotFoundError(
                f"No task is registered for config type {task_type!r}. "
                f"Registered task types: {known_types}."
            )

        output_directory, already_completed = get_or_create_output_folder(
            config,
            output_root=getattr(config, "output_dir", None),
            output_folder=output,
        )
        if already_completed and not force:
            return output_directory

        write_yaml(
            output_directory / "config.yaml",
            config.model_dump(mode="json"),
        )
        completion_marker = output_directory / COMPLETION_MARKER
        completion_marker.unlink(missing_ok=True)
        task_output = task_class(config).run(output_directory)
        if task_output != output_directory:
            raise RuntimeError(
                f"{task_class.__name__}.run() returned {task_output}, expected "
                f"its assigned output directory {output_directory}."
            )
        completion_marker.touch()
        return task_output
