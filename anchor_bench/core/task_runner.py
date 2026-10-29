# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parse task configuration and dispatch it to its implementation."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Dict, Optional, Type, Union

from anchor_bench.core.artifacts import (
    COMPLETION_MARKER,
    get_or_create_output_folder,
    write_yaml,
)
from anchor_bench.errors import (
    AnchorBenchError,
    MLOpsBackendNotFoundError,
    TaskNotFoundError,
)
from anchor_bench.mlops import (
    MLflowBackend,
    MLOpsBackend,
    bind_active_run,
    flatten_params,
)
from anchor_bench.schema.base import BaseConfig
from anchor_bench.schema.mlops import MLOpsConfig
from anchor_bench.schema.parse import parse
from anchor_bench.tasks.actuator_sys_id import ActuatorSysIdTask
from anchor_bench.tasks.base_task import BaseTask
from anchor_bench.tasks.mpm_sys_id.task import MpmSysIdTask
from anchor_bench.tasks.train.task import TrainingTask
from anchor_bench.tasks.trajectory import TrajectoryTask

logger = logging.getLogger(__name__)

TASKS: Dict[str, Type[BaseTask[Any]]] = {
    MpmSysIdTask.name(): MpmSysIdTask,
    ActuatorSysIdTask.name(): ActuatorSysIdTask,
    TrainingTask.name(): TrainingTask,
    TrajectoryTask.name(): TrajectoryTask,
}


class TaskRunner:
    def __init__(
        self,
        task_registry: Optional[Mapping[str, Type[BaseTask[Any]]]] = None,
        *,
        mlops_backends: Optional[Mapping[str, Callable[..., MLOpsBackend]]] = None,
    ) -> None:
        self._tasks = dict(TASKS if task_registry is None else task_registry)
        self._mlops_backends: Dict[str, Callable[..., MLOpsBackend]] = {
            "mlflow": MLflowBackend,
            **dict(mlops_backends or {}),
        }

    def run(
        self,
        config_path: str,
        *,
        force: bool = False,
        resume: bool = False,
        expected_type: Optional[str] = None,
        output: Optional[Union[str, Path]] = None,
    ) -> Path:
        logger.debug("Parsing task configuration path=%s", config_path)
        config = parse(config_path)
        task_type = config.type
        if expected_type is not None and task_type != expected_type:
            raise AnchorBenchError(
                f"Expected task type {expected_type}, got {task_type}"
            )
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
            initialize_config=False,
        )
        task = task_class(config)
        with task.output_context(output_directory, force=force, resume=resume):
            already_completed = (output_directory / COMPLETION_MARKER).exists()
            if already_completed and not force:
                logger.info(
                    "Skipping completed task type=%s output=%s",
                    task_type,
                    output_directory,
                )
                return output_directory
            if already_completed:
                logger.warning(
                    "Forcing completed task type=%s output=%s",
                    task_type,
                    output_directory,
                )

            completion_marker = output_directory / COMPLETION_MARKER
            completion_marker.unlink(missing_ok=True)
            write_yaml(
                output_directory / "config.yaml",
                config.model_dump(mode="json"),
            )
            logger.debug(
                "Wrote resolved configuration path=%s", output_directory / "config.yaml"
            )

            logger.info("Starting task type=%s output=%s", task_type, output_directory)
            mlops_config = getattr(config, "mlops", None)
            if isinstance(mlops_config, MLOpsConfig):
                result = self._run_with_mlops(
                    task,
                    config,
                    mlops_config,
                    output_directory,
                    completion_marker,
                )
            else:
                result = self._finalize_task(
                    task,
                    task.run(output_directory),
                    output_directory,
                    completion_marker,
                )
            logger.info("Completed task type=%s output=%s", task_type, result)
            return result

    def _run_with_mlops(
        self,
        task: BaseTask[Any],
        config: BaseConfig,
        mlops_config: MLOpsConfig,
        output_directory: Path,
        completion_marker: Path,
    ) -> Path:
        """Execute and finalize a task inside its backend-owned run context.

        Validation and marker creation deliberately happen before the backend
        context exits, so the tracking service observes finalization failures.
        """

        backend_factory = self._mlops_backends.get(mlops_config.backend)
        if backend_factory is None:
            known_backends = ", ".join(sorted(self._mlops_backends)) or "none"
            raise MLOpsBackendNotFoundError(
                f"Unknown MLOps backend {mlops_config.backend!r}. "
                f"Available backends: {known_backends}."
            )
        backend = backend_factory(**mlops_config.options)
        tags = _tracking_tags(config)
        logger.info(
            "Starting MLOps run backend=%s experiment=%s",
            mlops_config.backend,
            mlops_config.experiment_name,
        )
        try:
            with backend.start_run(
                mlops_config.experiment_name,
                run_name=mlops_config.run_name,
                tags=tags,
            ) as run:
                parameters = flatten_params(config.model_dump(mode="json"))
                parameters["anchor_bench/mlops/backend"] = mlops_config.backend
                run.log_params(parameters)
                with bind_active_run(run):
                    task_output = task.run(output_directory)
                return self._finalize_task(
                    task,
                    task_output,
                    output_directory,
                    completion_marker,
                )
        except BaseException:
            # Backend __exit__ can fail after the marker was touched. Never leave
            # a locally complete result when tracking finalization did not finish.
            completion_marker.unlink(missing_ok=True)
            raise

    @staticmethod
    def _finalize_task(
        task: BaseTask[Any],
        task_output: Path,
        output_directory: Path,
        completion_marker: Path,
    ) -> Path:
        """Validate the task contract and mark its assigned output complete."""

        if task_output != output_directory:
            raise RuntimeError(
                f"{type(task).__name__}.run() returned {task_output}, expected "
                f"its assigned output directory {output_directory}."
            )
        completion_marker.touch()
        logger.debug("Created completion marker path=%s", completion_marker)
        return task_output


def _tracking_tags(config: BaseConfig) -> Dict[str, str]:
    """Translate portable task labels into searchable backend string tags."""

    raw_tags = getattr(config, "tags", None)
    tags = {tag: "" for tag in raw_tags or [] if isinstance(tag, str) and tag}
    tags["anchor_bench.task_type"] = config.type

    description = getattr(config, "description", None)
    if isinstance(description, str) and description:
        tags["anchor_bench.description"] = description
    return tags
