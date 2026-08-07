"""Bundled W&B plugin for the Anchor Bench MLOps interfaces."""

from __future__ import annotations

import importlib
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any, Generator, Mapping, Optional, Union

from anchor_bench.errors import MLOpsDependencyError
from anchor_bench.mlops.base import MLOpsBackend, MLOpsRun, ParameterValue


class WandbRun(MLOpsRun):
    """Adapter around one active W&B SDK run."""

    def __init__(self, run: Any) -> None:
        self._run = run

    @property
    def run_id(self) -> str:
        return str(self._run.id)

    def log_params(self, params: Mapping[str, ParameterValue]) -> None:
        self._run.config.update(dict(params))

    def log_metrics(
        self,
        metrics: Mapping[str, float],
        *,
        step: Optional[int] = None,
    ) -> None:
        values = dict(metrics)
        if step is None:
            self._run.log(values)
        else:
            self._run.log(values, step=step)

    def set_tags(self, tags: Mapping[str, str]) -> None:
        encoded_tags = _encode_tags(tags)
        current_tags = tuple(self._run.tags or ())
        replaced_prefixes = tuple(f"{key}:" for key, value in tags.items() if value)
        replaced_labels = {key for key, value in tags.items() if not value}
        retained_tags = tuple(
            tag
            for tag in current_tags
            if tag not in replaced_labels and not tag.startswith(replaced_prefixes)
        )
        self._run.tags = tuple(dict.fromkeys(retained_tags + encoded_tags))

    def log_artifact(
        self,
        path: Union[str, Path],
        *,
        collection: Optional[str] = None,
        artifact_type: str = "anchor-bench",
    ) -> None:
        artifact = Path(path)
        if not artifact.exists():
            raise FileNotFoundError(f"Artifact does not exist: {artifact}.")
        self._run.log_artifact(
            str(artifact),
            name=collection,
            type=artifact_type,
        )


class WandbBackend(MLOpsBackend):
    """Entry-point-loaded W&B backend with lazily imported dependencies."""

    def __init__(
        self,
        *,
        entity: Optional[str] = None,
        mode: Optional[str] = None,
        directory: Optional[Union[str, Path]] = None,
        **init_options: Any,
    ) -> None:
        self._wandb = _import_wandb()
        self._entity = entity
        self._mode = mode
        self._directory = directory
        self._init_options = init_options

    def start_run(
        self,
        experiment_name: str,
        *,
        run_name: Optional[str] = None,
        tags: Optional[Mapping[str, str]] = None,
    ) -> AbstractContextManager[MLOpsRun]:
        return self._start_run(
            experiment_name,
            run_name=run_name,
            tags=tags,
        )

    @contextmanager
    def _start_run(
        self,
        experiment_name: str,
        *,
        run_name: Optional[str],
        tags: Optional[Mapping[str, str]],
    ) -> Generator[MLOpsRun, None, None]:
        options = dict(self._init_options)
        options.update(
            {
                "project": experiment_name,
                "name": run_name,
                "tags": (list(_encode_tags(tags)) if tags is not None else None),
                "entity": self._entity,
                "mode": self._mode,
                "dir": str(self._directory) if self._directory is not None else None,
            }
        )
        with self._wandb.init(**options) as active_run:
            yield WandbRun(active_run)


def _import_wandb() -> ModuleType:
    try:
        return importlib.import_module("wandb")
    except ImportError as error:
        raise MLOpsDependencyError(
            "The W&B plugin requires the optional 'wandb' dependency. "
            "Install it with `pip install 'anchor-bench[wandb]'`."
        ) from error


def _encode_tags(tags: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(f"{key}:{value}" if value else key for key, value in tags.items())
