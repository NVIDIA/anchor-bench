"""MLflow implementation of the Anchor Bench MLOps interfaces."""

from __future__ import annotations

import importlib
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from types import ModuleType
from typing import Generator, Mapping, Optional, Union

from anchor_bench.errors import MLOpsDependencyError
from anchor_bench.mlops.base import MLOpsBackend, MLOpsRun, ParameterValue


class MLflowRun(MLOpsRun):
    """Adapter around an active MLflow fluent-API run."""

    def __init__(self, mlflow: ModuleType, run_id: str) -> None:
        self._mlflow = mlflow
        self._run_id = run_id

    @property
    def run_id(self) -> str:
        return self._run_id

    def log_params(self, params: Mapping[str, ParameterValue]) -> None:
        self._mlflow.log_params(dict(params))

    def log_metrics(
        self,
        metrics: Mapping[str, float],
        *,
        step: Optional[int] = None,
    ) -> None:
        values = dict(metrics)
        if step is None:
            self._mlflow.log_metrics(values)
        else:
            self._mlflow.log_metrics(values, step=step)

    def set_tags(self, tags: Mapping[str, str]) -> None:
        self._mlflow.set_tags(dict(tags))

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

        if artifact.is_dir():
            self._mlflow.log_artifacts(
                str(artifact),
                artifact_path=collection,
            )
        else:
            self._mlflow.log_artifact(
                str(artifact),
                artifact_path=collection,
            )


class MLflowBackend(MLOpsBackend):
    """Default MLOps backend, loaded only when it is first constructed."""

    def __init__(
        self,
        *,
        tracking_uri: Optional[str] = None,
        registry_uri: Optional[str] = None,
    ) -> None:
        self._mlflow = _import_mlflow()
        self._tracking_uri = tracking_uri
        self._registry_uri = registry_uri

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
        if self._tracking_uri is not None:
            self._mlflow.set_tracking_uri(self._tracking_uri)
        if self._registry_uri is not None:
            self._mlflow.set_registry_uri(self._registry_uri)

        self._mlflow.set_experiment(experiment_name)
        start_options = {
            "run_name": run_name,
            "tags": dict(tags) if tags is not None else None,
        }
        with self._mlflow.start_run(**start_options) as active_run:
            yield MLflowRun(self._mlflow, active_run.info.run_id)


def _import_mlflow() -> ModuleType:
    try:
        return importlib.import_module("mlflow")
    except ImportError as error:
        raise MLOpsDependencyError(
            "The default MLflow backend requires the optional 'mlflow' "
            "dependency. Install it with `pip install 'anchor-bench[mlflow]'`."
        ) from error
