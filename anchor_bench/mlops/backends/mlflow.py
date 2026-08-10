"""Bundled MLflow plugin for the Anchor Bench MLOps interfaces."""

from __future__ import annotations

import importlib
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from types import ModuleType
from typing import Generator, Mapping, Optional, Union

import pandas as pd

from anchor_bench.errors import MLOpsDependencyError
from anchor_bench.mlops.base import MLOpsBackend, MLOpsRun, ParameterValue

# MLflow's native digest field accepts only 36 characters. The full SHA-256 is
# retained separately as portable run metadata and as an input tag.
_MLFLOW_DATASET_DIGEST_MAX_LENGTH = 36
_FULL_DATASET_DIGEST_TAG = "anchor_bench.dataset_digest"


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

    def log_dataset(
        self,
        dataset: pd.DataFrame,
        *,
        source: str,
        digest: str,
        name: str = "dataset",
        context: str = "input",
        path: Optional[Union[str, Path]] = None,
        metadata: Optional[Mapping[str, ParameterValue]] = None,
        upload: bool = False,
    ) -> None:
        """Record portable lineage plus MLflow's native dataset-input metadata.

        The base implementation owns optional payload upload; this adapter adds
        MLflow's searchable input relationship without changing that policy.
        """

        super().log_dataset(
            dataset,
            source=source,
            digest=digest,
            name=name,
            context=context,
            path=path,
            metadata=metadata,
            upload=upload,
        )
        mlflow_dataset = self._mlflow.data.from_pandas(
            dataset,
            source=source,
            name=name,
            digest=digest[:_MLFLOW_DATASET_DIGEST_MAX_LENGTH],
        )
        input_tags = (
            {key: str(value) for key, value in metadata.items()}
            if metadata is not None
            else {}
        )
        input_tags[_FULL_DATASET_DIGEST_TAG] = digest
        self._mlflow.log_input(
            mlflow_dataset,
            context=context,
            tags=input_tags,
        )

    def log_artifact(
        self,
        path: Union[str, Path],
        *,
        collection: Optional[str] = None,
        artifact_type: str = "anchor-bench",
    ) -> None:
        """Log a path, mapping its collection to an MLflow artifact directory.

        MLflow file artifacts do not have an artifact-type field, so that
        backend-neutral hint is intentionally unused by this adapter.
        """

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
    """Entry-point-loaded MLflow backend with lazily imported dependencies."""

    def __init__(
        self,
        *,
        tracking_uri: Optional[str] = None,
        registry_uri: Optional[str] = None,
    ) -> None:
        """Configure optional URI overrides for runs created by this backend.

        The tracking URI selects the MLflow Tracking Server or backend store for
        experiment and run metadata. The registry URI can independently select
        the backend used by MLflow's model registry; when omitted, the SDK's
        current registry setting remains in effect.

        MLflow exposes both settings through process-global state. Overrides are
        therefore applied only while a run context is open and the previous SDK
        values are restored when it exits.
        """

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
        """Start one MLflow run under the requested experiment."""

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
        """Apply URI overrides for exactly one run and then restore SDK state."""

        # Snapshot only settings this backend will override. A None option means
        # the caller intentionally wants to keep the SDK's current configuration.
        previous_tracking_uri = (
            self._mlflow.get_tracking_uri() if self._tracking_uri is not None else None
        )
        previous_registry_uri = (
            self._mlflow.get_registry_uri() if self._registry_uri is not None else None
        )
        try:
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
        finally:
            # Keep restoration independent: a registry failure must not prevent
            # the tracking URI from being returned to its previous value.
            try:
                if previous_registry_uri is not None:
                    self._mlflow.set_registry_uri(previous_registry_uri)
            finally:
                if previous_tracking_uri is not None:
                    self._mlflow.set_tracking_uri(previous_tracking_uri)


def _import_mlflow() -> ModuleType:
    """Import the optional SDK only when the MLflow backend is constructed."""

    try:
        return importlib.import_module("mlflow")
    except ImportError as error:
        raise MLOpsDependencyError(
            "The MLflow plugin requires the optional 'mlflow' "
            "dependency. Install it with `pip install 'anchor-bench[mlflow]'`."
        ) from error
