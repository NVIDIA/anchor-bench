"""Bundled MLflow backend for the Anchor Bench MLOps interfaces."""

from __future__ import annotations

import os
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from types import ModuleType
from typing import Generator, Mapping, Optional, Union

import mlflow
import pandas as pd

from anchor_bench.errors import MLOpsTrackingError
from anchor_bench.mlops.base import MLOpsBackend, MLOpsRun, ParameterValue

# MLflow's native digest field accepts only 36 characters. The full SHA-256 is
# retained separately as portable run metadata and as an input tag.
_MLFLOW_DATASET_DIGEST_MAX_LENGTH = 36
_FULL_DATASET_DIGEST_TAG = "anchor_bench.dataset_digest"
_DEFAULT_HTTP_REQUEST_TIMEOUT = 30
_DEFAULT_HTTP_REQUEST_MAX_RETRIES = 2
_HTTP_REQUEST_TIMEOUT_ENV = "MLFLOW_HTTP_REQUEST_TIMEOUT"
_HTTP_REQUEST_MAX_RETRIES_ENV = "MLFLOW_HTTP_REQUEST_MAX_RETRIES"


class MLflowRun(MLOpsRun):
    """Adapter around an active MLflow fluent-API run."""

    def __init__(self, mlflow: ModuleType) -> None:
        self._mlflow = mlflow

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
    """Backend adapter for the bundled MLflow dependency."""

    def __init__(
        self,
        *,
        tracking_uri: Optional[str] = None,
        registry_uri: Optional[str] = None,
        http_request_timeout: Optional[int] = None,
        http_request_max_retries: Optional[int] = None,
    ) -> None:
        """Configure URI and HTTP-policy overrides for this backend.

        The tracking URI selects the MLflow Tracking Server or backend store for
        experiment and run metadata. The registry URI can independently select
        the backend used by MLflow's model registry; when omitted, the SDK's
        current registry setting remains in effect.

        Remote requests default to a 30-second timeout and two retries. Existing
        MLflow environment settings take precedence over those defaults, while
        explicit constructor options take precedence over the environment.

        MLflow exposes these settings through process-global state. Overrides are
        therefore applied only while a run context is open and the previous SDK
        values are restored when it exits.
        """

        self._http_request_timeout = _validate_integer_option(
            "http_request_timeout",
            http_request_timeout,
            minimum=1,
        )
        self._http_request_max_retries = _validate_integer_option(
            "http_request_max_retries",
            http_request_max_retries,
            minimum=0,
        )
        self._mlflow = mlflow
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
        """Apply HTTP defaults around exactly one MLflow run."""

        with (
            _temporary_http_request_settings(
                timeout=self._http_request_timeout,
                max_retries=self._http_request_max_retries,
            ),
            _translate_mlflow_errors(self._mlflow),
            self._start_run_with_uris(
                experiment_name,
                run_name=run_name,
                tags=tags,
            ) as run,
        ):
            yield run

    @contextmanager
    def _start_run_with_uris(
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
            with self._mlflow.start_run(**start_options):
                yield MLflowRun(self._mlflow)
        finally:
            # Keep restoration independent: a registry failure must not prevent
            # the tracking URI from being returned to its previous value.
            try:
                if previous_registry_uri is not None:
                    self._mlflow.set_registry_uri(previous_registry_uri)
            finally:
                if previous_tracking_uri is not None:
                    self._mlflow.set_tracking_uri(previous_tracking_uri)


@contextmanager
def _temporary_http_request_settings(
    *,
    timeout: Optional[int],
    max_retries: Optional[int],
) -> Generator[None, None, None]:
    """Apply Anchor Bench defaults without replacing MLflow environment overrides."""

    settings = {
        _HTTP_REQUEST_TIMEOUT_ENV: (timeout, _DEFAULT_HTTP_REQUEST_TIMEOUT),
        _HTTP_REQUEST_MAX_RETRIES_ENV: (
            max_retries,
            _DEFAULT_HTTP_REQUEST_MAX_RETRIES,
        ),
    }
    previous_values: dict[str, Optional[str]] = {}
    try:
        for name, (configured_value, default_value) in settings.items():
            if configured_value is None and name in os.environ:
                continue
            previous_values[name] = os.environ.get(name)
            os.environ[name] = str(
                default_value if configured_value is None else configured_value
            )
        yield
    finally:
        for name, previous_value in previous_values.items():
            if previous_value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous_value


@contextmanager
def _translate_mlflow_errors(mlflow: ModuleType) -> Generator[None, None, None]:
    """Translate SDK failures while preserving unrelated task exceptions."""

    try:
        yield
    except Exception as error:
        if not _is_mlflow_exception(mlflow, error):
            raise
        raise MLOpsTrackingError(f"MLflow tracking failed: {error}") from error


def _is_mlflow_exception(mlflow: ModuleType, error: Exception) -> bool:
    """Identify failures raised by the MLflow SDK."""

    exceptions = getattr(mlflow, "exceptions", None)
    exception_type = getattr(exceptions, "MlflowException", None)
    return isinstance(exception_type, type) and isinstance(error, exception_type)


def _validate_integer_option(
    name: str,
    value: Optional[int],
    *,
    minimum: int,
) -> Optional[int]:
    """Validate an optional integer backend setting."""

    if value is not None and (
        isinstance(value, bool) or not isinstance(value, int) or value < minimum
    ):
        raise ValueError(
            f"{name} must be an integer greater than or equal to {minimum}."
        )
    return value
