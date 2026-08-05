from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import pytest

from anchor_bench.core import task_runner
from anchor_bench.core.task_runner import TaskRunner
from anchor_bench.mlops import MLOpsBackend, MLOpsRun, get_active_run
from anchor_bench.mlops.base import ParameterValue
from anchor_bench.schema.base import BaseConfig
from anchor_bench.schema.mlops import MLOpsConfig
from anchor_bench.tasks.base_task import BaseTask


class TrackedConfig(BaseConfig):
    description: str
    tags: List[str]
    mlops: MLOpsConfig
    payload: Dict[str, Any]


class RecordingRun(MLOpsRun):
    def __init__(self) -> None:
        self.params: Dict[str, ParameterValue] = {}
        self.metrics = []
        self.tags = []
        self.artifacts = []

    @property
    def run_id(self) -> str:
        return "recording-run"

    def log_params(self, params: Mapping[str, ParameterValue]) -> None:
        self.params.update(params)

    def log_metrics(
        self,
        metrics: Mapping[str, float],
        *,
        step: Optional[int] = None,
    ) -> None:
        self.metrics.append((dict(metrics), step))

    def set_tags(self, tags: Mapping[str, str]) -> None:
        self.tags.append(dict(tags))

    def log_artifact(
        self,
        path: Union[str, Path],
        *,
        artifact_path: Optional[str] = None,
    ) -> None:
        self.artifacts.append((Path(path), artifact_path))


class RecordingRunContext(AbstractContextManager[MLOpsRun]):
    def __init__(self, run: RecordingRun) -> None:
        self.run = run
        self.exit_exception_type = None

    def __enter__(self) -> MLOpsRun:
        return self.run

    def __exit__(self, exception_type, exception, traceback) -> None:
        self.exit_exception_type = exception_type


class RecordingBackend(MLOpsBackend):
    def __init__(self) -> None:
        self.run = RecordingRun()
        self.context = RecordingRunContext(self.run)
        self.start_arguments = None

    def start_run(
        self,
        experiment_name: str,
        *,
        run_name: Optional[str] = None,
        tags: Optional[Mapping[str, str]] = None,
    ) -> AbstractContextManager[MLOpsRun]:
        self.start_arguments = (experiment_name, run_name, dict(tags or {}))
        return self.context


class ReportingTask(BaseTask[TrackedConfig]):
    def __init__(self, config: TrackedConfig) -> None:
        self.config = config

    @classmethod
    def name(cls) -> str:
        return "tracked"

    def run(self) -> str:
        assert self.mlops_run is not None
        self.mlops_run.log_metrics({"validation/loss": 0.25}, step=4)
        self.mlops_run.log_artifact("checkpoint.pt", artifact_path="models")
        return "complete"


def _tracked_config() -> TrackedConfig:
    return TrackedConfig(
        type="tracked",
        description="Exercise reporting.",
        tags=["baseline", "so101"],
        mlops=MLOpsConfig(
            backend="recording",
            experiment_name="actuator-training",
            options={"api_token": "must-not-be-logged"},
        ),
        payload={
            "epochs": 4,
            "nested": {"enabled": True, "missing": None},
            "empty": [],
        },
    )


def test_task_runner_owns_run_and_task_reports_to_active_run(monkeypatch):
    config = _tracked_config()
    backend = RecordingBackend()
    resolver_calls = []

    def resolve_backend(name, **options):
        resolver_calls.append((name, options))
        return backend

    monkeypatch.setattr(task_runner, "parse", lambda path: config)

    result = TaskRunner(
        {"tracked": ReportingTask},
        mlops_backend_resolver=resolve_backend,
    ).run("tracked.yaml")

    assert result == "complete"
    assert resolver_calls == [("recording", {"api_token": "must-not-be-logged"})]
    assert backend.start_arguments == (
        "actuator-training",
        None,
        {
            "baseline": "",
            "so101": "",
            "anchor_bench.task_type": "tracked",
            "anchor_bench.description": "Exercise reporting.",
        },
    )
    assert backend.run.params == {
        "type": "tracked",
        "description": "Exercise reporting.",
        "tags/0": "baseline",
        "tags/1": "so101",
        "payload/epochs": 4,
        "payload/nested/enabled": True,
        "payload/nested/missing": "null",
        "payload/empty": "[]",
        "anchor_bench/mlops/backend": "recording",
    }
    assert backend.run.metrics == [({"validation/loss": 0.25}, 4)]
    assert backend.run.artifacts == [(Path("checkpoint.pt"), "models")]
    assert backend.context.exit_exception_type is None
    assert get_active_run() is None


def test_active_run_is_reset_when_task_fails(monkeypatch):
    class FailingTask(ReportingTask):
        def run(self) -> str:
            assert self.mlops_run is not None
            raise RuntimeError("training failed")

    config = _tracked_config()
    backend = RecordingBackend()
    monkeypatch.setattr(task_runner, "parse", lambda path: config)

    with pytest.raises(RuntimeError, match="training failed"):
        TaskRunner(
            {"tracked": FailingTask},
            mlops_backend_resolver=lambda name, **options: backend,
        ).run("tracked.yaml")

    assert backend.context.exit_exception_type is RuntimeError
    assert get_active_run() is None
