from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import pandas as pd
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
        self.artifacts = []

    def log_params(self, params: Mapping[str, ParameterValue]) -> None:
        self.params.update(params)

    def log_metrics(
        self,
        metrics: Mapping[str, float],
        *,
        step: Optional[int] = None,
    ) -> None:
        self.metrics.append((dict(metrics), step))

    def log_artifact(
        self,
        path: Union[str, Path],
        *,
        collection: Optional[str] = None,
        artifact_type: str = "anchor-bench",
    ) -> None:
        self.artifacts.append((Path(path), collection, artifact_type))


@pytest.mark.parametrize("upload", [False, True])
def test_default_dataset_logging_records_lineage_and_optional_payload(
    tmp_path: Path,
    upload: bool,
) -> None:
    dataset = pd.DataFrame({"experiment": ["wave", "step"], "value": [1.0, 2.0]})
    source_path = tmp_path / "dataset.parquet"
    source_path.write_text("dataset", encoding="utf-8")
    run = RecordingRun()

    run.log_dataset(
        dataset,
        source="datasets/actuator.parquet",
        digest="abc123",
        name="training-dataset",
        context="training",
        path=source_path,
        metadata={"format": "parquet", "size_bytes": 7},
        upload=upload,
    )

    assert run.params == {
        "datasets/training-dataset/source": "datasets/actuator.parquet",
        "datasets/training-dataset/digest": "abc123",
        "datasets/training-dataset/context": "training",
        "datasets/training-dataset/rows": 2,
        "datasets/training-dataset/columns": '["experiment","value"]',
        "datasets/training-dataset/uploaded": upload,
        "datasets/training-dataset/metadata/format": "parquet",
        "datasets/training-dataset/metadata/size_bytes": 7,
    }
    assert run.artifacts == (
        [(source_path, "training-dataset", "dataset")] if upload else []
    )


def test_default_dataset_logging_does_not_record_failed_upload(
    tmp_path: Path,
) -> None:
    class FailingArtifactRun(RecordingRun):
        def log_artifact(
            self,
            path: Union[str, Path],
            *,
            collection: Optional[str] = None,
            artifact_type: str = "anchor-bench",
        ) -> None:
            raise RuntimeError("upload failed")

    dataset = pd.DataFrame({"value": [1.0]})
    source_path = tmp_path / "dataset.parquet"
    source_path.write_text("dataset", encoding="utf-8")
    run = FailingArtifactRun()

    with pytest.raises(RuntimeError, match="upload failed"):
        run.log_dataset(
            dataset,
            source="datasets/actuator.parquet",
            digest="abc123",
            name="training-dataset",
            path=source_path,
            upload=True,
        )

    assert "datasets/training-dataset/uploaded" not in run.params


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

    def run(self, output_directory: Path) -> Path:
        assert self.mlops_run is not None
        self.mlops_run.log_metrics({"validation/loss": 0.25}, step=4)
        self.mlops_run.log_artifact(
            "checkpoint.pt",
            collection="training-checkpoints",
            artifact_type="model",
        )
        return output_directory


def _tracked_config() -> TrackedConfig:
    return TrackedConfig(
        type="tracked",
        description="Exercise reporting.",
        tags=["baseline", "so101"],
        mlops=MLOpsConfig(
            backend="recording",
            experiment_name="actuator-training",
            run_name="gru-seed-42",
            options={"api_token": "must-not-be-logged"},
        ),
        payload={
            "epochs": 4,
            "nested": {"enabled": True, "missing": None},
            "empty": [],
        },
    )


def test_task_runner_owns_run_and_task_reports_to_active_run(monkeypatch, tmp_path):
    config = _tracked_config()
    backend = RecordingBackend()
    resolver_calls = []

    def resolve_backend(name, **options):
        resolver_calls.append((name, options))
        return backend

    monkeypatch.setattr(task_runner, "parse", lambda path: config)
    monkeypatch.setattr(
        task_runner,
        "get_or_create_output_folder",
        lambda config, *, output_root=None, output_folder=None: (tmp_path, False),
    )

    result = TaskRunner(
        {"tracked": ReportingTask},
        mlops_backend_resolver=resolve_backend,
    ).run("tracked.yaml")

    assert result == tmp_path
    assert (result / ".completed").is_file()
    assert resolver_calls == [("recording", {"api_token": "must-not-be-logged"})]
    assert backend.start_arguments == (
        "actuator-training",
        "gru-seed-42",
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
    assert backend.run.artifacts == [
        (Path("checkpoint.pt"), "training-checkpoints", "model")
    ]
    assert backend.context.exit_exception_type is None
    assert get_active_run() is None


def test_active_run_is_reset_when_task_fails(monkeypatch, tmp_path):
    class FailingTask(BaseTask[TrackedConfig]):
        def run(self, output_directory: Path) -> Path:
            assert self.mlops_run is not None
            raise RuntimeError("training failed")

    config = _tracked_config()
    backend = RecordingBackend()
    monkeypatch.setattr(task_runner, "parse", lambda path: config)
    monkeypatch.setattr(
        task_runner,
        "get_or_create_output_folder",
        lambda config, *, output_root=None, output_folder=None: (tmp_path, False),
    )

    with pytest.raises(RuntimeError, match="training failed"):
        TaskRunner(
            {"tracked": FailingTask},
            mlops_backend_resolver=lambda name, **options: backend,
        ).run("tracked.yaml")

    assert backend.context.exit_exception_type is RuntimeError
    assert get_active_run() is None
    assert not (tmp_path / ".completed").exists()


def test_output_validation_failure_marks_tracked_run_failed(monkeypatch, tmp_path):
    class MisdirectedTask(BaseTask[TrackedConfig]):
        def run(self, output_directory: Path) -> Path:
            assert self.mlops_run is not None
            return output_directory.parent

    config = _tracked_config()
    backend = RecordingBackend()
    monkeypatch.setattr(task_runner, "parse", lambda path: config)
    monkeypatch.setattr(
        task_runner,
        "get_or_create_output_folder",
        lambda config, *, output_root=None, output_folder=None: (tmp_path, False),
    )

    with pytest.raises(
        RuntimeError,
        match="expected its assigned output directory",
    ):
        TaskRunner(
            {"tracked": MisdirectedTask},
            mlops_backend_resolver=lambda name, **options: backend,
        ).run("tracked.yaml")

    assert backend.context.exit_exception_type is RuntimeError
    assert get_active_run() is None
    assert not (tmp_path / ".completed").exists()


def test_backend_exit_failure_removes_completion_marker(monkeypatch, tmp_path):
    class FailingExitContext(RecordingRunContext):
        def __exit__(self, exception_type, exception, traceback) -> None:
            super().__exit__(exception_type, exception, traceback)
            raise OSError("run finalization failed")

    config = _tracked_config()
    backend = RecordingBackend()
    backend.context = FailingExitContext(backend.run)
    monkeypatch.setattr(task_runner, "parse", lambda path: config)
    monkeypatch.setattr(
        task_runner,
        "get_or_create_output_folder",
        lambda config, *, output_root=None, output_folder=None: (tmp_path, False),
    )

    with pytest.raises(OSError, match="run finalization failed"):
        TaskRunner(
            {"tracked": ReportingTask},
            mlops_backend_resolver=lambda name, **options: backend,
        ).run("tracked.yaml")

    assert backend.context.exit_exception_type is None
    assert get_active_run() is None
    assert not (tmp_path / ".completed").exists()


def test_artifact_failure_fails_run_without_completion_marker(monkeypatch, tmp_path):
    config = _tracked_config()
    backend = RecordingBackend()

    def fail_artifact_upload(*args, **kwargs):
        raise OSError("artifact upload failed")

    monkeypatch.setattr(backend.run, "log_artifact", fail_artifact_upload)
    monkeypatch.setattr(task_runner, "parse", lambda path: config)
    monkeypatch.setattr(
        task_runner,
        "get_or_create_output_folder",
        lambda config, *, output_root=None, output_folder=None: (tmp_path, False),
    )

    with pytest.raises(OSError, match="artifact upload failed"):
        TaskRunner(
            {"tracked": ReportingTask},
            mlops_backend_resolver=lambda name, **options: backend,
        ).run("tracked.yaml")

    assert backend.context.exit_exception_type is OSError
    assert get_active_run() is None
    assert not (tmp_path / ".completed").exists()
