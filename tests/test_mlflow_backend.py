# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from types import SimpleNamespace

import mlflow
import pandas as pd
import pytest

from anchor_bench.errors import MLOpsTrackingError
from anchor_bench.mlops.backends import mlflow as mlflow_backend
from anchor_bench.mlops.backends.mlflow import MLflowBackend


class FakeMlflowException(Exception):
    pass


class FakeActiveRun:
    def __init__(self, calls) -> None:
        self.calls = calls
        self.info = SimpleNamespace(run_id="mlflow-run-123")

    def __enter__(self):
        self.calls.append(("enter",))
        return self

    def __exit__(self, exception_type, exception, traceback):
        self.calls.append(("exit", exception_type))


class FakeMLflowData:
    def __init__(self) -> None:
        self.arguments = None

    def from_pandas(self, dataset, **options):
        self.arguments = (dataset, options)
        return SimpleNamespace(
            source="resolved-dataset-source",
            schema="resolved-dataset-schema",
        )


class FakeMLflow:
    def __init__(self) -> None:
        self.calls = []
        self.data = FakeMLflowData()
        self.exceptions = SimpleNamespace(MlflowException=FakeMlflowException)
        self.tracking_uri = "file:///default-tracking"
        self.registry_uri = "file:///default-registry"

    def get_tracking_uri(self):
        return self.tracking_uri

    def get_registry_uri(self):
        return self.registry_uri

    def set_tracking_uri(self, uri):
        self.tracking_uri = uri
        self.calls.append(("set_tracking_uri", uri))

    def set_registry_uri(self, uri):
        self.registry_uri = uri
        self.calls.append(("set_registry_uri", uri))

    def set_experiment(self, name):
        self.calls.append(("set_experiment", name))

    def start_run(self, **options):
        self.calls.append(("start_run", options))
        return FakeActiveRun(self.calls)

    def log_params(self, params):
        self.calls.append(("log_params", params))

    def log_metrics(self, metrics, **options):
        self.calls.append(("log_metrics", metrics, options))

    def log_input(self, dataset, **options):
        self.calls.append(("log_input", dataset, options))

    def log_artifact(self, path, **options):
        self.calls.append(("log_artifact", path, options))

    def log_artifacts(self, path, **options):
        self.calls.append(("log_artifacts", path, options))


def test_mlflow_backend_adapts_run_operations(monkeypatch, tmp_path):
    fake_mlflow = FakeMLflow()
    monkeypatch.setattr(mlflow_backend, "mlflow", fake_mlflow)
    artifact = tmp_path / "model.pt"
    artifact.write_text("weights", encoding="utf-8")
    artifact_directory = tmp_path / "plots"
    artifact_directory.mkdir()

    backend = MLflowBackend(
        tracking_uri="sqlite:///tracking.db",
        registry_uri="sqlite:///registry.db",
    )
    with backend.start_run(
        "actuator-training",
        run_name="gru-42",
        tags={"robot": "so101"},
    ) as run:
        run.log_params({"hidden_size": 128, "carry_state": True})
        run.log_metrics({"loss": 0.4})
        run.log_metrics({"loss": 0.2}, step=2)
        run.log_artifact(
            artifact,
            collection="training-checkpoints",
            artifact_type="model",
        )
        run.log_artifact(
            artifact_directory,
            collection="training-output",
            artifact_type="model",
        )

    assert fake_mlflow.calls == [
        ("set_tracking_uri", "sqlite:///tracking.db"),
        ("set_registry_uri", "sqlite:///registry.db"),
        ("set_experiment", "actuator-training"),
        (
            "start_run",
            {"run_name": "gru-42", "tags": {"robot": "so101"}},
        ),
        ("enter",),
        ("log_params", {"hidden_size": 128, "carry_state": True}),
        ("log_metrics", {"loss": 0.4}, {}),
        ("log_metrics", {"loss": 0.2}, {"step": 2}),
        (
            "log_artifact",
            str(artifact),
            {"artifact_path": "training-checkpoints"},
        ),
        (
            "log_artifacts",
            str(artifact_directory),
            {"artifact_path": "training-output"},
        ),
        ("exit", None),
        ("set_registry_uri", "file:///default-registry"),
        ("set_tracking_uri", "file:///default-tracking"),
    ]
    assert fake_mlflow.tracking_uri == "file:///default-tracking"
    assert fake_mlflow.registry_uri == "file:///default-registry"


def test_mlflow_translates_sdk_errors_without_changing_the_message(monkeypatch):
    fake_mlflow = FakeMLflow()
    monkeypatch.setattr(mlflow_backend, "mlflow", fake_mlflow)

    def fail_to_set_experiment(name):
        assert name == "actuator-training"
        raise FakeMlflowException(
            "API request to http://user:secret@mlflow.example:5000 failed"
        )

    monkeypatch.setattr(fake_mlflow, "set_experiment", fail_to_set_experiment)

    with (
        pytest.raises(MLOpsTrackingError, match="MLflow tracking failed") as error,
        MLflowBackend().start_run("actuator-training"),
    ):
        pytest.fail("The run should not start.")

    assert (
        str(error.value) == "MLflow tracking failed: API request to "
        "http://user:secret@mlflow.example:5000 failed"
    )


def test_mlflow_applies_bounded_http_defaults_for_one_run(monkeypatch):
    fake_mlflow = FakeMLflow()
    monkeypatch.setattr(mlflow_backend, "mlflow", fake_mlflow)
    monkeypatch.delenv("MLFLOW_HTTP_REQUEST_TIMEOUT", raising=False)
    monkeypatch.delenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", raising=False)

    with MLflowBackend().start_run("actuator-training"):
        assert os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] == "30"
        assert os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] == "2"

    assert "MLFLOW_HTTP_REQUEST_TIMEOUT" not in os.environ
    assert "MLFLOW_HTTP_REQUEST_MAX_RETRIES" not in os.environ


def test_mlflow_http_options_override_environment_for_one_run(monkeypatch):
    fake_mlflow = FakeMLflow()
    monkeypatch.setattr(mlflow_backend, "mlflow", fake_mlflow)
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_TIMEOUT", "90")
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "5")

    with MLflowBackend().start_run("environment-policy"):
        assert os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] == "90"
        assert os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] == "5"

    with MLflowBackend(
        http_request_timeout=12,
        http_request_max_retries=0,
    ).start_run("configured-policy"):
        assert os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] == "12"
        assert os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] == "0"

    assert os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] == "90"
    assert os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] == "5"


def test_mlflow_restores_http_defaults_when_run_fails(monkeypatch):
    fake_mlflow = FakeMLflow()
    monkeypatch.setattr(mlflow_backend, "mlflow", fake_mlflow)
    monkeypatch.delenv("MLFLOW_HTTP_REQUEST_TIMEOUT", raising=False)
    monkeypatch.delenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", raising=False)

    with (
        pytest.raises(RuntimeError, match="training failed"),
        MLflowBackend().start_run("actuator-training"),
    ):
        raise RuntimeError("training failed")

    assert "MLFLOW_HTTP_REQUEST_TIMEOUT" not in os.environ
    assert "MLFLOW_HTTP_REQUEST_MAX_RETRIES" not in os.environ


def test_mlflow_rejects_invalid_http_options(monkeypatch):
    monkeypatch.setattr(mlflow_backend, "mlflow", FakeMLflow())

    with pytest.raises(ValueError, match="http_request_timeout"):
        MLflowBackend(http_request_timeout=0)
    with pytest.raises(ValueError, match="http_request_max_retries"):
        MLflowBackend(http_request_max_retries=-1)


def test_mlflow_restores_backend_uris_when_run_fails(monkeypatch):
    fake_mlflow = FakeMLflow()
    monkeypatch.setattr(mlflow_backend, "mlflow", fake_mlflow)
    backend = MLflowBackend(
        tracking_uri="sqlite:///tracking.db",
        registry_uri="sqlite:///registry.db",
    )

    with (
        pytest.raises(RuntimeError, match="training failed"),
        backend.start_run("actuator-training"),
    ):
        raise RuntimeError("training failed")

    assert fake_mlflow.tracking_uri == "file:///default-tracking"
    assert fake_mlflow.registry_uri == "file:///default-registry"
    assert fake_mlflow.calls[-3:] == [
        ("exit", RuntimeError),
        ("set_registry_uri", "file:///default-registry"),
        ("set_tracking_uri", "file:///default-tracking"),
    ]


def test_mlflow_logs_native_dataset_input_without_upload(monkeypatch, tmp_path):
    fake_mlflow = FakeMLflow()
    monkeypatch.setattr(mlflow_backend, "mlflow", fake_mlflow)
    dataset = pd.DataFrame({"field": ["actual_q"], "value": [1.0]})
    full_digest = "a" * 64
    source_path = tmp_path / "dataset.parquet"
    dataset.to_parquet(source_path, index=False)

    with MLflowBackend().start_run("dataset-lineage") as run:
        run.log_dataset(
            dataset,
            source="datasets/actuator.parquet",
            digest=full_digest,
            name="training-dataset",
            context="training",
            path=source_path,
            metadata={"format": "parquet"},
        )

    assert fake_mlflow.data.arguments is not None
    reported_dataset, options = fake_mlflow.data.arguments
    assert reported_dataset.empty
    assert reported_dataset.columns.tolist() == dataset.columns.tolist()
    assert reported_dataset.dtypes.equals(dataset.dtypes)
    assert options == {
        "source": "datasets/actuator.parquet",
        "name": "training-dataset",
        "digest": full_digest[:36],
    }
    log_input_call = next(call for call in fake_mlflow.calls if call[0] == "log_input")
    native_dataset = log_input_call[1]
    assert native_dataset.source == "resolved-dataset-source"
    assert native_dataset.name == "training-dataset"
    assert native_dataset.digest == full_digest[:36]
    assert native_dataset.schema == "resolved-dataset-schema"
    assert log_input_call[2] == {
        "context": "training",
        "tags": {
            "format": "parquet",
            "anchor_bench.dataset_digest": full_digest,
        },
    }
    assert (
        "log_params",
        {
            "datasets/training-dataset/source": "datasets/actuator.parquet",
            "datasets/training-dataset/digest": full_digest,
            "datasets/training-dataset/context": "training",
            "datasets/training-dataset/rows": 1,
            "datasets/training-dataset/columns": '["field","value"]',
            "datasets/training-dataset/metadata/format": "parquet",
        },
    ) in fake_mlflow.calls
    assert (
        "log_params",
        {"datasets/training-dataset/uploaded": False},
    ) in fake_mlflow.calls
    assert not any(call[0] == "log_artifact" for call in fake_mlflow.calls)


def test_mlflow_artifact_must_exist(monkeypatch, tmp_path):
    monkeypatch.setattr(mlflow_backend, "mlflow", FakeMLflow())

    with (
        MLflowBackend().start_run("test") as run,
        pytest.raises(FileNotFoundError, match="Artifact does not exist"),
    ):
        run.log_artifact(tmp_path / "missing.pt")


def test_mlflow_persists_artifacts_and_dataset_input(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    previous_tracking_uri = mlflow.get_tracking_uri()
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    checkpoint = tmp_path / "epoch-000001.pt"
    checkpoint.write_text("checkpoint", encoding="utf-8")
    output_directory = tmp_path / "training-output"
    output_directory.mkdir()
    (output_directory / "model.pt").write_text("model", encoding="utf-8")
    (output_directory / "metadata.yaml").write_text(
        "format_version: 1\n",
        encoding="utf-8",
    )
    dataset = pd.DataFrame({"field": ["actual_q"], "value": [1.0]})
    full_digest = "a" * 64
    dataset_path = tmp_path / "dataset.parquet"
    dataset.to_parquet(dataset_path, index=False)

    backend = MLflowBackend(tracking_uri=tracking_uri)
    with backend.start_run("artifact-persistence") as run:
        active_run = mlflow.active_run()
        assert active_run is not None
        run_id = active_run.info.run_id
        run.log_artifact(
            checkpoint,
            collection="training-checkpoints",
            artifact_type="model",
        )
        run.log_artifact(
            output_directory,
            collection="training-output",
            artifact_type="model",
        )
        run.log_dataset(
            dataset,
            source="datasets/actuator.parquet",
            digest=full_digest,
            name="training-dataset",
            context="training",
            path=dataset_path,
            metadata={"format": "parquet"},
            upload=True,
        )

    assert mlflow.get_tracking_uri() == previous_tracking_uri
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    checkpoint_artifacts = client.list_artifacts(
        run_id,
        path="training-checkpoints",
    )
    output_artifacts = client.list_artifacts(
        run_id,
        path="training-output",
    )
    dataset_artifacts = client.list_artifacts(
        run_id,
        path="training-dataset",
    )
    tracked_run = client.get_run(run_id)
    dataset_inputs = tracked_run.inputs.dataset_inputs

    assert {artifact.path for artifact in checkpoint_artifacts} == {
        "training-checkpoints/epoch-000001.pt"
    }
    assert {artifact.path for artifact in output_artifacts} == {
        "training-output/metadata.yaml",
        "training-output/model.pt",
    }
    assert {artifact.path for artifact in dataset_artifacts} == {
        "training-dataset/dataset.parquet"
    }
    assert len(dataset_inputs) == 1
    assert dataset_inputs[0].dataset.name == "training-dataset"
    assert dataset_inputs[0].dataset.digest == full_digest[:36]
    inferred_schema = mlflow.data.from_pandas(dataset.iloc[:0]).schema
    assert json.loads(dataset_inputs[0].dataset.schema) == {
        "mlflow_colspec": json.loads(inferred_schema.to_json())
    }
    assert tracked_run.data.params["datasets/training-dataset/digest"] == full_digest
    input_tags = {tag.key: tag.value for tag in dataset_inputs[0].tags}
    assert input_tags["mlflow.data.context"] == "training"
    assert input_tags["anchor_bench.dataset_digest"] == full_digest
