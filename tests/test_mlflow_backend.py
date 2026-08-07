from types import SimpleNamespace

import pandas as pd
import pytest

from anchor_bench.errors import MLOpsDependencyError
from anchor_bench.mlops.backends import mlflow as mlflow_backend
from anchor_bench.mlops.backends.mlflow import MLflowBackend


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
        return "native-dataset"


class FakeMLflow:
    def __init__(self) -> None:
        self.calls = []
        self.data = FakeMLflowData()

    def set_tracking_uri(self, uri):
        self.calls.append(("set_tracking_uri", uri))

    def set_registry_uri(self, uri):
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

    def set_tags(self, tags):
        self.calls.append(("set_tags", tags))

    def log_input(self, dataset, **options):
        self.calls.append(("log_input", dataset, options))

    def log_artifact(self, path, **options):
        self.calls.append(("log_artifact", path, options))

    def log_artifacts(self, path, **options):
        self.calls.append(("log_artifacts", path, options))


def test_mlflow_backend_adapts_run_operations(monkeypatch, tmp_path):
    fake_mlflow = FakeMLflow()
    monkeypatch.setattr(mlflow_backend, "_import_mlflow", lambda: fake_mlflow)
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
        assert run.run_id == "mlflow-run-123"
        run.log_params({"hidden_size": 128, "carry_state": True})
        run.log_metrics({"loss": 0.4})
        run.log_metrics({"loss": 0.2}, step=2)
        run.set_tags({"checkpoint": "best"})
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
        ("set_tags", {"checkpoint": "best"}),
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
    ]


def test_mlflow_logs_native_dataset_input_without_upload(monkeypatch, tmp_path):
    fake_mlflow = FakeMLflow()
    monkeypatch.setattr(mlflow_backend, "_import_mlflow", lambda: fake_mlflow)
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
    assert reported_dataset is dataset
    assert options == {
        "source": "datasets/actuator.parquet",
        "name": "training-dataset",
        "digest": full_digest[:36],
    }
    assert (
        "log_input",
        "native-dataset",
        {
            "context": "training",
            "tags": {
                "format": "parquet",
                "anchor_bench.dataset_digest": full_digest,
            },
        },
    ) in fake_mlflow.calls
    assert (
        "log_params",
        {
            "datasets/training-dataset/source": "datasets/actuator.parquet",
            "datasets/training-dataset/digest": full_digest,
            "datasets/training-dataset/context": "training",
            "datasets/training-dataset/rows": 1,
            "datasets/training-dataset/columns": '["field","value"]',
            "datasets/training-dataset/uploaded": False,
            "datasets/training-dataset/metadata/format": "parquet",
        },
    ) in fake_mlflow.calls
    assert not any(call[0] == "log_artifact" for call in fake_mlflow.calls)


def test_mlflow_artifact_must_exist(monkeypatch, tmp_path):
    monkeypatch.setattr(mlflow_backend, "_import_mlflow", FakeMLflow)

    with (
        MLflowBackend().start_run("test") as run,
        pytest.raises(FileNotFoundError, match="Artifact does not exist"),
    ):
        run.log_artifact(tmp_path / "missing.pt")


def test_mlflow_dependency_is_loaded_lazily(monkeypatch):
    def missing_mlflow(name):
        assert name == "mlflow"
        raise ImportError("missing")

    monkeypatch.setattr(mlflow_backend.importlib, "import_module", missing_mlflow)

    with pytest.raises(MLOpsDependencyError, match=r"anchor-bench\[mlflow\]"):
        MLflowBackend()


def test_mlflow_persists_artifacts_and_dataset_input(tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.chdir(tmp_path)
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
        run_id = run.run_id
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
    assert tracked_run.data.params["datasets/training-dataset/digest"] == full_digest
    input_tags = {tag.key: tag.value for tag in dataset_inputs[0].tags}
    assert input_tags["mlflow.data.context"] == "training"
    assert input_tags["anchor_bench.dataset_digest"] == full_digest
