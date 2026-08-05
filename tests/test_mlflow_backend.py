from types import SimpleNamespace

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


class FakeMLflow:
    def __init__(self) -> None:
        self.calls = []

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
        run.log_artifact(artifact, artifact_path="models")
        run.log_artifact(artifact_directory, artifact_path="diagnostics")

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
        ("log_artifact", str(artifact), {"artifact_path": "models"}),
        (
            "log_artifacts",
            str(artifact_directory),
            {"artifact_path": "diagnostics"},
        ),
        ("exit", None),
    ]


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
