import pandas as pd
import pytest

from anchor_bench.mlops.backends import wandb as wandb_backend
from anchor_bench.mlops.backends.wandb import WandbBackend


class FakeConfig(dict):
    pass


class FakeRun:
    def __init__(self, calls) -> None:
        self.id = "wandb-run-456"
        self.config = FakeConfig()
        self.tags = ("initial",)
        self.calls = calls

    def __enter__(self):
        self.calls.append(("enter",))
        return self

    def __exit__(self, exception_type, exception, traceback):
        self.calls.append(("exit", exception_type))

    def log(self, metrics, **options):
        self.calls.append(("log", metrics, options))

    def log_artifact(self, path, **options):
        self.calls.append(("log_artifact", path, options))


class FakeWandb:
    def __init__(self) -> None:
        self.calls = []
        self.run = FakeRun(self.calls)

    def init(self, **options):
        self.calls.append(("init", options))
        return self.run


def test_wandb_is_a_backend_plugin(monkeypatch, tmp_path):
    fake_wandb = FakeWandb()
    monkeypatch.setattr(wandb_backend, "_import_wandb", lambda: fake_wandb)
    artifact = tmp_path / "checkpoint.pt"
    artifact.write_text("weights", encoding="utf-8")

    backend = WandbBackend(entity="robotics", mode="offline", group="baseline")
    with backend.start_run(
        "actuator-training",
        run_name="gru-42",
        tags={"robot": "so101"},
    ) as run:
        assert run.run_id == "wandb-run-456"
        run.log_params({"hidden_size": 128})
        run.log_metrics({"loss": 0.2}, step=2)
        run.set_tags({"checkpoint": "best"})
        run.set_tags({"checkpoint": "final"})
        run.set_tags({"baseline": ""})
        run.log_artifact(
            artifact,
            collection="training-output",
            artifact_type="model",
        )

    assert fake_wandb.run.config == {"hidden_size": 128}
    assert fake_wandb.run.tags == ("initial", "checkpoint:final", "baseline")
    assert fake_wandb.calls == [
        (
            "init",
            {
                "group": "baseline",
                "project": "actuator-training",
                "name": "gru-42",
                "tags": ["robot:so101"],
                "entity": "robotics",
                "mode": "offline",
                "dir": None,
            },
        ),
        ("enter",),
        ("log", {"loss": 0.2}, {"step": 2}),
        (
            "log_artifact",
            str(artifact),
            {"name": "training-output", "type": "model"},
        ),
        ("exit", None),
    ]


def test_wandb_logs_dataset_metadata_and_opt_in_payload(monkeypatch, tmp_path):
    fake_wandb = FakeWandb()
    monkeypatch.setattr(wandb_backend, "_import_wandb", lambda: fake_wandb)
    dataset = pd.DataFrame({"field": ["actual_q"], "value": [1.0]})
    source_path = tmp_path / "dataset.parquet"
    dataset.to_parquet(source_path, index=False)

    with WandbBackend(mode="offline").start_run("dataset-lineage") as run:
        run.log_dataset(
            dataset,
            source="datasets/actuator.parquet",
            digest="abc123",
            name="training-dataset",
            context="training",
            path=source_path,
            metadata={"format": "parquet"},
            upload=True,
        )

    assert fake_wandb.run.config == {
        "datasets/training-dataset/source": "datasets/actuator.parquet",
        "datasets/training-dataset/digest": "abc123",
        "datasets/training-dataset/context": "training",
        "datasets/training-dataset/rows": 1,
        "datasets/training-dataset/columns": '["field","value"]',
        "datasets/training-dataset/uploaded": True,
        "datasets/training-dataset/metadata/format": "parquet",
    }
    assert (
        "log_artifact",
        str(source_path),
        {"name": "training-dataset", "type": "dataset"},
    ) in fake_wandb.calls


def test_wandb_artifact_must_exist(monkeypatch, tmp_path):
    fake_wandb = FakeWandb()
    monkeypatch.setattr(wandb_backend, "_import_wandb", lambda: fake_wandb)

    with (
        WandbBackend(mode="offline").start_run("test") as run,
        pytest.raises(FileNotFoundError, match="Artifact does not exist"),
    ):
        run.log_artifact(tmp_path / "missing.pt")
