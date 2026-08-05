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
        run.log_artifact(artifact, artifact_path="model")

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
            {"name": "model", "type": "anchor-bench"},
        ),
        ("exit", None),
    ]
