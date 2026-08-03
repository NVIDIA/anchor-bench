from anchor_bench.cli import main
from tests.test_parser_and_runner import VALID_YAML


def test_cli_runs_config(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"{VALID_YAML}\noutput_dir: {tmp_path / 'output'}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "anchor_bench.tasks.actuator_sys_id.ActuatorSysIdTask.run",
        lambda self: tmp_path / "output",
    )

    assert main(["--config", str(config_path)]) == 0


def test_cli_passes_force_to_task_runner(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(VALID_YAML, encoding="utf-8")
    calls = []

    def run(self, path, *, force=False):
        calls.append((path, force))
        return tmp_path / "output"

    monkeypatch.setattr("anchor_bench.cli.TaskRunner.run", run)

    assert main(["--config", str(config_path), "--force"]) == 0
    assert calls == [(str(config_path), True)]
