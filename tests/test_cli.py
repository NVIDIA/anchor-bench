import pytest

from anchor_bench.cli import main
from anchor_bench.errors import MLOpsTrackingError
from tests.test_parser_and_runner import VALID_YAML


def test_cli_runs_config(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"{VALID_YAML}\noutput_dir: {tmp_path / 'output'}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "anchor_bench.tasks.actuator_sys_id.ActuatorSysIdTask.run",
        lambda self, output_directory: output_directory,
    )

    assert main(["--config", str(config_path)]) == 0


def test_cli_passes_force_and_output_to_task_runner(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(VALID_YAML, encoding="utf-8")
    calls = []

    output_path = tmp_path / "named-output"

    def run(self, path, *, force=False, output=None):
        calls.append((path, force, output))
        return tmp_path / "output"

    monkeypatch.setattr("anchor_bench.cli.TaskRunner.run", run)

    assert (
        main(
            [
                "--config",
                str(config_path),
                "--force",
                "--output",
                str(output_path),
            ]
        )
        == 0
    )
    assert calls == [(str(config_path), True, str(output_path))]


def test_cli_reports_mlops_tracking_errors(monkeypatch, capsys):
    def fail_to_run(self, path, *, force=False, output=None):
        raise MLOpsTrackingError("MLflow tracking failed: connection refused")

    monkeypatch.setattr("anchor_bench.cli.TaskRunner.run", fail_to_run)

    with pytest.raises(SystemExit) as exit_error:
        main(["--config", "config.yaml"])

    captured = capsys.readouterr()
    assert exit_error.value.code == 2
    assert (
        "anchor-bench: error: MLflow tracking failed: connection refused"
        in captured.err
    )
    assert "Traceback" not in captured.err
