from anchor_bench.cli import main
from tests.test_parser_and_runner import VALID_YAML


def test_cli_runs_config(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(VALID_YAML, encoding="utf-8")

    assert main(["--config", str(config_path)]) == 0
