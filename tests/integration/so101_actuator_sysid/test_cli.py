"""End-to-end CLI coverage for SO-101 actuator system identification."""

from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from tests.integration.cli_runner import CliResult
from tests.integration.usd_trajectory import usd_trajectory_accuracy

pytestmark = pytest.mark.integration

TEST_DIRECTORY = Path(__file__).parent
CONFIG_PATH = TEST_DIRECTORY / "config.yaml"
SOURCE_USD_PATH = TEST_DIRECTORY / "so101.usd"
IDENTIFIED_USD_PATH = TEST_DIRECTORY / "identified.usda"
EVALUATION_TRAJECTORY_PATH = (
    TEST_DIRECTORY / "trajectories" / "heldout-square-pattern.parquet"
)
EXPECTED_ARTIFACTS = (
    ".completed",
    "best_params.yaml",
    "optimization_log.csv",
    "run_summary.yaml",
    "identified.usda",
)


def test_runs_actuator_sysid_config(
    run_cli: Callable[[Sequence[str]], CliResult],
) -> None:
    result = run_cli(
        [
            "--config",
            str(CONFIG_PATH),
            "--force",
            "--output",
            str(TEST_DIRECTORY),
        ]
    )

    assert result.exit_code == 0, result.stderr or result.stdout
    assert all((TEST_DIRECTORY / artifact).is_file() for artifact in EXPECTED_ARTIFACTS)

    source_accuracy = usd_trajectory_accuracy(
        SOURCE_USD_PATH,
        EVALUATION_TRAJECTORY_PATH,
    )
    identified_accuracy = usd_trajectory_accuracy(
        IDENTIFIED_USD_PATH,
        EVALUATION_TRAJECTORY_PATH,
    )

    assert identified_accuracy > source_accuracy, (
        f"identified USD accuracy ({identified_accuracy:.6f}%) did not exceed "
        f"source USD accuracy ({source_accuracy:.6f}%)"
    )
