# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end CLI coverage for SO-101 actuator system identification."""

import csv
from collections.abc import Callable, Sequence
from pathlib import Path

import newton.utils
import pytest
import yaml

from anchor_bench.schema.actuator_sys_id import ActuatorSysIdConfig
from anchor_bench.schema.parse import parse
from anchor_bench.tasks.actuator_sys_id.dataset import load_trajectories
from tests.integration.cli_runner import CliResult
from tests.integration.usd_trajectory import usd_trajectory_accuracy

pytestmark = pytest.mark.integration

TEST_DIRECTORY = Path(__file__).parent
CONFIG_TEMPLATE_PATH = TEST_DIRECTORY / "config.yaml"
NEWTON_ASSETS_REF = "a96f0973b6ae69c90609f9fafef9d2c1db2d6431"
SO101_ASSET_FOLDER = "robotstudio_so101"
SO101_USD_RELATIVE_PATH = Path("usd_structured/so101.usda")
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


@pytest.fixture(scope="session")
def source_usd_path() -> Path:
    """Download the revision-pinned SO-101 asset through Newton's cache."""

    asset_root = newton.utils.download_asset(
        SO101_ASSET_FOLDER,
        ref=NEWTON_ASSETS_REF,
    )
    source = asset_root / SO101_USD_RELATIVE_PATH
    assert source.is_file()
    return source


@pytest.fixture
def config_path(tmp_path: Path, source_usd_path: Path) -> Path:
    """Materialize the integration config with the downloaded USD path."""

    document = yaml.safe_load(CONFIG_TEMPLATE_PATH.read_text(encoding="utf-8"))
    assert document["simulation"]["usd_path"] == str(
        Path(SO101_ASSET_FOLDER) / SO101_USD_RELATIVE_PATH
    )
    document["simulation"]["usd_path"] = str(source_usd_path)

    resolved_config = tmp_path / "config.yaml"
    resolved_config.write_text(
        yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
    )
    return resolved_config


def test_runs_actuator_sysid_config(
    run_cli: Callable[[Sequence[str]], CliResult],
    config_path: Path,
    source_usd_path: Path,
    tmp_path: Path,
) -> None:
    config = parse(config_path)
    assert isinstance(config, ActuatorSysIdConfig)
    joint_names = [joint for group in config.joint_parameters for joint in group.names]
    trajectories = load_trajectories(config.dataset, joint_names=joint_names)
    assert [trajectory.experiment for trajectory in trajectories] == [
        "so101-sysid-50motion-train-random-waypoints",
        "so101-sysid-50motion-train-staircase-motion",
        "so101-sysid-50motion-train-wave-motion",
    ]

    output_directory = tmp_path / "output"
    result = run_cli(
        [
            "--config",
            str(config_path),
            "--force",
            "--output",
            str(output_directory),
        ]
    )

    assert result.exit_code == 0, result.stderr or result.stdout
    assert all(
        (output_directory / artifact).is_file() for artifact in EXPECTED_ARTIFACTS
    )
    with (output_directory / "optimization_log.csv").open(
        newline="",
        encoding="utf-8",
    ) as stream:
        best_scores = [float(row["best_mse"]) for row in csv.DictReader(stream)]
    assert best_scores[-1] < best_scores[0] * 0.1

    source_accuracy = usd_trajectory_accuracy(
        source_usd_path,
        EVALUATION_TRAJECTORY_PATH,
        config,
    )
    identified_accuracy = usd_trajectory_accuracy(
        output_directory / "identified.usda",
        EVALUATION_TRAJECTORY_PATH,
        config,
    )

    assert identified_accuracy >= source_accuracy - 0.25, (
        f"identified USD accuracy ({identified_accuracy:.6f}%) regressed by more "
        f"than 0.25 percentage points from source accuracy ({source_accuracy:.6f}%)"
    )
