# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end CLI coverage for SO-101 actuator system identification."""

from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
import yaml

from anchor_bench.schema.actuator_sys_id import ActuatorSysIdConfig
from anchor_bench.schema.parse import parse
from anchor_bench.tasks.actuator_sys_id.dataset import load_trajectories
from tests.integration.cli_runner import CliResult
from tests.integration.usd_trajectory import usd_trajectory_accuracy

pytestmark = pytest.mark.integration

TEST_DIRECTORY = Path(__file__).parent
REPOSITORY = TEST_DIRECTORY.parents[2]
CONFIG_PATH = REPOSITORY / "examples" / "system_identification" / "so101.yaml"
SOURCE_USD_PATH = REPOSITORY / "assets" / "robots" / "so101" / "files" / "so101.usd"
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
    tmp_path: Path,
) -> None:
    raw_config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw_config["optimizer"]["max_iterations"] = 2
    raw_config["num_envs"] = 6
    smoke_config_path = tmp_path / "so101.yaml"
    smoke_config_path.write_text(
        yaml.safe_dump(raw_config, sort_keys=False),
        encoding="utf-8",
    )

    config = parse(smoke_config_path)
    assert isinstance(config, ActuatorSysIdConfig)
    output_directory = tmp_path / "output"
    joint_names = [joint for group in config.joint_parameters for joint in group.names]
    trajectories = load_trajectories(config.dataset, joint_names=joint_names)
    assert [trajectory.experiment for trajectory in trajectories] == [
        "so101-sysid-50motion-train-random-waypoints",
        "so101-sysid-50motion-train-staircase-motion",
        "so101-sysid-50motion-train-wave-motion",
    ]

    result = run_cli(
        [
            "--config",
            str(smoke_config_path),
            "--force",
            "--output",
            str(output_directory),
        ]
    )

    assert result.exit_code == 0, result.stderr or result.stdout
    assert all(
        (output_directory / artifact).is_file() for artifact in EXPECTED_ARTIFACTS
    )

    source_accuracy = usd_trajectory_accuracy(
        SOURCE_USD_PATH,
        EVALUATION_TRAJECTORY_PATH,
        config,
    )
    identified_accuracy = usd_trajectory_accuracy(
        output_directory / "identified.usda",
        EVALUATION_TRAJECTORY_PATH,
        config,
    )

    assert identified_accuracy > source_accuracy, (
        f"identified USD accuracy ({identified_accuracy:.6f}%) did not exceed "
        f"source USD accuracy ({source_accuracy:.6f}%)"
    )
