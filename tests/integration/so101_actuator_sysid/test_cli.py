# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end CLI coverage for SO-101 actuator system identification."""

import csv
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from anchor_bench.assets import resolve_usd_path
from anchor_bench.schema.actuator_sys_id import ActuatorSysIdConfig
from anchor_bench.schema.parse import parse
from anchor_bench.tasks.actuator_sys_id.dataset import load_trajectories
from tests.integration.cli_runner import CliResult
from tests.integration.usd_trajectory import usd_trajectory_accuracy

pytestmark = pytest.mark.integration

TEST_DIRECTORY = Path(__file__).parent
CONFIG_PATH = TEST_DIRECTORY / "config.yaml"
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
    config = parse(CONFIG_PATH)
    assert isinstance(config, ActuatorSysIdConfig)
    source_usd_path = resolve_usd_path(config.simulation.usd)
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
            str(CONFIG_PATH),
            "--force",
            "--output",
            str(output_directory),
        ]
    )

    assert result.exit_code == 0, result.stderr or result.stdout
    assert all(
        (output_directory / artifact).is_file() for artifact in EXPECTED_ARTIFACTS
    )
    persisted_config = parse(output_directory / "config.yaml")
    assert isinstance(persisted_config, ActuatorSysIdConfig)
    assert persisted_config.simulation.usd == config.simulation.usd
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
