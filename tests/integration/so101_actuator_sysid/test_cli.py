# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# pyright: reportAttributeAccessIssue=false

"""End-to-end CLI coverage for SO-101 actuator system identification."""

import csv
import math
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
import yaml
from pxr import Usd, UsdPhysics

from anchor_bench.assets import resolve_usd_path
from anchor_bench.schema.actuator_sys_id import ActuatorSysIdConfig
from anchor_bench.schema.parse import parse
from anchor_bench.tasks.actuator_sys_id.dataset import load_trajectories
from tests.integration.cli_runner import CliResult
from tests.integration.usd_trajectory import (
    usd_trajectory_accuracy,
    usd_trajectory_mse,
)

# The OpenUSD extension modules do not publish complete Python type information.

pytestmark = pytest.mark.integration

TEST_DIRECTORY = Path(__file__).parent
REPOSITORY = TEST_DIRECTORY.parents[2]
CONFIG_PATH = REPOSITORY / "examples" / "system_identification" / "so101.yaml"
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
    persisted_config = parse(output_directory / "config.yaml")
    assert isinstance(persisted_config, ActuatorSysIdConfig)
    assert persisted_config.simulation.usd == config.simulation.usd
    identified_stage = Usd.Stage.Open(str(output_directory / "identified.usda"))
    assert identified_stage is not None
    identified_joints = {
        prim.GetName(): prim
        for prim in identified_stage.Traverse()
        if prim.IsA(UsdPhysics.Joint)
    }
    clamps_by_joint = {
        joint_name: group.controller.clamps
        for group in config.joint_parameters
        for joint_name in group.names
    }
    for joint_name in joint_names:
        joint = identified_joints[joint_name]
        drive = UsdPhysics.DriveAPI.Get(joint, "angular")
        assert drive.GetMaxForceAttr().Get() == pytest.approx(
            float(clamps_by_joint[joint_name].effort_limit)
        )
        assert joint.GetAttribute("newton:velocityLimit").Get() == pytest.approx(
            float(clamps_by_joint[joint_name].velocity_limit) / (math.pi / 180.0)
        )
        assert joint.GetAttribute("mjc:armature").Get() is None
        assert joint.GetAttribute("mjc:damping").Get() is None
        assert joint.GetAttribute("mjc:frictionloss").Get() is None
        assert not identified_stage.GetPrimAtPath(
            f"/so101/Physics/{joint_name}"
        ).IsActive()
    with (output_directory / "optimization_log.csv").open(
        newline="",
        encoding="utf-8",
    ) as stream:
        best_scores = [float(row["best_mse"]) for row in csv.DictReader(stream)]
    assert best_scores[-1] < best_scores[0]
    exported_mse = usd_trajectory_mse(
        output_directory / "identified.usda",
        config.dataset.source,
        config,
    )
    assert exported_mse == pytest.approx(best_scores[-1], rel=1.0e-4, abs=1.0e-7)

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

    assert source_accuracy > 0.0
    assert identified_accuracy > 0.0
