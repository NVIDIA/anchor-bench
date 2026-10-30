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

from anchor_bench.core.simulation import SimulationContext
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
CONFIG_DIRECTORY = REPOSITORY / "examples" / "system_identification"
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


@pytest.mark.parametrize("config_name", ["so101.yaml", "so101_vbd.yaml"])
def test_runs_actuator_sysid_config(
    run_cli: Callable[[Sequence[str]], CliResult],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config_name: str,
) -> None:
    # The simulation context is a process-wide singleton, so each config needs
    # a fresh one to build its own solver.
    monkeypatch.setattr(SimulationContext, "_instance", None)
    raw_config = yaml.safe_load(
        (CONFIG_DIRECTORY / config_name).read_text(encoding="utf-8")
    )
    raw_config["optimizer"]["max_iterations"] = 2
    raw_config["num_envs"] = 6
    smoke_config_path = tmp_path / config_name
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
    summary = yaml.safe_load((output_directory / "run_summary.yaml").read_text())
    assert summary["solver"]["type"] == raw_config["simulation"]["solver"]["type"]
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
    with (output_directory / "optimization_log.csv").open(
        newline="",
        encoding="utf-8",
    ) as stream:
        best_scores = [float(row["best_mse"]) for row in csv.DictReader(stream)]
    exported_mse = usd_trajectory_mse(
        output_directory / "identified.usda",
        config.dataset.source,
        config,
    )
    # The USD is replayed with MuJoCo: it reproduces a MuJoCo fit exactly, and
    # a fit from another solver to within that solver's modeling difference.
    same_solver = raw_config["simulation"]["solver"]["type"] == "warp_mujoco"
    assert exported_mse == pytest.approx(
        best_scores[-1], rel=1.0e-4 if same_solver else 0.5, abs=1.0e-7
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
