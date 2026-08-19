# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from anchor_bench.schema.parse import parse
from anchor_bench.schema.training import TrainingConfig

CONFIG_DIRECTORY = Path(__file__).parent
JOINTS = [
    "/g1_29dof_with_hand_rev_1_0/joints/right_shoulder_pitch_joint",
    "/g1_29dof_with_hand_rev_1_0/joints/right_shoulder_roll_joint",
    "/g1_29dof_with_hand_rev_1_0/joints/right_shoulder_yaw_joint",
    "/g1_29dof_with_hand_rev_1_0/joints/right_elbow_joint",
]


def test_g1_residual_training_config() -> None:
    """Keep the residual model contract and feature order executable."""

    config = parse(CONFIG_DIRECTORY / "g1_arm_residual.yaml")

    assert isinstance(config, TrainingConfig)
    assert list(config.input_features) == [
        "actual_q",
        "position_error",
        "actual_dq",
    ]
    assert config.mapping["right_arm"].inputs == JOINTS
    assert config.mapping["right_arm"].outputs == JOINTS
    assert config.model.hidden_size == 128
    assert config.model.num_layers == 2
    assert config.model.output.activation == "tanh"
    assert config.model.output.scale == 5.0
    assert config.validation.validators[0].config["disable_contacts"] is True


def test_g1_full_torque_training_config() -> None:
    """Keep the enriched autoregressive model contract executable."""

    config = parse(CONFIG_DIRECTORY / "g1_arm_full_torque.yaml")

    assert isinstance(config, TrainingConfig)
    assert list(config.input_features) == [
        "actual_q",
        "position_error",
        "actual_dq",
        "solver_pd",
        "qfrc_bias",
        "previous_torque",
    ]
    assert config.mapping["right_arm"].inputs == JOINTS
    assert config.model.hidden_size == 192
    assert config.model.num_layers == 2
    assert config.model.output.activation == "tanh"
    assert config.model.output.scale == 15.0
    assert config.training.feedback is not None
    assert config.training.feedback.input == "previous_torque"
    assert config.training.feedback.output == "actual_effort"
    assert config.validation.validators[0].config["disable_contacts"] is True
