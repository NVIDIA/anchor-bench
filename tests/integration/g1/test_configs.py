# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# pyright: reportAttributeAccessIssue=false

import hashlib
from pathlib import Path

import pytest
from pxr import Usd, UsdGeom, UsdPhysics

from anchor_bench.schema.parse import parse
from anchor_bench.schema.training import TrainingConfig

CONFIG_DIRECTORY = Path(__file__).parent
ROBOT_ROOT = "/g1_29dof_rev_1_0_tri_hand"
JOINTS = [
    f"{ROBOT_ROOT}/joints/right_shoulder_pitch_joint",
    f"{ROBOT_ROOT}/joints/right_shoulder_roll_joint",
    f"{ROBOT_ROOT}/joints/right_shoulder_yaw_joint",
    f"{ROBOT_ROOT}/joints/right_elbow_joint",
]
ASSET_SHA256 = "9e61f39e3ace1cd60b3cf9eb7dc5422a02584ad5b23b049d2d792bbbec2b9ddc"
RAW_JOINTS = [
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
]


def test_g1_asset_is_pinned_and_upright() -> None:
    """Keep the corrected TriHand asset identity, pose, and gravity stable."""

    asset = CONFIG_DIRECTORY / "usd" / "g1_29dof_tri_hand_closed.usd"
    with asset.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == ASSET_SHA256

    stage = Usd.Stage.Open(str(CONFIG_DIRECTORY / "usd" / "g1.usd"))
    assert stage.GetDefaultPrim().GetPath().pathString == ROBOT_ROOT

    transforms = UsdGeom.XformCache()
    pelvis = transforms.GetLocalToWorldTransform(
        stage.GetPrimAtPath(f"{ROBOT_ROOT}/pelvis")
    )
    torso = transforms.GetLocalToWorldTransform(
        stage.GetPrimAtPath(f"{ROBOT_ROOT}/torso_link")
    )
    orientation = pelvis.ExtractRotationQuat()
    assert orientation.GetReal() == pytest.approx(1.0)
    assert tuple(orientation.GetImaginary()) == pytest.approx((0.0, 0.0, 0.0))
    assert torso.ExtractTranslation()[2] > pelvis.ExtractTranslation()[2]

    scene = UsdPhysics.Scene.Get(stage, "/physicsScene")
    assert tuple(scene.GetGravityDirectionAttr().Get()) == pytest.approx(
        (0.0, 0.0, -1.0)
    )
    assert scene.GetGravityMagnitudeAttr().Get() == pytest.approx(9.81)


def test_g1_residual_training_config() -> None:
    """Keep the residual model contract and feature order executable."""

    config = parse(CONFIG_DIRECTORY / "g1_arm_residual.yaml")

    assert isinstance(config, TrainingConfig)
    assert config.dataset.source.endswith("g1-arm-multijoint-*.parquet")
    assert list(config.dataset.joint_name_mapping) == RAW_JOINTS
    assert list(config.dataset.joint_name_mapping.values()) == JOINTS
    assert config.dataset.feature_name_mapping == {
        "dq": "actual_dq",
        "tau_est": "actual_effort",
    }
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
    assert config.dataset.source.endswith("g1-arm-multijoint-*.parquet")
    assert list(config.dataset.joint_name_mapping) == RAW_JOINTS
    assert list(config.dataset.joint_name_mapping.values()) == JOINTS
    assert config.dataset.feature_name_mapping == {
        "dq": "actual_dq",
        "tau_est": "actual_effort",
    }
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
