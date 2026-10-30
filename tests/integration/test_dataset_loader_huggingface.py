# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from anchor_bench.data import DatasetLoader
from anchor_bench.schema.dataset import HuggingFaceDatasetSource

ANCHOR_LAB_REVISION = "be6010113e7b769e68f2779fbdf321908725f929"
ANCHOR_LAB_SOURCE = HuggingFaceDatasetSource(
    provider="huggingface",
    repo_id="nvidia/Anchor-Lab",
    revision=ANCHOR_LAB_REVISION,
    paths=["data/so101_arm_50motion/so101-sysid-50motion-train-wave-motion.parquet"],
)
EXPECTED_EXPERIMENT = "so101-sysid-50motion-train-wave-motion"
EXPECTED_JOINTS = {
    "elbow",
    "jaw",
    "pitch",
    "rotation",
    "wrist_pitch",
    "wrist_roll",
}


@pytest.mark.integration
def test_downloads_loads_and_resamples_real_anchor_lab_parquet() -> None:
    """Download and resample one pinned real SO-101 recording from Hugging Face."""

    aligned = DatasetLoader.load(
        ANCHOR_LAB_SOURCE,
        dt=0.02,
    )

    assert aligned.shape[1] == 48
    assert len(aligned) > 0
    assert aligned.index.names == ["experiment", "time_ns"]
    assert aligned.index.get_level_values("experiment").unique().tolist() == [
        EXPECTED_EXPERIMENT
    ]

    timestamps = aligned.index.get_level_values("time_ns").to_numpy()
    assert np.all(np.diff(timestamps) == 20_000_000)
    assert not aligned.isnull().to_numpy().any()

    expected_actual_fields = {f"{joint}/actual_q" for joint in EXPECTED_JOINTS}
    expected_command_fields = {f"{joint}/command_q" for joint in EXPECTED_JOINTS}
    assert expected_actual_fields.issubset(aligned.columns)
    assert expected_command_fields.issubset(aligned.columns)

    # The source records commands on a slower clock. Many unique resampled values
    # demonstrate that the end-to-end path performed linear interpolation.
    assert aligned["wrist_roll/command_q"].nunique() > 1_000
