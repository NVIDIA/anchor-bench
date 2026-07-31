import numpy as np
import pytest

from anchor_bench.data import DatasetLoader

ANCHOR_LAB_REVISION = "be6010113e7b769e68f2779fbdf321908725f929"
ANCHOR_LAB_PARQUET_URL = (
    "https://huggingface.co/datasets/nvidia/Anchor-Lab/resolve/"
    f"{ANCHOR_LAB_REVISION}/data/so101_arm_50motion/"
    "so101-sysid-50motion-train-wave-motion.parquet"
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
def test_loads_and_aligns_real_anchor_lab_parquet() -> None:
    """Download and align one pinned real SO-101 recording from Hugging Face."""

    aligned = DatasetLoader.load(
        ANCHOR_LAB_PARQUET_URL,
        reference_field="actual_q",
        http_timeout_seconds=60.0,
    )

    assert aligned.shape == (12_454, 48)
    assert aligned.index.names == ["experiment", "time_ns"]
    assert aligned.index.get_level_values("experiment").unique().tolist() == [
        EXPECTED_EXPERIMENT
    ]

    timestamps = aligned.index.get_level_values("time_ns").to_numpy()
    assert np.all(np.diff(timestamps) > 0)
    assert not aligned.isnull().to_numpy().any()

    expected_reference_fields = {f"{joint}/actual_q" for joint in EXPECTED_JOINTS}
    expected_command_fields = {f"{joint}/command_q" for joint in EXPECTED_JOINTS}
    assert expected_reference_fields.issubset(aligned.columns)
    assert expected_command_fields.issubset(aligned.columns)

    # The source records commands on a slower clock. Many unique aligned values
    # demonstrate that the end-to-end path performed linear interpolation.
    assert aligned["wrist_roll/command_q"].nunique() > 1_000
