"""Download and normalize the pinned Anchor-Lab G1 arm recordings."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from anchor_bench.data import download_from_huggingface

ANCHOR_LAB_REVISION = "be6010113e7b769e68f2779fbdf321908725f929"
DATASET_DIRECTORY = Path(__file__).parent / "data"
DATASET_PATH = DATASET_DIRECTORY / "g1_arm_multijoint.parquet"
ROBOT_PATH = "/g1_29dof_with_hand_rev_1_0/joints"
ENTITY_NAMES = {
    "right_elbow": f"{ROBOT_PATH}/right_elbow_joint",
    "right_shoulder_pitch": f"{ROBOT_PATH}/right_shoulder_pitch_joint",
    "right_shoulder_roll": f"{ROBOT_PATH}/right_shoulder_roll_joint",
    "right_shoulder_yaw": f"{ROBOT_PATH}/right_shoulder_yaw_joint",
}
FEATURE_NAMES = {
    "dq": "actual_dq",
    "tau_est": "actual_effort",
}
FILENAMES = (
    "g1-arm-multijoint-antiphase-trial-004-20260407-120501.parquet",
    "g1-arm-multijoint-broadband-pose-b-trial-007-20260407-121437.parquet",
    "g1-arm-multijoint-fast-wave-trial-007-20260407-120730.parquet",
    "g1-arm-multijoint-multi-chirp-pose-c-trial-010-20260407-121711.parquet",
    "g1-arm-multijoint-multi-chirp-trial-008-20260407-120820.parquet",
    "g1-arm-multijoint-multi-freq-trial-002-20260407-120310.parquet",
    "g1-arm-multijoint-multisine-trial-005-20260407-120547.parquet",
    "g1-arm-multijoint-slow-wave-pose-c-trial-009-20260407-121619.parquet",
    "g1-arm-multijoint-slow-wave-trial-006-20260407-120641.parquet",
    "g1-arm-multijoint-wave-chirp-pose-b-trial-006-20260407-121344.parquet",
    "g1-arm-multijoint-wave-chirp-trial-003-20260407-120403.parquet",
    "g1-arm-multijoint-wave-sine-pose-b-trial-005-20260407-121245.parquet",
    "g1-arm-multijoint-wave-sine-pose-c-trial-008-20260407-121530.parquet",
    "g1-arm-multijoint-wave-sine-trial-001-20260407-115544.parquet",
    "g1-arm-multijoint-wave-sine-trial-001-20260407-120216.parquet",
)


def _url(filename: str) -> str:
    """Return one revision-pinned Anchor-Lab download URL."""

    return (
        "https://huggingface.co/datasets/nvidia/Anchor-Lab/resolve/"
        f"{ANCHOR_LAB_REVISION}/data/g1_arm_multijoint/{filename}"
    )


def _normalize_fields(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Map raw log entity and feature names to the training contract."""

    parts = dataframe["field"].str.rsplit("/", n=1, expand=True)
    entities = parts[0].map(ENTITY_NAMES)
    if entities.isnull().any():
        unknown = sorted(parts.loc[entities.isnull(), 0].unique())
        raise ValueError(f"Unknown G1 dataset entities: {unknown}.")
    features = parts[1].replace(FEATURE_NAMES)
    result = dataframe.copy()
    result["field"] = entities + "/" + features
    return result


def prepare_dataset(
    output_path: Path = DATASET_PATH,
    *,
    force: bool = False,
) -> Path:
    """Download, normalize, and combine all G1 motions deterministically."""

    output_path = output_path.resolve()
    if output_path.is_file() and not force:
        return output_path

    download_directory = output_path.parent / "downloads"
    paths = [
        download_from_huggingface(_url(filename), download_directory)
        for filename in FILENAMES
    ]
    dataframe = pd.concat(
        (_normalize_fields(pd.read_parquet(path)) for path in paths),
        ignore_index=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_parquet(output_path, index=False)
    return output_path


def main() -> None:
    """Prepare the default integration dataset from the command line."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    print(prepare_dataset(force=arguments.force))


if __name__ == "__main__":
    main()
