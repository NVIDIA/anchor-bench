# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Download the pinned Anchor-Lab recordings used by the G1 examples."""

from __future__ import annotations

import argparse
from pathlib import Path

from anchor_bench.data import download_from_huggingface

ANCHOR_LAB_REVISION = "be6010113e7b769e68f2779fbdf321908725f929"
DATASET_DIRECTORY = Path(__file__).parent / "data"
DATASET_GLOB = "g1-arm-multijoint-*.parquet"
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


def prepare_dataset(
    output_directory: Path = DATASET_DIRECTORY,
    *,
    force: bool = False,
) -> Path:
    """Download every raw G1 motion into one glob-compatible directory."""

    output_directory = output_directory.resolve()
    for filename in FILENAMES:
        if force or not (output_directory / filename).is_file():
            download_from_huggingface(_url(filename), output_directory)
    return output_directory


def main() -> None:
    """Prepare the G1 example dataset from the command line."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    print(prepare_dataset(force=arguments.force) / DATASET_GLOB)


if __name__ == "__main__":
    main()
