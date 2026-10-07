# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for the reproducible G1 arm training examples."""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml

from anchor_bench.schema.parse import parse
from anchor_bench.schema.training import TrainingConfig
from examples.training.prepare_dataset import (
    DATASET_GLOB,
    FILENAMES,
    prepare_dataset,
)

REPOSITORY = Path(__file__).parents[3]
CONFIG_DIRECTORY = REPOSITORY / "examples" / "training"

CONFIGS = ("g1_arm_residual.yaml", "g1_arm_full_torque.yaml")


@pytest.fixture(scope="session")
def g1_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Download the pinned G1 recordings once per test session."""

    return prepare_dataset(tmp_path_factory.mktemp("g1-data"))


@pytest.mark.integration
def test_prepares_complete_g1_dataset(g1_dataset: Path) -> None:
    """Preserve all pinned experiments and their original field names."""

    paths = sorted(g1_dataset.glob(DATASET_GLOB))
    fields = pd.concat(
        (pd.read_parquet(path, columns=["experiment", "field"]) for path in paths),
        ignore_index=True,
    )
    assert [path.name for path in paths] == sorted(FILENAMES)
    assert fields["experiment"].nunique() == len(FILENAMES)
    assert fields["field"].str.startswith("right_").all()
    assert fields["field"].str.endswith("/dq").any()
    assert fields["field"].str.endswith("/tau_est").any()


@pytest.mark.integration
@pytest.mark.parametrize("config_name", CONFIGS)
def test_trains_and_validates_g1_example(
    config_name: str,
    g1_dataset: Path,
    tmp_path: Path,
) -> None:
    """Train one production G1 config in an isolated CLI process."""

    if not torch.cuda.is_available():
        pytest.skip("G1 training integration requires CUDA.")

    config = parse(CONFIG_DIRECTORY / config_name)
    assert isinstance(config, TrainingConfig)
    data = config.model_dump(mode="json", exclude_none=True)
    data["dataset"]["source"] = str(g1_dataset / DATASET_GLOB)
    data["training"]["epochs"] = 2
    data["training"]["checkpoint"]["every_n_epochs"] = 2
    config_path = tmp_path / config_name
    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    output = tmp_path / Path(config_name).stem

    subprocess.run(
        [
            str(Path(sys.executable).with_name("anchor-bench")),
            "--force",
            "--config",
            str(config_path),
            "--output",
            str(output),
        ],
        cwd=REPOSITORY,
        check=True,
    )

    assert (output / ".completed").is_file()
    assert (output / "model.onnx").is_file()
    from newton.actuators import DriveNeuralGRU

    DriveNeuralGRU(str(output / "model.onnx"))
    assert (output / "history.yaml").is_file()
    assert (output / "validation.yaml").is_file()
    assert (output / "checkpoints" / "epoch-000002.pt").is_file()
