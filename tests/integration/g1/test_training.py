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
from tests.integration.g1.prepare_dataset import FILENAMES, prepare_dataset

DIRECTORY = Path(__file__).parent
REPOSITORY = DIRECTORY.parents[2]

CONFIGS = ("g1_arm_residual.yaml", "g1_arm_full_torque.yaml")


@pytest.fixture(scope="session")
def g1_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Download and combine the pinned G1 recordings once per test session."""

    return prepare_dataset(
        tmp_path_factory.mktemp("g1-data") / "g1_arm_multijoint.parquet"
    )


@pytest.mark.integration
def test_prepares_complete_g1_dataset(g1_dataset: Path) -> None:
    """Preserve all pinned experiments and normalize their field names."""

    fields = pd.read_parquet(
        g1_dataset,
        columns=["experiment", "field"],
    )
    assert fields["experiment"].nunique() == len(FILENAMES)
    assert fields["field"].str.startswith("/g1_29dof_with_hand_rev_1_0/joints/").all()
    assert fields["field"].str.endswith("/actual_dq").any()
    assert fields["field"].str.endswith("/actual_effort").any()


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

    config = parse(DIRECTORY / config_name)
    assert isinstance(config, TrainingConfig)
    data = config.model_dump(mode="json", exclude_none=True)
    data["dataset"]["source"] = str(g1_dataset)
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
    assert (output / "history.yaml").is_file()
    assert (output / "validation.yaml").is_file()
    assert (output / "checkpoints" / "epoch-000005.pt").is_file()
