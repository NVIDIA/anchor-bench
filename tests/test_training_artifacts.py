from pathlib import Path

import torch
import yaml

from anchor_bench.schema.training import TrainingConfig
from anchor_bench.tasks.train import (
    EpochLoss,
    GruModel,
    TrainingResult,
    save_training_artifacts,
)
from anchor_bench.train.validators import ValidationResult
from tests.test_training_data import _config_data, _sequence


def test_save_training_artifacts_writes_portable_run(
    tmp_path: Path,
) -> None:
    """Persist CPU weights, history, and resolved feature ordering."""

    config = TrainingConfig.model_validate(_config_data())
    model = GruModel(config.model, input_size=1, output_size=1)
    result = TrainingResult(
        model=model,
        history=[
            EpochLoss(epoch=1, training=2.0, validation=3.0),
            EpochLoss(epoch=2, training=1.0, validation=None),
        ],
    )

    output_folder = tmp_path / "artifacts"
    output_folder.mkdir()
    saved_output_folder = save_training_artifacts(
        result,
        [
            _sequence("first", [0.0, 1.0]),
            _sequence("second", [2.0, 3.0]),
        ],
        [
            ValidationResult(
                name="held-out loss",
                type="offline",
                metrics={"loss": 0.25},
            )
        ],
        config,
        output_folder,
    )

    assert saved_output_folder == output_folder
    assert yaml.safe_load((output_folder / "history.yaml").read_text()) == [
        {"epoch": 1, "training": 2.0, "validation": 3.0},
        {"epoch": 2, "training": 1.0, "validation": None},
    ]
    assert yaml.safe_load((output_folder / "validation.yaml").read_text()) == [
        {
            "name": "held-out loss",
            "type": "offline",
            "metrics": {"loss": 0.25},
        }
    ]
    assert yaml.safe_load((output_folder / "metadata.yaml").read_text()) == {
        "format_version": 1,
        "model_type": "gru",
        "seed": 7,
        "mappings": {
            "joint": {
                "inputs": ["joint/input"],
                "targets": ["joint/target"],
            }
        },
    }

    saved_state = torch.load(
        output_folder / "model.pt",
        map_location="cpu",
        weights_only=True,
    )
    expected_state = model.state_dict()
    assert saved_state.keys() == expected_state.keys()
    assert all(tensor.device.type == "cpu" for tensor in saved_state.values())
    assert all(
        torch.equal(saved_state[name], expected_state[name]) for name in expected_state
    )
