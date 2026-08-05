from pathlib import Path

import torch
import yaml

from anchor_bench.schema.training import TrainingConfig
from anchor_bench.tasks.train import (
    EpochLoss,
    GruModel,
    SearchResult,
    SearchTrial,
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
        "input_features": [
            {
                "name": "input",
                "data_processor": None,
            }
        ],
        "output_features": [
            {
                "name": "target",
                "data_processor": None,
            }
        ],
        "simulation": {"backend": "newton", "usd_path": "robot.usda"},
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


def test_save_training_artifacts_writes_search_summary(tmp_path: Path) -> None:
    """Persist the objective, trials, parameters, and winning configuration."""

    config = TrainingConfig.model_validate(_config_data())
    model = GruModel(config.model, input_size=1, output_size=1)
    training_result = TrainingResult(
        model=model,
        history=[EpochLoss(epoch=1, training=2.0, validation=1.0)],
    )
    sequences = [_sequence("first", [0.0, 1.0])]
    search_result = SearchResult(
        training_result=training_result,
        best_config=config,
        validation_sequences=tuple(sequences),
        objective_metric="validation_loss",
        objective_direction="minimize",
        best_trial=1,
        best_value=1.0,
        best_parameters={"model.hidden_size": 8},
        trials=(
            SearchTrial(
                number=0,
                parameters={"model.hidden_size": 16},
                value=2.0,
            ),
            SearchTrial(
                number=1,
                parameters={"model.hidden_size": 8},
                value=1.0,
            ),
        ),
    )
    output_folder = tmp_path / "artifacts"
    output_folder.mkdir()

    save_training_artifacts(
        training_result,
        sequences,
        [],
        config,
        output_folder,
        search_result=search_result,
    )

    summary = yaml.safe_load((output_folder / "search.yaml").read_text())
    assert summary["objective"] == {
        "metric": "validation_loss",
        "direction": "minimize",
    }
    assert summary["best_trial"] == 1
    assert summary["best_parameters"] == {"model.hidden_size": 8}
    assert [trial["value"] for trial in summary["trials"]] == [2.0, 1.0]
    assert summary["best_config"]["model"]["hidden_size"] == 8
