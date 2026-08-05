from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
import torch

from anchor_bench.schema.training import TrainingConfig
from anchor_bench.tasks.train import (
    MappedTrainingSequence,
    SequenceDataset,
    create_data_loaders,
)


def _sequence(experiment: str, values: list[float]) -> MappedTrainingSequence:
    """Build one scalar mapped sequence with identifiable timesteps."""

    index = pd.MultiIndex.from_product(
        [[experiment], range(len(values))],
        names=["experiment", "time_ns"],
    )
    return MappedTrainingSequence(
        experiment=experiment,
        name="joint",
        inputs=pd.DataFrame({"joint/input": values}, index=index),
        targets=pd.DataFrame(
            {"joint/target": [value + 1_000.0 for value in values]},
            index=index,
        ),
    )


def _config_data() -> dict[str, Any]:
    """Build a complete configuration for data-loader tests."""

    return {
        "type": "training",
        "dataset": {
            "type": "parquet",
            "source": "dataset.parquet",
            "resampling": {"type": "uniform", "dt": 0.002},
        },
        "simulation": {"backend": "newton", "usd_path": "robot.usda"},
        "input_features": {"input": None},
        "output_features": "target",
        "mapping": {"joint": {"inputs": ["joint"], "outputs": ["joint"]}},
        "model": {
            "type": "gru",
            "hidden_size": 8,
            "num_layers": 1,
            "dropout": 0.0,
            "output": {"activation": "identity", "scale": 1.0},
        },
        "training": {
            "epochs": 1,
            "batch_size": 3,
            "seed": 7,
            "validation_ratio": 0.25,
            "sequence": {"length": 2, "stride": 1, "hidden_state": "reset"},
            "optimizer": {
                "type": "adamw",
                "learning_rate": 0.001,
                "weight_decay": 0.0,
            },
            "loss": {"type": "mse"},
            "gradient": {"clip_norm": 1.0},
            "augmentation": {"input_noise_std": 0.0},
        },
        "validation": {
            "validators": [
                {
                    "name": "loss",
                    "type": "offline",
                    "entrypoint": "validators:loss",
                }
            ]
        },
    }


def test_sequence_dataset_respects_length_stride_and_boundaries() -> None:
    """Create windows without joining the end of one experiment to the next."""

    dataset = SequenceDataset(
        [
            _sequence("first", [0.0, 1.0, 2.0, 3.0, 4.0]),
            _sequence("second", [100.0, 101.0, 102.0, 103.0, 104.0]),
        ],
        window_length=3,
        stride=2,
    )

    assert len(dataset) == 4
    assert dataset.input_size == 1
    assert dataset.output_size == 1
    assert [inputs[:, 0].tolist() for inputs, _ in dataset] == [
        [0.0, 1.0, 2.0],
        [2.0, 3.0, 4.0],
        [100.0, 101.0, 102.0],
        [102.0, 103.0, 104.0],
    ]
    inputs, targets = dataset[0]
    assert inputs.shape == (3, 1)
    assert targets.shape == (3, 1)
    assert inputs.dtype == torch.float32
    assert targets.dtype == torch.float32


def test_data_loaders_split_before_windowing_without_timestep_leakage() -> None:
    """Keep every raw timestep exclusively in train or validation windows."""

    config = TrainingConfig.model_validate(_config_data())
    loaders = create_data_loaders(
        [
            _sequence("first", [float(value) for value in range(8)]),
            _sequence("second", [float(value) for value in range(100, 108)]),
        ],
        config,
    )

    train_dataset = loaders.train.dataset
    validation_dataset = loaders.validation.dataset
    assert isinstance(train_dataset, SequenceDataset)
    assert isinstance(validation_dataset, SequenceDataset)
    assert len(train_dataset) == 10
    assert len(validation_dataset) == 2

    train_timesteps = {
        value for inputs, _ in train_dataset for value in inputs[:, 0].tolist()
    }
    validation_timesteps = {
        value for inputs, _ in validation_dataset for value in inputs[:, 0].tolist()
    }
    assert train_timesteps == set(range(6)) | set(range(100, 106))
    assert validation_timesteps == {6.0, 7.0, 106.0, 107.0}
    assert train_timesteps.isdisjoint(validation_timesteps)

    inputs, targets = next(iter(loaders.train))
    assert inputs.shape == (3, 2, 1)
    assert targets.shape == (3, 2, 1)


def test_sequence_dataset_rejects_inconsistent_tensor_widths() -> None:
    """Reject mapped sequences that cannot be collated into one batch."""

    scalar = _sequence("first", [0.0, 1.0])
    vector = _sequence("second", [2.0, 3.0])
    vector.inputs["joint/second_input"] = [4.0, 5.0]

    with pytest.raises(ValueError, match="identical input and target widths"):
        SequenceDataset([scalar, vector], window_length=2, stride=1)
