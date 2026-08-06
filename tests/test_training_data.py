from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from anchor_bench.schema.training import TrainingConfig
from anchor_bench.tasks.train import (
    MappedTrainingSequence,
    SequenceDataset,
    create_data_loaders,
    create_validation_dataset,
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
        "output_features": {"target": None},
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
            "sequence": {"length": 2},
            "optimizer": {
                "type": "adamw",
                "learning_rate": 0.001,
                "weight_decay": 0.0,
            },
            "loss": {"type": "mse"},
            "gradient": {"clip_norm": 1.0},
            "augmentation": {"input_noise_std": 0.0},
            "normalization": {"inputs": "none", "outputs": "none"},
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


def test_data_loaders_hold_out_complete_experiments() -> None:
    """Keep every experiment exclusively in train or validation windows."""

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
    assert len(train_dataset) == 4
    assert len(validation_dataset) == 4

    train_timesteps = {
        value for inputs, _ in train_dataset for value in inputs[:, 0].tolist()
    }
    validation_timesteps = {
        value for inputs, _ in validation_dataset for value in inputs[:, 0].tolist()
    }
    assert train_timesteps == set(range(100, 108))
    assert validation_timesteps == set(range(8))
    assert train_timesteps.isdisjoint(validation_timesteps)
    assert [sequence.experiment for sequence in loaders.validation_sequences] == [
        "first"
    ]
    assert [
        sequence.inputs.iloc[:, 0].tolist() for sequence in loaders.validation_sequences
    ] == [[float(value) for value in range(8)]]

    processed = pd.DataFrame(
        {
            "raw/command_q": [float(value) for value in range(8)]
            + [float(value) for value in range(100, 108)]
        },
        index=pd.MultiIndex.from_tuples(
            [("first", value) for value in range(8)]
            + [("second", value) for value in range(8)],
            names=["experiment", "time_ns"],
        ),
    )
    validation = create_validation_dataset(processed, loaders.validation_sequences)
    assert validation.sequences == tuple(loaders.validation_sequences)
    assert validation.dataframe["raw/command_q"].tolist() == [
        float(value) for value in range(8)
    ]

    inputs, targets = next(iter(loaders.train))
    assert inputs.shape == (3, 2, 1)
    assert targets.shape == (3, 2, 1)


def test_data_loaders_fit_standard_normalization_on_training_split_only() -> None:
    """Fit training statistics and retain held-out sequences in physical units."""

    config_data = _config_data()
    config_data["training"]["normalization"] = {
        "inputs": "standard",
        "outputs": "standard",
    }
    config = TrainingConfig.model_validate(config_data)
    loaders = create_data_loaders(
        [
            _sequence("validation", [float(value) for value in range(100, 108)]),
            _sequence("training", [float(value) for value in range(8)]),
        ],
        config,
    )

    expected_standard_deviation = float(np.std(np.arange(8, dtype=float)))
    assert loaders.normalization.inputs.mean == (3.5,)
    assert loaders.normalization.inputs.standard_deviation == pytest.approx(
        (expected_standard_deviation,)
    )
    assert loaders.normalization.outputs.mean == (1003.5,)
    assert loaders.normalization.outputs.standard_deviation == pytest.approx(
        (expected_standard_deviation,)
    )

    train_dataset = loaders.train.dataset
    validation_dataset = loaders.validation.dataset
    assert isinstance(train_dataset, SequenceDataset)
    assert isinstance(validation_dataset, SequenceDataset)
    train_inputs, train_targets = train_dataset[0]
    validation_inputs, validation_targets = validation_dataset[0]
    assert train_inputs[:, 0].tolist() == pytest.approx(
        [(value - 3.5) / expected_standard_deviation for value in (0.0, 1.0)]
    )
    assert train_targets[:, 0].tolist() == pytest.approx(
        [(value - 1003.5) / expected_standard_deviation for value in (1000.0, 1001.0)]
    )
    assert validation_inputs[:, 0].tolist() == pytest.approx(
        [(value - 3.5) / expected_standard_deviation for value in (100.0, 101.0)]
    )
    assert validation_targets[:, 0].tolist() == pytest.approx(
        [(value - 1003.5) / expected_standard_deviation for value in (1100.0, 1101.0)]
    )
    assert loaders.validation_sequences[0].inputs.iloc[:, 0].tolist() == [
        float(value) for value in range(100, 108)
    ]


def test_sequence_dataset_rejects_inconsistent_tensor_widths() -> None:
    """Reject mapped sequences that cannot be collated into one batch."""

    scalar = _sequence("first", [0.0, 1.0])
    vector = _sequence("second", [2.0, 3.0])
    vector.inputs["joint/second_input"] = [4.0, 5.0]

    with pytest.raises(ValueError, match="identical input and target widths"):
        SequenceDataset([scalar, vector], window_length=2, stride=1)
