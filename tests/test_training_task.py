# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Optional, Sequence
from unittest.mock import Mock

import pandas as pd
import pytest
import yaml

from anchor_bench.core.task_runner import TASKS, TaskRunner
from anchor_bench.mlops import MLOpsRun, bind_active_run
from anchor_bench.schema.training import TrainingConfig
from anchor_bench.tasks.train import (
    MappedTrainingSequence,
    TrainingDataSplit,
    TrainingTask,
    ValidationDataset,
)
from anchor_bench.train import ValidationResult


def _config_data(source: str) -> dict[str, Any]:
    """Build the smallest complete training configuration used by task tests."""

    return {
        "type": "training",
        "dataset": {
            "type": "parquet",
            "source": source,
            "resampling": {"type": "uniform", "dt": 0.02},
        },
        "simulation": {
            "backend": "newton",
            "usd_path": "robot.usda",
        },
        "input_features": {
            "actual_q": None,
            "position_error": "position_error",
        },
        "output_features": {"actual_effort": None},
        "mapping": {
            "arm": {
                "inputs": ["shoulder", "elbow"],
                "outputs": ["elbow", "shoulder"],
            }
        },
        "model": {
            "type": "gru",
            "hidden_size": 8,
            "num_layers": 1,
            "dropout": 0.0,
            "output": {"activation": "identity", "scale": 1.0},
        },
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "seed": 0,
            "validation_ratio": 0.2,
            "sequence": {"length": 2},
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


def _aligned_dataset() -> pd.DataFrame:
    """Create two aligned experiments in declaration order."""

    index = pd.MultiIndex.from_tuples(
        [
            ("wave", 0),
            ("wave", 20_000_000),
            ("step", 0),
            ("step", 20_000_000),
        ],
        names=["experiment", "time_ns"],
    )
    return pd.DataFrame(
        {
            "shoulder/actual_q": [1.0, 1.1, 2.0, 2.1],
            "elbow/actual_q": [3.0, 3.1, 4.0, 4.1],
            "shoulder/command_q": [1.2, 1.4, 2.3, 2.5],
            "elbow/command_q": [3.5, 3.7, 4.6, 4.8],
            "shoulder/actual_effort": [5.0, 5.1, 6.0, 6.1],
            "elbow/actual_effort": [7.0, 7.1, 8.0, 8.1],
        },
        index=index,
    )


@pytest.mark.parametrize(
    ("search_trials", "expected_step"),
    [(None, 4), (3, 3)],
)
def test_training_task_runs_loader_data_processor_then_mapping(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    search_trials: int | None,
    expected_step: int,
) -> None:
    """Pass configured data through each preparation stage in order."""

    events: list[str] = []
    loaded = _aligned_dataset()
    usd_path = tmp_path / "robot.usda"
    usd_path.write_text("#usda 1.0", encoding="utf-8")
    config_data = _config_data("dataset.parquet")
    config_data["simulation"] = {
        "backend": "newton",
        "usd_path": str(usd_path),
    }
    config_data["output_features"] = {"torque_residual": "torque_residual"}
    config_data["training"]["epochs"] = 4
    if search_trials is not None:
        config_data["search"] = {
            "schema_version": 1,
            "method": "optuna",
            "trials": search_trials,
            "objective": {
                "metric": "validation_loss",
                "direction": "minimize",
            },
            "search_space": {
                "model.hidden_size": {
                    "data_type": "integer",
                    "min": 2,
                    "max": 4,
                    "scale": "linear",
                }
            },
        }
    config = TrainingConfig.model_validate(config_data)

    def load(
        source: str,
        *,
        dt: float,
        dataset_tracker: MLOpsRun,
        dataset_name: str,
        dataset_context: str,
        upload_dataset: bool,
        joint_name_mapping: Mapping[str, str],
        feature_name_mapping: Mapping[str, str],
    ) -> pd.DataFrame:
        """Record configured loading arguments."""

        assert dataset_tracker is mlops_run
        assert dataset_name == "training-dataset"
        assert dataset_context == "training"
        assert upload_dataset is False
        assert joint_name_mapping == {}
        assert feature_name_mapping == {}
        events.append(f"load:{source}:{dt}")
        return loaded

    def initialize_context(
        backend: str,
        path: str,
        *,
        fixed_base: bool,
    ) -> object:
        """Record simulation initialization before preprocessing."""

        events.append(f"simulation:{backend}:{path}")
        assert fixed_base is False
        return object()

    def preprocess(
        dataset: pd.DataFrame,
        data_processors: Mapping[str, Optional[str]],
    ) -> pd.DataFrame:
        """Generate the configured position-error feature."""

        if data_processors == config.output_features:
            events.append("process-output")
            result = dataset.copy(deep=False)
            result["shoulder/torque_residual"] = result["shoulder/actual_effort"]
            result["elbow/torque_residual"] = result["elbow/actual_effort"]
            return result
        assert dataset is loaded
        assert data_processors == config.input_features
        events.append("process-input")
        result = dataset.copy(deep=False)
        result["shoulder/position_error"] = (
            result["shoulder/command_q"] - result["shoulder/actual_q"]
        )
        result["elbow/position_error"] = (
            result["elbow/command_q"] - result["elbow/actual_q"]
        )
        return result

    prepared: list[MappedTrainingSequence] = []
    trained_model = object()
    validation_results = [
        ValidationResult(
            name="held-out loss",
            type="offline",
            metrics={"mse": 0.25},
        )
    ]

    class StubTrainer:
        """Record data passed through the trainer boundary."""

        def __init__(
            self,
            training_config: TrainingConfig,
            dataset: Sequence[MappedTrainingSequence],
            *,
            checkpoint_directory: Path,
        ) -> None:
            """Retain the prepared sequences."""

            assert training_config is config
            assert checkpoint_directory.name == "checkpoints"
            prepared.extend(dataset)
            self.data_loaders = SimpleNamespace(
                split=TrainingDataSplit((), tuple(dataset), ())
            )

        def train(self) -> SimpleNamespace:
            """Record training execution."""

            events.append("train")
            return SimpleNamespace(
                model=trained_model,
                normalization=SimpleNamespace(wrap=lambda model: model),
            )

    def validate(
        validators: object,
        *,
        model: object,
        validation_dataset: object,
    ) -> list[ValidationResult]:
        """Record post-training validator execution."""

        assert validators == config.validation.validators
        assert model is trained_model
        assert isinstance(validation_dataset, ValidationDataset)
        assert validation_dataset.sequences == tuple(prepared)
        assert "shoulder/command_q" in validation_dataset.dataframe
        assert "elbow/command_q" in validation_dataset.dataframe
        events.append("validate")
        return validation_results

    def save(
        result: object,
        sequences: object,
        received_validation_results: object,
        training_config: object,
        output_directory: object,
        *,
        search_result: object = None,
        data_split: TrainingDataSplit,
        evaluation_split: str,
    ) -> Path:
        """Record validation result persistence."""

        assert received_validation_results is validation_results
        assert (search_result is None) == (search_trials is None)
        assert data_split.validation == tuple(prepared)
        assert evaluation_split == "validation"
        events.append("save")
        return tmp_path / "artifacts"

    if search_trials is not None:

        def search(
            _task: TrainingTask,
            dataset: Sequence[MappedTrainingSequence],
            output_directory: Path,
        ) -> SimpleNamespace:
            """Return an inert search result for logging-step coverage."""

            prepared.extend(dataset)
            events.append("train")
            training_result = SimpleNamespace(
                model=trained_model,
                normalization=SimpleNamespace(wrap=lambda model: model),
            )
            return SimpleNamespace(
                training_result=training_result,
                split=TrainingDataSplit((), tuple(dataset), ()),
                best_config=config,
            )

        monkeypatch.setattr(TrainingTask, "_search", search)

    monkeypatch.setattr(
        "anchor_bench.tasks.train.task.DatasetLoader.load",
        load,
    )
    monkeypatch.setattr(
        "anchor_bench.tasks.train.task.create_simulation_context",
        initialize_context,
    )
    monkeypatch.setattr(
        "anchor_bench.tasks.train.task.DataProcessorRunner.run",
        preprocess,
    )
    monkeypatch.setattr(
        "anchor_bench.tasks.train.trainer.Trainer",
        StubTrainer,
    )
    monkeypatch.setattr(
        "anchor_bench.tasks.train.validator_runner.ValidatorRunner.run",
        validate,
    )
    monkeypatch.setattr(
        "anchor_bench.tasks.train.artifacts.save_training_artifacts",
        save,
    )

    output_directory = tmp_path / "output"
    output_directory.mkdir()
    mlops_run = Mock(spec=MLOpsRun)
    with bind_active_run(mlops_run):
        result = TrainingTask(config).run(output_directory)
    trajectories = pd.read_parquet(result / "trajectories.parquet")

    assert events == [
        "load:dataset.parquet:0.02",
        f"simulation:newton:{usd_path}",
        "process-input",
        "process-output",
        "train",
        "validate",
        "save",
    ]
    assert result == output_directory
    assert trajectories["experiment"].drop_duplicates().tolist() == ["wave", "step"]
    assert trajectories["mapping"].drop_duplicates().tolist() == ["arm"]
    wave_inputs = trajectories.loc[
        (trajectories["sequence"] == 0) & (trajectories["kind"] == "input")
    ]
    assert wave_inputs["feature"].drop_duplicates().tolist() == [
        "shoulder/actual_q",
        "elbow/actual_q",
        "shoulder/position_error",
        "elbow/position_error",
    ]
    wave_targets = trajectories.loc[
        (trajectories["sequence"] == 0) & (trajectories["kind"] == "target")
    ]
    assert wave_targets["feature"].drop_duplicates().tolist() == [
        "elbow/torque_residual",
        "shoulder/torque_residual",
    ]
    assert [(sequence.experiment, sequence.name) for sequence in prepared] == [
        ("wave", "arm"),
        ("step", "arm"),
    ]
    mlops_run.log_metrics.assert_called_once_with(
        {"validation/held-out loss/mse": 0.25},
        step=expected_step,
    )
    mlops_run.log_artifact.assert_called_once_with(
        output_directory,
        collection="training-output",
        artifact_type="model",
    )


def test_training_task_initializes_simulation_for_validators(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Create the shared simulation context even without preprocessing."""

    config_data = _config_data("dataset.parquet")
    config_data["input_features"] = {"actual_q": None}
    config = TrainingConfig.model_validate(config_data)

    monkeypatch.setattr(
        "anchor_bench.tasks.train.task.DatasetLoader.load",
        lambda source, *, dt, **kwargs: _aligned_dataset(),
    )
    simulation_calls = []
    monkeypatch.setattr(
        "anchor_bench.tasks.train.task.create_simulation_context",
        lambda backend, path, *, fixed_base: simulation_calls.append((backend, path)),
    )

    class StubTrainer:
        """Avoid running PyTorch in the validator initialization test."""

        def __init__(
            self,
            config: object,
            dataset: Sequence[MappedTrainingSequence],
            *,
            checkpoint_directory: Path,
        ) -> None:
            """Accept the prepared training inputs."""

            self.data_loaders = SimpleNamespace(
                split=TrainingDataSplit((), tuple(dataset), ())
            )
            assert checkpoint_directory.name == "checkpoints"

        def train(self) -> SimpleNamespace:
            """Return an inert training result."""

            return SimpleNamespace(
                model=object(),
                normalization=SimpleNamespace(wrap=lambda model: model),
            )

    monkeypatch.setattr(
        "anchor_bench.tasks.train.trainer.Trainer",
        StubTrainer,
    )
    monkeypatch.setattr(
        "anchor_bench.tasks.train.validator_runner.ValidatorRunner.run",
        lambda validators, *, model, validation_dataset: [],
    )
    monkeypatch.setattr(
        "anchor_bench.tasks.train.artifacts.save_training_artifacts",
        lambda result, sequences, validation_results, training_config, output_directory, **kwargs: (
            tmp_path / "artifacts"
        ),
    )

    output_directory = tmp_path / "output"
    output_directory.mkdir()
    result = TrainingTask(config).run(output_directory)
    trajectories = pd.read_parquet(result / "trajectories.parquet")

    assert result == output_directory
    assert simulation_calls == [("newton", "robot.usda")]
    assert trajectories["sequence"].nunique() == 2


def test_task_runner_registers_and_runs_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run the prepared-sequence pipeline through a parsed YAML config."""

    assert TASKS["training"] is TrainingTask
    monkeypatch.setattr(
        "anchor_bench.core.artifacts.resolve_repo_root",
        lambda: tmp_path,
    )
    monkeypatch.setattr(
        "anchor_bench.tasks.train.task.create_simulation_context",
        lambda backend, path, *, fixed_base: object(),
    )
    evaluated = []

    def validate(validators, *, model, validation_dataset):
        """Record the final-held-out dataset selected by the task."""

        evaluated.append(validation_dataset)
        return []

    monkeypatch.setattr(
        "anchor_bench.tasks.train.validator_runner.ValidatorRunner.run",
        validate,
    )

    timestamps = [index * 20_000_000 for index in range(12)]
    dataset = pd.DataFrame(
        [
            {
                "experiment": experiment,
                "time_ns": timestamp,
                "field": f"shoulder/{feature}",
                "value": float(index + offset),
            }
            for experiment, offset in (
                ("training", 0),
                ("checkpoint", 100),
                ("final", 200),
            )
            for feature in ("actual_q", "actual_effort")
            for index, timestamp in enumerate(timestamps)
        ]
    )
    dataset_path = tmp_path / "training.parquet"
    dataset.to_parquet(dataset_path)
    config_data = _config_data(str(dataset_path))
    config_data["input_features"] = {"actual_q": None}
    config_data["training"].pop("validation_ratio")
    config_data["training"]["split"] = {
        "validation_experiments": ["checkpoint"],
        "test_experiments": ["final"],
    }
    config_data["mapping"] = {
        "shoulder": {"inputs": ["shoulder"], "outputs": ["shoulder"]}
    }
    config_path = tmp_path / "training.yaml"
    config_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    output_directory = TaskRunner().run(str(config_path))
    trajectories = pd.read_parquet(output_directory / "trajectories.parquet")

    assert output_directory.parent == tmp_path / "output"
    assert (output_directory / "config.yaml").is_file()
    assert (output_directory / ".completed").is_file()
    assert (output_directory / "model.pt").is_file()
    assert (output_directory / "history.yaml").is_file()
    assert not (output_directory / "validation.yaml").exists()
    assert (output_directory / "test.yaml").is_file()
    assert yaml.safe_load((output_directory / "splits.yaml").read_text()) == {
        "training": ["training"],
        "validation": ["checkpoint"],
        "test": ["final"],
    }
    assert (output_directory / "metadata.yaml").is_file()
    assert [sequence.experiment for sequence in evaluated[0].sequences] == ["final"]
    assert trajectories["mapping"].drop_duplicates().tolist() == ["shoulder"]
    assert trajectories.loc[
        trajectories["kind"] == "input", "feature"
    ].drop_duplicates().tolist() == ["shoulder/actual_q"]
    assert trajectories.loc[
        trajectories["kind"] == "target", "feature"
    ].drop_duplicates().tolist() == ["shoulder/actual_effort"]
    assert trajectories["time_ns"].drop_duplicates().tolist() == timestamps
