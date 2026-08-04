from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd
import pytest
import yaml

from anchor_bench.core.task_runner import TASKS, TaskRunner
from anchor_bench.schema.training import TrainingConfig
from anchor_bench.tasks.train import TrainingTask


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
        "output_features": "actual_effort",
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


def test_training_task_runs_loader_preprocessor_then_mapping(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
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
    config = TrainingConfig.model_validate(config_data)

    def load(source: str, *, dt: float) -> pd.DataFrame:
        """Record configured loading arguments."""

        events.append(f"load:{source}:{dt}")
        return loaded

    def initialize_context(backend: str, path: str) -> object:
        """Record simulation initialization before preprocessing."""

        events.append(f"simulation:{backend}:{path}")
        return object()

    def preprocess(
        dataset: pd.DataFrame,
        preprocessors: Mapping[str, Optional[str]],
    ) -> pd.DataFrame:
        """Generate the configured position-error feature."""

        assert dataset is loaded
        assert preprocessors == config.input_features
        events.append("preprocess")
        result = dataset.copy(deep=False)
        result["shoulder/position_error"] = (
            result["shoulder/command_q"] - result["shoulder/actual_q"]
        )
        result["elbow/position_error"] = (
            result["elbow/command_q"] - result["elbow/actual_q"]
        )
        return result

    monkeypatch.setattr(
        "anchor_bench.tasks.train.task.DatasetLoader.load",
        load,
    )
    monkeypatch.setattr(
        "anchor_bench.tasks.train.task.create_simulation_context",
        initialize_context,
    )
    monkeypatch.setattr(
        "anchor_bench.tasks.train.task.PreprocessorRunner.run",
        preprocess,
    )

    sequences = TrainingTask(config).run()

    assert events == [
        "load:dataset.parquet:0.02",
        f"simulation:newton:{usd_path}",
        "preprocess",
    ]
    assert [(sequence.experiment, sequence.name) for sequence in sequences] == [
        ("wave", "arm"),
        ("step", "arm"),
    ]
    assert sequences[0].inputs.columns.tolist() == [
        "shoulder/actual_q",
        "elbow/actual_q",
        "shoulder/position_error",
        "elbow/position_error",
    ]
    assert sequences[0].targets.columns.tolist() == [
        "elbow/actual_effort",
        "shoulder/actual_effort",
    ]


def test_training_task_skips_simulation_for_existing_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Avoid creating Newton when preprocessing is not requested."""

    config_data = _config_data("dataset.parquet")
    config_data["input_features"] = {"actual_q": None}
    config = TrainingConfig.model_validate(config_data)

    monkeypatch.setattr(
        "anchor_bench.tasks.train.task.DatasetLoader.load",
        lambda source, *, dt: _aligned_dataset(),
    )
    monkeypatch.setattr(
        "anchor_bench.tasks.train.task.create_simulation_context",
        lambda backend, path: pytest.fail("simulation context should not be created"),
    )

    sequences = TrainingTask(config).run()

    assert len(sequences) == 2


def test_task_runner_registers_and_runs_training(tmp_path: Path) -> None:
    """Run the prepared-sequence pipeline through a parsed YAML config."""

    assert TASKS["training"] is TrainingTask

    dataset = pd.DataFrame(
        {
            "experiment": ["wave"] * 4,
            "time_ns": [0, 20_000_000] * 2,
            "field": [
                "shoulder/actual_q",
                "shoulder/actual_q",
                "shoulder/actual_effort",
                "shoulder/actual_effort",
            ],
            "value": [1.0, 1.1, 5.0, 5.1],
        }
    )
    dataset_path = tmp_path / "training.parquet"
    dataset.to_parquet(dataset_path)
    config_data = _config_data(str(dataset_path))
    config_data["input_features"] = {"actual_q": None}
    config_data["mapping"] = {
        "shoulder": {"inputs": ["shoulder"], "outputs": ["shoulder"]}
    }
    config_path = tmp_path / "training.yaml"
    config_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    sequences = TaskRunner().run(str(config_path))

    assert len(sequences) == 1
    assert sequences[0].experiment == "wave"
    assert sequences[0].name == "shoulder"
    assert sequences[0].inputs.columns.tolist() == ["shoulder/actual_q"]
    assert sequences[0].targets.columns.tolist() == ["shoulder/actual_effort"]
    assert sequences[0].inputs.index.get_level_values("time_ns").tolist() == [
        0,
        20_000_000,
    ]
