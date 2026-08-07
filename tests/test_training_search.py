from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import optuna
import pytest
from torch import nn

from anchor_bench.schema.training import TrainingConfig
from anchor_bench.tasks.train import (
    EpochLoss,
    MappedTrainingSequence,
    TrainingResult,
    TrainingTask,
    fit_normalization,
)
from tests.test_training_data import _config_data, _sequence


def _search_config() -> TrainingConfig:
    """Build a small search over every supported parameter type."""

    data = _config_data()
    data["search"] = {
        "schema_version": 1,
        "method": "optuna",
        "trials": 3,
        "objective": {
            "metric": "validation_loss",
            "direction": "minimize",
        },
        "search_space": {
            "model.hidden_size": {
                "data_type": "integer",
                "min": 2,
                "max": 4,
                "step": 1,
                "scale": "linear",
            },
            "training.optimizer.learning_rate": {
                "data_type": "float",
                "min": 0.0001,
                "max": 0.01,
                "scale": "log",
            },
            "model.output.activation": {
                "data_type": "categorical",
                "values": ["identity"],
            },
        },
    }
    return TrainingConfig.model_validate(data)


def test_search_trains_trials_and_retains_best_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Revalidate sampled configs and retain the lowest validation-loss run."""

    dataset = [_sequence("experiment", [0.0, 1.0])]
    normalization = fit_normalization(
        dataset,
        inputs="none",
        outputs="none",
    )
    trained_configs = []
    checkpoint_directories = []

    class ScoredModel(nn.Linear):
        """Attach the sampled score to a small real module."""

        def __init__(self, value: float) -> None:
            super().__init__(1, 1)
            self.search_value = value

    class StubTrainer:
        """Return a deterministic score from each sampled model width."""

        def __init__(
            self,
            config: TrainingConfig,
            sequences: Sequence[MappedTrainingSequence],
            *,
            checkpoint_directory: Path,
        ) -> None:
            trained_configs.append(config)
            self.config = config
            checkpoint_directories.append(checkpoint_directory)
            self.data_loaders = SimpleNamespace(validation_sequences=tuple(sequences))

        def train(self) -> TrainingResult:
            value = float(self.config.model.hidden_size)
            return TrainingResult(
                model=ScoredModel(value),
                history=[EpochLoss(epoch=1, training=value + 1.0, validation=value)],
                normalization=normalization,
            )

    monkeypatch.setattr("anchor_bench.tasks.train.trainer.Trainer", StubTrainer)

    result = TrainingTask(_search_config())._search(dataset, tmp_path)

    assert len(trained_configs) == 3
    assert all(config.search is None for config in trained_configs)
    assert [path.name for path in checkpoint_directories] == [
        "trial-0000",
        "trial-0001",
        "trial-0002",
    ]
    assert all(
        path.parent == tmp_path / "checkpoints" for path in checkpoint_directories
    )
    assert len(result.trials) == 3
    assert result.best_value == min(trial.value for trial in result.trials)
    assert (
        result.best_config.model.hidden_size
        == result.best_parameters["model.hidden_size"]
    )
    assert result.training_result.model.search_value == result.best_value
    assert result.validation_sequences == tuple(dataset)


def test_search_prunes_non_finite_loss_before_tracking_trial(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Allow a finite trial to win after the first trial returns NaN."""

    dataset = [_sequence("experiment", [0.0, 1.0])]
    normalization = fit_normalization(
        dataset,
        inputs="none",
        outputs="none",
    )
    validation_losses = iter([float("nan"), 2.0, 1.0])

    class StubTrainer:
        def __init__(
            self,
            _config: TrainingConfig,
            sequences: Sequence[MappedTrainingSequence],
            *,
            checkpoint_directory: Path,
        ) -> None:
            self.data_loaders = SimpleNamespace(validation_sequences=tuple(sequences))

        def train(self) -> TrainingResult:
            value = next(validation_losses)
            return TrainingResult(
                model=nn.Linear(1, 1),
                history=[EpochLoss(epoch=1, training=1.0, validation=value)],
                normalization=normalization,
            )

    monkeypatch.setattr("anchor_bench.tasks.train.trainer.Trainer", StubTrainer)

    result = TrainingTask(_search_config())._search(dataset, tmp_path)

    assert [trial.value for trial in result.trials] == [2.0, 1.0]
    assert result.best_value == 1.0
    assert result.best_trial == 2


def test_search_rejects_unknown_parameter_path(tmp_path: Path) -> None:
    """Fail clearly when a search-space path is absent from the YAML config."""

    config = _search_config()
    data = config.model_dump(mode="python", exclude_none=True)
    parameter = data["search"]["search_space"].pop("model.hidden_size")
    data["search"]["search_space"]["model.missing"] = parameter
    task = TrainingTask(TrainingConfig.model_validate(data))
    study = optuna.create_study()

    with pytest.raises(ValueError, match="does not exist"):
        task._trial_config(study.ask())


def test_search_requires_validation_loss() -> None:
    """Reject completed trials without a held-out objective value."""

    sequences = [_sequence("experiment", [0.0, 1.0])]
    result = TrainingResult(
        model=nn.Linear(1, 1),
        history=[EpochLoss(epoch=1, training=1.0, validation=None)],
        normalization=fit_normalization(
            sequences,
            inputs="none",
            outputs="none",
        ),
    )

    with pytest.raises(ValueError, match="requires validation windows"):
        TrainingTask._validation_loss(result)
