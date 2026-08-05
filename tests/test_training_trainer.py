from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from anchor_bench.schema.training import TrainingConfig
from anchor_bench.tasks.train import (
    MODEL_REGISTRY,
    GruModel,
    Trainer,
    get_model,
)
from anchor_bench.tasks.train.trainer import _create_loss, _create_optimizer
from tests.test_training_data import _config_data, _sequence


def test_registry_builds_the_configured_model() -> None:
    """Resolve the YAML model type through the internal registry."""

    config = TrainingConfig.model_validate(_config_data())

    model = get_model(config.model, input_size=3, output_size=2)

    assert MODEL_REGISTRY["gru"] is GruModel
    assert isinstance(model, GruModel)
    assert model.recurrent.input_size == 3
    assert model.output.out_features == 2


def test_gru_model_returns_one_prediction_per_timestep() -> None:
    """Apply the configured activation and scale to every GRU timestep."""

    config_data = _config_data()
    config_data["model"]["hidden_size"] = 4
    config_data["model"]["num_layers"] = 2
    config_data["model"]["output"] = {
        "activation": "tanh",
        "scale": 2.0,
    }
    config = TrainingConfig.model_validate(config_data)
    model = GruModel(config.model, input_size=3, output_size=2)
    with torch.no_grad():
        model.output.weight.zero_()
        model.output.bias.fill_(1.0)

    predictions = model(torch.zeros(5, 7, 3))

    expected = torch.tanh(torch.tensor(1.0)).item() * 2.0
    assert predictions.shape == (5, 7, 2)
    assert torch.allclose(predictions, torch.full_like(predictions, expected))


@pytest.mark.parametrize(
    ("loss_config", "expected_type"),
    [
        ({"type": "mse"}, nn.MSELoss),
        ({"type": "mae"}, nn.L1Loss),
        ({"type": "huber", "delta": 0.5}, nn.HuberLoss),
    ],
)
def test_trainer_builds_configured_loss(
    loss_config: dict[str, object],
    expected_type: type[nn.Module],
) -> None:
    """Build every regression loss admitted by the training schema."""

    config_data = _config_data()
    config_data["training"]["loss"] = loss_config
    config = TrainingConfig.model_validate(config_data)
    loss = _create_loss(config.training.loss)

    assert isinstance(loss, expected_type)
    if isinstance(loss, nn.HuberLoss):
        assert loss.delta == 0.5


def test_trainer_builds_configured_optimizer() -> None:
    """Configure the optimizer from the validated YAML values."""

    config_data = _config_data()
    config_data["training"]["optimizer"] = {
        "type": "adamw",
        "learning_rate": 0.02,
        "weight_decay": 0.3,
    }
    config = TrainingConfig.model_validate(config_data)
    optimizer = _create_optimizer(
        config.training.optimizer,
        nn.Linear(1, 1).parameters(),
    )

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == 0.02
    assert optimizer.param_groups[0]["weight_decay"] == 0.3


def test_trainer_updates_weights_once_per_batch_and_reports_epochs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run optimizer updates inside the batch loop and collect epoch losses."""

    config_data = _config_data()
    config_data["training"]["epochs"] = 2
    config_data["training"]["batch_size"] = 2
    config_data["training"]["sequence"] = {
        "length": 2,
        "stride": 2,
        "hidden_state": "reset",
    }
    config = TrainingConfig.model_validate(config_data)
    trainer = Trainer(
        config,
        [_sequence("wave", [float(value) for value in range(20)])],
        device="cpu",
    )
    initial_parameters = {
        name: parameter.detach().clone()
        for name, parameter in trainer.model.named_parameters()
    }
    step = Mock(wraps=trainer.optimizer.step)
    monkeypatch.setattr(trainer.optimizer, "step", step)

    result = trainer.train()

    assert step.call_count == config.training.epochs * len(trainer.data_loaders.train)
    assert any(
        not torch.equal(initial_parameters[name], parameter)
        for name, parameter in result.model.named_parameters()
    )
    assert [entry.epoch for entry in result.history] == [1, 2]
    assert all(entry.training >= 0.0 for entry in result.history)
    assert all(entry.validation is not None for entry in result.history)
    assert result.model.training is False


def test_training_is_repeatable_for_the_configured_seed() -> None:
    """Reproduce model parameters and history from the same CPU seed."""

    config_data = _config_data()
    config_data["training"]["epochs"] = 2
    config_data["training"]["sequence"] = {
        "length": 2,
        "stride": 2,
        "hidden_state": "reset",
    }
    config = TrainingConfig.model_validate(config_data)
    dataset = [_sequence("wave", [float(value) for value in range(20)])]

    first = Trainer(config, dataset, device="cpu").train()
    second = Trainer(config, dataset, device="cpu").train()

    assert first.history == second.history
    assert all(
        torch.equal(first.model.state_dict()[name], second.model.state_dict()[name])
        for name in first.model.state_dict()
    )


def test_config_rejects_unsafe_hidden_state_carry() -> None:
    """Reject hidden-state behavior the windowed loader cannot preserve."""

    config_data = _config_data()
    config_data["training"]["sequence"]["hidden_state"] = "carry"

    with pytest.raises(ValueError, match="hidden_state"):
        TrainingConfig.model_validate(config_data)


def test_trainer_rejects_training_without_a_complete_window() -> None:
    """Fail clearly when sequence settings produce no trainable batch."""

    config = TrainingConfig.model_validate(_config_data())

    with pytest.raises(ValueError, match="complete training sequence window"):
        Trainer(config, [_sequence("wave", [0.0, 1.0])], device="cpu")


def test_trainer_saves_interval_checkpoints(tmp_path: Path) -> None:
    """Save complete checkpoints only at configured epoch intervals."""

    config_data = _config_data()
    config_data["training"]["epochs"] = 5
    config_data["training"]["checkpoint"] = {"every_n_epochs": 2}
    config_data["training"]["sequence"] = {
        "length": 2,
        "stride": 2,
        "hidden_state": "reset",
    }
    config = TrainingConfig.model_validate(config_data)
    checkpoint_directory = tmp_path / "checkpoints"
    trainer = Trainer(
        config,
        [_sequence("wave", [float(value) for value in range(20)])],
        device="cpu",
        checkpoint_directory=checkpoint_directory,
    )

    trainer.train()

    assert sorted(path.name for path in checkpoint_directory.iterdir()) == [
        "epoch-000002.pt",
        "epoch-000004.pt",
    ]
    checkpoint = torch.load(
        checkpoint_directory / "epoch-000004.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert checkpoint["epoch"] == 4
    assert [entry["epoch"] for entry in checkpoint["history"]] == [1, 2, 3, 4]
    assert checkpoint["config"]["training"]["checkpoint"] == {"every_n_epochs": 2}
    assert checkpoint["model_state_dict"].keys() == trainer.model.state_dict().keys()
    assert checkpoint["optimizer_state_dict"]["state"]
    assert not list(checkpoint_directory.glob("*.tmp"))


def test_trainer_requires_directory_when_checkpointing() -> None:
    """Reject checkpointing without an artifact destination."""

    config_data = _config_data()
    config_data["training"]["checkpoint"] = {"every_n_epochs": 1}
    config = TrainingConfig.model_validate(config_data)

    with pytest.raises(ValueError, match="checkpoint directory"):
        Trainer(
            config,
            [_sequence("wave", [float(value) for value in range(20)])],
            device="cpu",
        )
