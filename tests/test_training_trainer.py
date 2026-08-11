from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from anchor_bench.schema.training import TrainingConfig
from anchor_bench.tasks.train import (
    MODEL_REGISTRY,
    BaseModel,
    FeatureNormalization,
    GruModel,
    ModelOutput,
    NormalizationStatistics,
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

    output = model(torch.zeros(5, 7, 3))

    expected = torch.tanh(torch.tensor(1.0)).item() * 2.0
    assert isinstance(output, ModelOutput)
    assert output.predictions.shape == (5, 7, 2)
    assert output.state.shape == (2, 5, 4)
    assert torch.allclose(
        output.predictions,
        torch.full_like(output.predictions, expected),
    )
    next_output = model(torch.zeros(5, 1, 3), output.state)
    assert next_output.predictions.shape == (5, 1, 2)
    assert next_output.state.shape == output.state.shape


def test_normalized_model_keeps_inference_in_physical_units() -> None:
    """Normalize raw inputs and restore model outputs before returning them."""

    statistics = NormalizationStatistics(
        inputs=FeatureNormalization(
            method="standard",
            mean=(10.0,),
            standard_deviation=(2.0,),
        ),
        outputs=FeatureNormalization(
            method="standard",
            mean=(100.0,),
            standard_deviation=(5.0,),
        ),
    )

    class IdentityModel(nn.Module):
        """Return normalized inputs as normalized predictions."""

        def forward(
            self,
            inputs: torch.Tensor,
            hidden: torch.Tensor | None = None,
        ) -> ModelOutput[torch.Tensor]:
            return ModelOutput(predictions=inputs, state=torch.ones(1))

    result = statistics.wrap(IdentityModel())(torch.tensor([[[12.0]]]))

    assert result.predictions.item() == 105.0
    assert result.state.item() == 1.0


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
    """Run optimizer updates for each independent batch and collect losses."""

    config_data = _config_data()
    config_data["training"]["epochs"] = 2
    config_data["training"]["batch_size"] = 2
    config_data["training"]["sequence"] = {"length": 2}
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
    assert all(entry.validation is None for entry in result.history)
    assert result.model.training is False


def test_training_is_repeatable_for_the_configured_seed() -> None:
    """Reproduce model parameters and history from the same CPU seed."""

    config_data = _config_data()
    config_data["training"]["epochs"] = 2
    config_data["training"]["sequence"] = {"length": 2}
    config = TrainingConfig.model_validate(config_data)
    dataset = [_sequence("wave", [float(value) for value in range(20)])]

    first = Trainer(config, dataset, device="cpu").train()
    second = Trainer(config, dataset, device="cpu").train()

    assert first.history == second.history
    assert all(
        torch.equal(first.model.state_dict()[name], second.model.state_dict()[name])
        for name in first.model.state_dict()
    )


def test_trainer_resets_state_between_independent_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Start every fixed-length training window without recurrent state."""

    config_data = _config_data()
    config_data["training"]["batch_size"] = 2
    config_data["training"]["validation_ratio"] = 0.0
    config_data["training"]["sequence"] = {"length": 2}
    trainer = Trainer(
        TrainingConfig.model_validate(config_data),
        [
            _sequence("first", [0.0, 1.0, 2.0, 3.0, 4.0]),
            _sequence("second", [5.0, 6.0, 7.0, 8.0, 9.0]),
            _sequence("third", [10.0, 11.0, 12.0, 13.0, 14.0]),
        ],
        device="cpu",
    )
    original_forward = trainer._forward_with_state
    starts: list[bool] = []

    def record_state(
        inputs: torch.Tensor,
        *,
        state: torch.Tensor | None = None,
        previous_physical: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        starts.append(state is None)
        return original_forward(
            inputs,
            state=state,
            previous_physical=previous_physical,
        )

    monkeypatch.setattr(trainer, "_forward_with_state", record_state)

    trainer._train_epoch()

    assert starts == [True, True, True]


def test_trainer_rejects_a_sequence_shorter_than_the_window_length() -> None:
    """Reject training data that contains no complete sequence window."""

    config_data = _config_data()
    config_data["training"]["validation_ratio"] = 0.0
    with pytest.raises(
        ValueError,
        match="does not contain a complete training sequence window",
    ):
        Trainer(
            TrainingConfig.model_validate(config_data),
            [_sequence("wave", [0.0])],
            device="cpu",
        )


def test_trainer_saves_interval_checkpoints(tmp_path: Path) -> None:
    """Save complete checkpoints only at configured epoch intervals."""

    config_data = _config_data()
    config_data["training"]["epochs"] = 5
    config_data["training"]["checkpoint"] = {"every_n_epochs": 2}
    config_data["training"]["sequence"] = {"length": 2}
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
    assert checkpoint["normalization"] == {
        "inputs": {"method": "none", "mean": (0.0,), "standard_deviation": (1.0,)},
        "outputs": {"method": "none", "mean": (0.0,), "standard_deviation": (1.0,)},
    }
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


def test_trainer_feeds_prior_prediction_into_configured_input() -> None:
    """Use the configured initial value, then raw prior model predictions."""

    config_data = _config_data()
    config_data["input_features"]["feedback"] = None
    config_data["training"]["feedback"] = {
        "input": "feedback",
        "output": "target",
        "initial_value": 2.0,
    }
    config = TrainingConfig.model_validate(config_data)
    sequence = _sequence("wave", [0.0, 0.0, 0.0, 0.0])
    sequence.inputs["joint/feedback"] = 99.0
    trainer = Trainer(config, [sequence], device="cpu")

    class RecordingModel(BaseModel[torch.Tensor]):
        """Predict one more than the feedback input and record each step."""

        def __init__(self) -> None:
            super().__init__()
            self.seen: list[float] = []

        def forward(
            self,
            inputs: torch.Tensor,
            state: torch.Tensor | None = None,
        ) -> ModelOutput[torch.Tensor]:
            self.seen.append(float(inputs[0, 0, 1]))
            return ModelOutput(
                predictions=inputs[:, :, 1:2] + 1.0,
                state=torch.ones(1),
            )

    model = RecordingModel()
    trainer.model = model

    predictions, _, _ = trainer._forward_with_state(torch.zeros(1, 3, 2))

    assert model.seen == [2.0, 3.0, 4.0]
    assert predictions.flatten().tolist() == [3.0, 4.0, 5.0]
