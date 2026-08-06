"""Network implementations supported by the training task."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import torch
from torch import Tensor, nn

from anchor_bench.schema.training import GruModelConfig

StateT = TypeVar("StateT")


@dataclass(frozen=True)
class ModelOutput(Generic[StateT]):
    """Return predictions and model-specific state from one forward pass."""

    predictions: Tensor
    state: StateT


class BaseModel(nn.Module, ABC, Generic[StateT]):
    """Common prediction and model-state interface for trainable networks."""

    @abstractmethod
    def forward(
        self,
        inputs: Tensor,
        state: StateT | None = None,
    ) -> ModelOutput[StateT]:
        """Return predictions and state for a model-specific forward pass."""
        raise NotImplementedError


class GruModel(BaseModel[Tensor]):
    """Predict every timestep with a GRU and linear output layer."""

    output_scale: Tensor

    def __init__(
        self,
        config: GruModelConfig,
        *,
        input_size: int,
        output_size: int,
    ) -> None:
        """Build a GRU for the prepared input and target widths."""

        super().__init__()
        dropout = float(config.dropout) if config.num_layers > 1 else 0.0
        self.recurrent = nn.GRU(
            input_size=input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.output = nn.Linear(config.hidden_size, output_size)
        self.activation = _activation(config.output.activation)
        self.register_buffer(
            "output_scale",
            torch.tensor(float(config.output.scale)),
        )

    def forward(
        self,
        inputs: Tensor,
        state: Tensor | None = None,
    ) -> ModelOutput[Tensor]:
        """Return timestep predictions and the next recurrent state."""

        recurrent, next_state = self.recurrent(inputs, state)
        predictions = self.activation(self.output(recurrent)) * self.output_scale
        return ModelOutput(predictions=predictions, state=next_state)


def _activation(name: str) -> nn.Module:
    """Create one configured output activation."""

    activations: dict[str, type[nn.Module]] = {
        "identity": nn.Identity,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
    }
    try:
        return activations[name]()
    except KeyError as error:
        raise ValueError(f"Unsupported output activation {name!r}.") from error
