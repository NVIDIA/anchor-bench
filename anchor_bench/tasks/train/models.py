"""Network implementations supported by the training task."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from anchor_bench.schema.training import GruModelConfig


class GruModel(nn.Module):
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

    def forward(self, inputs: Tensor) -> Tensor:
        """Return a prediction for every input timestep."""

        recurrent, _ = self.recurrent(inputs)
        return self.activation(self.output(recurrent)) * self.output_scale


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
