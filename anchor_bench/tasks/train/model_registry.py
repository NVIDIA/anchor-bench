"""Registry for network models supported by the training task."""

from __future__ import annotations

from typing import Any, Callable

from anchor_bench.schema.training import ModelConfig
from anchor_bench.tasks.train.models import BaseModel, GruModel

MODEL_REGISTRY: dict[str, Callable[..., BaseModel[Any]]] = {
    "gru": GruModel,
}


def get_model(
    config: ModelConfig,
    *,
    input_size: int,
    output_size: int,
) -> BaseModel[Any]:
    """Instantiate the model selected by the training configuration."""

    try:
        model_type = MODEL_REGISTRY[config.type]
    except KeyError as error:
        raise ValueError(f"Unsupported model type {config.type!r}.") from error
    return model_type(
        config,
        input_size=input_size,
        output_size=output_size,
    )
