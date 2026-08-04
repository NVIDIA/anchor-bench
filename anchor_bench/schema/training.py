"""Pydantic models for general network-training configurations."""

from __future__ import annotations

from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import (
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)
from typing_extensions import TypeAlias

from anchor_bench.schema.base import BaseConfig, StrictBaseModel
from anchor_bench.schema.dataset import ParquetDatasetConfig

JsonNumber: TypeAlias = Union[StrictInt, StrictFloat]
CategoricalValue: TypeAlias = Union[
    StrictBool,
    StrictInt,
    StrictFloat,
    StrictStr,
]
NonEmptyString: TypeAlias = Annotated[str, Field(min_length=1)]
PositiveInteger: TypeAlias = Annotated[StrictInt, Field(ge=1)]
NonNegativeInteger: TypeAlias = Annotated[StrictInt, Field(ge=0)]
PositiveNumber: TypeAlias = Annotated[JsonNumber, Field(gt=0.0)]
NonNegativeNumber: TypeAlias = Annotated[JsonNumber, Field(ge=0.0)]
SearchParameterName: TypeAlias = Annotated[
    str,
    Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)+$"),
]


def _duplicate_values(values: List[CategoricalValue]) -> bool:
    """Return whether JSON-compatible scalar values contain a duplicate."""

    seen = set()
    for value in values:
        if isinstance(value, bool):
            key = ("boolean", value)
        elif isinstance(value, (int, float)):
            key = ("number", value)
        else:
            key = ("string", value)

        if key in seen:
            return True
        seen.add(key)

    return False


class SimulationConfig(StrictBaseModel):
    backend: Literal["newton"] = "newton"
    usd_path: NonEmptyString


class NetworkMappingConfig(StrictBaseModel):
    inputs: Annotated[List[NonEmptyString], Field(min_length=1)]
    outputs: Annotated[List[NonEmptyString], Field(min_length=1)]

    @field_validator("inputs", "outputs")
    @classmethod
    def joint_names_must_be_unique(cls, names: List[str]) -> List[str]:
        if len(names) != len(set(names)):
            raise ValueError("joint names must be unique")
        return names


class ModelOutputConfig(StrictBaseModel):
    activation: Literal["identity", "tanh", "sigmoid", "relu", "gelu", "silu"]
    scale: PositiveNumber


class GruModelConfig(StrictBaseModel):
    """Architecture-specific configuration for a GRU network."""

    type: Literal["gru"]
    hidden_size: PositiveInteger
    num_layers: PositiveInteger
    dropout: Annotated[JsonNumber, Field(ge=0.0, le=1.0)]
    output: ModelOutputConfig


class SequenceConfig(StrictBaseModel):
    length: PositiveInteger
    stride: PositiveInteger
    hidden_state: Literal["reset", "carry"]


class AdamwOptimizerConfig(StrictBaseModel):
    type: Literal["adamw"]
    learning_rate: PositiveNumber
    weight_decay: NonNegativeNumber


class MseLossConfig(StrictBaseModel):
    type: Literal["mse"]


class MaeLossConfig(StrictBaseModel):
    type: Literal["mae"]


class HuberLossConfig(StrictBaseModel):
    type: Literal["huber"]
    delta: PositiveNumber


LossConfig: TypeAlias = Annotated[
    Union[MseLossConfig, MaeLossConfig, HuberLossConfig],
    Field(discriminator="type"),
]


class GradientConfig(StrictBaseModel):
    clip_norm: PositiveNumber


class AugmentationConfig(StrictBaseModel):
    input_noise_std: NonNegativeNumber


class TrainingParametersConfig(StrictBaseModel):
    """Settings shared by all supported network architectures."""

    epochs: PositiveInteger
    batch_size: PositiveInteger
    seed: NonNegativeInteger
    validation_ratio: Annotated[JsonNumber, Field(ge=0.0, lt=1.0)]
    sequence: SequenceConfig
    optimizer: AdamwOptimizerConfig
    loss: LossConfig
    gradient: GradientConfig
    augmentation: AugmentationConfig


class SearchObjectiveConfig(StrictBaseModel):
    metric: NonEmptyString
    direction: Literal["minimize", "maximize"]


class CategoricalSearchParameter(StrictBaseModel):
    data_type: Literal["categorical"]
    values: Annotated[List[CategoricalValue], Field(min_length=1)]

    @field_validator("values")
    @classmethod
    def values_must_be_unique(
        cls, values: List[CategoricalValue]
    ) -> List[CategoricalValue]:
        if _duplicate_values(values):
            raise ValueError("categorical search values must be unique")
        return values


class IntegerSearchParameter(StrictBaseModel):
    data_type: Literal["integer"]
    min: StrictInt
    max: StrictInt
    step: Optional[Annotated[StrictInt, Field(ge=1)]] = None
    scale: Literal["linear", "log"]

    @model_validator(mode="after")
    def reject_explicit_null_step(self) -> IntegerSearchParameter:
        if "step" in self.model_fields_set and self.step is None:
            raise ValueError("step must be an integer when provided")
        return self

    @model_validator(mode="after")
    def validate_range(self) -> IntegerSearchParameter:
        if self.min > self.max:
            raise ValueError("min must be less than or equal to max")
        if self.scale == "log" and self.min < 1:
            raise ValueError("min must be at least 1 when scale is 'log'")
        if self.scale == "log" and self.step not in (None, 1):
            raise ValueError("step must be 1 when scale is 'log'")
        return self


class FloatSearchParameter(StrictBaseModel):
    data_type: Literal["float"]
    min: JsonNumber
    max: JsonNumber
    step: Optional[PositiveNumber] = None
    scale: Literal["linear", "log"]

    @model_validator(mode="after")
    def reject_explicit_null_step(self) -> FloatSearchParameter:
        if "step" in self.model_fields_set and self.step is None:
            raise ValueError("step must be a number when provided")
        return self

    @model_validator(mode="after")
    def validate_range(self) -> FloatSearchParameter:
        if self.min > self.max:
            raise ValueError("min must be less than or equal to max")
        if self.scale == "log" and self.min <= 0:
            raise ValueError("min must be greater than 0 when scale is 'log'")
        if self.scale == "log" and self.step is not None:
            raise ValueError("step must be omitted when scale is 'log'")
        return self


SearchParameter: TypeAlias = Annotated[
    Union[
        CategoricalSearchParameter,
        IntegerSearchParameter,
        FloatSearchParameter,
    ],
    Field(discriminator="data_type"),
]


class SearchConfig(StrictBaseModel):
    schema_version: Literal[1]
    method: Literal["optuna"]
    trials: PositiveInteger
    objective: SearchObjectiveConfig
    search_space: Annotated[
        Dict[SearchParameterName, SearchParameter],
        Field(min_length=1),
    ]


class TrainingValidatorConfig(StrictBaseModel):
    name: NonEmptyString
    type: Literal["offline", "simulation"]
    entrypoint: NonEmptyString
    config: Dict[str, Any] = Field(default_factory=dict)


class TrainingValidationConfig(StrictBaseModel):
    validators: Annotated[List[TrainingValidatorConfig], Field(min_length=1)]


class TrainingConfig(BaseConfig):
    """Top-level configuration for the architecture-agnostic training pipeline."""

    type: Literal["training"]  # pyright: ignore[reportIncompatibleVariableOverride]
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    dataset: ParquetDatasetConfig
    simulation: Optional[SimulationConfig] = None
    input_features: Annotated[
        Dict[str, Optional[NonEmptyString]],
        Field(min_length=1),
    ]
    output_features: NonEmptyString
    mapping: Annotated[
        Dict[str, NetworkMappingConfig],
        Field(min_length=1),
    ]
    model: GruModelConfig
    training: TrainingParametersConfig
    search: Optional[SearchConfig] = None
    validation: TrainingValidationConfig

    @model_validator(mode="after")
    def reject_explicit_null_optional_fields(self) -> TrainingConfig:
        for field_name in ("description", "tags", "simulation", "search"):
            if (
                field_name in self.model_fields_set
                and getattr(self, field_name) is None
            ):
                raise ValueError(f"{field_name} cannot be null")
        return self

    @model_validator(mode="after")
    def require_simulation_for_preprocessors(self) -> TrainingConfig:
        if any(self.input_features.values()) and self.simulation is None:
            raise ValueError("simulation is required when preprocessors are configured")
        return self
