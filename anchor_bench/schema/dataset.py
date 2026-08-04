"""Shared dataset configuration models."""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import Field, StrictFloat, StrictInt
from typing_extensions import TypeAlias

from anchor_bench.schema.base import StrictBaseModel

JsonNumber: TypeAlias = Union[StrictInt, StrictFloat]
NonEmptyString: TypeAlias = Annotated[str, Field(min_length=1)]
PositiveNumber: TypeAlias = Annotated[JsonNumber, Field(gt=0.0)]


class ResamplingConfig(StrictBaseModel):
    """Configure uniform dataset resampling with ``dt`` in seconds."""

    type: Literal["uniform"]
    dt: PositiveNumber
    time_range: Literal["intersection"] = "intersection"
    interpolation: Literal["linear"] = "linear"


class ParquetDatasetConfig(StrictBaseModel):
    """Locate and resample one long-form Parquet time-series dataset."""

    type: Literal["parquet"]
    source: NonEmptyString
    resampling: ResamplingConfig
