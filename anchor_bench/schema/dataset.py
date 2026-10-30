# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared dataset configuration models."""

from __future__ import annotations

from typing import Annotated, Dict, List, Literal, Union

from pydantic import Field, StrictFloat, StrictInt, field_validator
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


class HuggingFaceDatasetSource(StrictBaseModel):
    """Select revision-pinned files from a Hugging Face dataset repository."""

    provider: Literal["huggingface"]
    repo_id: NonEmptyString
    revision: NonEmptyString
    paths: Annotated[List[NonEmptyString], Field(min_length=1)]

    @field_validator("paths")
    @classmethod
    def paths_must_be_relative_and_unique(cls, paths: List[str]) -> List[str]:
        """Reject duplicate, absolute, and parent-traversing repository paths."""

        if len(paths) != len(set(paths)):
            raise ValueError("Hugging Face dataset paths must be unique")
        invalid = [
            path
            for path in paths
            if path.startswith("/") or "\\" in path or ".." in path.split("/")
        ]
        if invalid:
            raise ValueError("Hugging Face dataset paths must be relative POSIX paths")
        return paths


DatasetSource: TypeAlias = Union[NonEmptyString, HuggingFaceDatasetSource]


class ParquetDatasetConfig(StrictBaseModel):
    """Locate and resample long-form Parquet time-series datasets."""

    type: Literal["parquet"]
    source: DatasetSource
    joint_name_mapping: Dict[NonEmptyString, NonEmptyString] = Field(
        default_factory=dict,
        description=(
            "Optional mapping from Parquet joint prefixes to the corresponding "
            "USD joint names. Unmapped joint names are left unchanged."
        ),
    )
    feature_name_mapping: Dict[NonEmptyString, NonEmptyString] = Field(
        default_factory=dict,
        description=(
            "Optional mapping from Parquet feature suffixes to canonical feature "
            "names. Unmapped feature names are left unchanged."
        ),
    )
    resampling: ResamplingConfig
