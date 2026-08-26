# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration models for local and managed USD assets."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Annotated, Literal, Union

from pydantic import Field, field_validator
from typing_extensions import TypeAlias

from anchor_bench.schema.base import StrictBaseModel

NonEmptyString: TypeAlias = Annotated[str, Field(min_length=1)]
GitCommit: TypeAlias = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]


def _validate_repository_path(value: str) -> str:
    """Require a normalized relative POSIX path within an asset repository."""

    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or "\\" in value
        or value != path.as_posix()
        or any(part in ("", ".", "..") for part in value.split("/"))
    ):
        raise ValueError("must be a normalized relative POSIX path")
    return value


class LocalUsdSource(StrictBaseModel):
    """Reference a USD file on the local filesystem."""

    type: Literal["local"]
    path: NonEmptyString


class NewtonAssetsUsdSource(StrictBaseModel):
    """Reference a revision-pinned file in the Newton asset repository."""

    type: Literal["newton-assets"]
    folder: NonEmptyString
    path: NonEmptyString
    ref: GitCommit

    @field_validator("folder", "path")
    @classmethod
    def paths_must_be_repository_relative(cls, value: str) -> str:
        return _validate_repository_path(value)


UsdSource: TypeAlias = Annotated[
    Union[LocalUsdSource, NewtonAssetsUsdSource],
    Field(discriminator="type"),
]
