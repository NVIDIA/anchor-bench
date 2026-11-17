# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared Pydantic configuration base classes."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict


class StrictBaseModel(BaseModel):
    """Base model matching JSON Schema's strict object semantics."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        validate_default=True,
        allow_inf_nan=False,
    )


class BaseConfig(StrictBaseModel):
    """Common shape shared by all top-level task configurations."""

    type: str

    def resolve_paths(self, directory: Path) -> None:
        """Resolve task-specific paths relative to the configuration directory.

        Called after validation with the absolute parent directory of the YAML
        file. The default leaves paths unchanged; overrides may normalize path
        fields in memory but must not create or modify filesystem artifacts.
        """
