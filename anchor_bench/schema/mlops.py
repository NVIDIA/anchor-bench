# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration shared by MLOps tracking backends."""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import Field

from anchor_bench.schema.base import StrictBaseModel


class MLOpsConfig(StrictBaseModel):
    """Select a backend without coupling task schemas to its SDK."""

    backend: str = Field(
        default="mlflow",
        min_length=1,
        description="MLOps backend name available to the task runner.",
    )
    experiment_name: str = Field(
        default="anchor-bench",
        min_length=1,
        description="Backend experiment or project that owns the run.",
    )
    run_name: Optional[str] = Field(
        default=None,
        min_length=1,
        description="Optional human-readable run name.",
    )
    upload_dataset: bool = Field(
        default=False,
        description=(
            "Upload source dataset bytes in addition to recording dataset lineage."
        ),
    )
    options: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Backend constructor options persisted and logged with the run "
            "configuration."
        ),
    )
