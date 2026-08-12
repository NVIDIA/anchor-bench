"""Configuration shared by MLOps tracking backends."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import Field

from anchor_bench.schema.base import BaseConfig, StrictBaseModel


class MLOpsConfig(StrictBaseModel):
    """Select a backend without coupling task schemas to its SDK."""

    backend: str = Field(
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
            "Backend constructor options persisted with the run configuration; "
            "credentials must be configured outside the YAML."
        ),
    )


class MLOpsTaskConfig(BaseConfig):
    """Top-level config for tasks that support optional experiment tracking."""

    description: Optional[str] = None
    tags: Optional[List[str]] = None
    mlops: Optional[MLOpsConfig] = Field(
        default=None,
        description="Optional experiment-tracking configuration for this task.",
    )
