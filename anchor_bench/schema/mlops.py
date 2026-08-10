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
        description="Registered backend or entry-point name.",
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
