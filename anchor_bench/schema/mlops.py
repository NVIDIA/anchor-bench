"""Configuration shared by MLOps tracking backends."""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import Field

from anchor_bench.schema.base import StrictBaseModel


class MLOpsConfig(StrictBaseModel):
    """Select a backend without coupling task schemas to its SDK."""

    backend: str = Field(default="mlflow", min_length=1)
    experiment_name: str = Field(default="anchor-bench", min_length=1)
    run_name: Optional[str] = Field(default=None, min_length=1)
    options: Dict[str, Any] = Field(default_factory=dict)
