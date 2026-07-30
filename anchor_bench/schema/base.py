"""Shared Pydantic configuration base classes."""

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
