"""Shared Pydantic primitives for external YAML configuration models."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, ValidationError


def _location(context: str, parts: tuple[Any, ...]) -> str:
    """Render a Pydantic error location relative to its source context."""

    result = context
    for part in parts:
        if isinstance(part, int):
            result += f"[{part}]"
        else:
            result += f".{part}"
    return result


def _format_validation_error(error: ValidationError, context: str) -> str:
    """Convert one Pydantic error group into a concise source-aware message."""

    errors = error.errors(include_url=False)
    extras = [item for item in errors if item["type"] == "extra_forbidden"]
    if extras and len(extras) == len(errors):
        parent = tuple(extras[0]["loc"][:-1])
        if all(tuple(item["loc"][:-1]) == parent for item in extras):
            keys = sorted(str(item["loc"][-1]) for item in extras)
            return f"{_location(context, parent)} has unknown keys: {', '.join(keys)}"

    item = errors[0]
    location = tuple(item["loc"])
    error_type = item["type"]
    if error_type == "missing":
        parent, key = location[:-1], location[-1]
        return f"{_location(context, parent)} missing required key '{key}'"
    if error_type in {"dict_type", "model_type"}:
        return f"{_location(context, location)} must be a mapping"
    if error_type == "bool_type":
        return f"{_location(context, location)} must be a boolean"
    if error_type == "greater_than_equal":
        return f"{_location(context, location)} must be >= {item.get('ctx', {}).get('ge')}"
    if error_type == "greater_than":
        return f"{_location(context, location)} must be > {item.get('ctx', {}).get('gt')}"
    if error_type == "value_error":
        cause = item.get("ctx", {}).get("error")
        message = str(cause) if cause is not None else item["msg"].removeprefix("Value error, ")
        return f"{_location(context, location)} {message}"
    return f"{_location(context, location)}: {item['msg']}"


class ConfigModel(BaseModel):
    """Immutable, unknown-key rejecting model used at file/input boundaries."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None, context: str | Path) -> Self:
        """Validate one external mapping with source-aware error messages."""

        text_context = str(context)
        if data is None:
            data = {}
        if not isinstance(data, Mapping):
            raise ValueError(f"{text_context} must be a mapping")
        try:
            return cls.model_validate(dict(data))
        except ValidationError as exc:
            raise ValueError(_format_validation_error(exc, text_context)) from exc

    def to_dict(self) -> dict[str, Any]:
        """Return the validated model as a plain mapping without null fields."""

        return self.model_dump(exclude_none=True)
