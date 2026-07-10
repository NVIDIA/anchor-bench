"""Shared Pydantic primitives for external YAML configuration models."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, ValidationError


RUNTIME_GRU_INPUT_COLUMNS = frozenset({"position", "position_error", "velocity", "solver_pd"})
GRU_INPUT_COLUMNS = RUNTIME_GRU_INPUT_COLUMNS | {"dynamic_bias", "previous_torque"}


def parse_gru_columns(value: Any, *, allowed: frozenset[str] = GRU_INPUT_COLUMNS) -> list[str]:
    """Parse an ordered GRU feature list against the requested feature set."""

    if isinstance(value, str):
        columns = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, (list, tuple)):
        columns = [str(item).strip() for item in value if str(item).strip()]
    else:
        raise ValueError("must be a string or list of strings")
    if not columns:
        raise ValueError("must contain at least one column")
    duplicates = sorted({column for column in columns if columns.count(column) > 1})
    if duplicates:
        raise ValueError(f"must not contain duplicate columns: {', '.join(duplicates)}")
    unsupported = sorted(set(columns) - allowed)
    if unsupported:
        raise ValueError(f"unsupported GRU input columns: {', '.join(unsupported)}")
    return columns


def parse_joint_names(value: Any) -> list[str]:
    """Parse a non-empty ordered list of joint names."""

    if isinstance(value, str):
        names = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, (list, tuple)):
        names = [str(item).strip() for item in value if str(item).strip()]
    else:
        raise ValueError("must be a string or list of strings")
    if not names:
        raise ValueError("must contain at least one joint")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"must not contain duplicate joints: {', '.join(duplicates)}")
    return names


def _location(context: str, parts: tuple[Any, ...]) -> str:
    result = context
    for part in parts:
        if isinstance(part, int):
            result += f"[{part}]"
        else:
            result += f".{part}"
    return result


def _format_validation_error(error: ValidationError, context: str) -> str:
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
        if message.startswith("unsupported GRU input columns:"):
            return message
        return f"{_location(context, location)} {message}"
    return f"{_location(context, location)}: {item['msg']}"


class ConfigModel(BaseModel):
    """Immutable, unknown-key rejecting model used at file/input boundaries."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None, context: str | Path) -> Self:
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
        return self.model_dump(exclude_none=True)
