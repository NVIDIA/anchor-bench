"""Helpers for introspecting validated configuration models."""

from __future__ import annotations

from typing import Type, get_args

from pydantic import BaseModel


def _config_type(config_class: Type[BaseModel]) -> str:
    """Return the single string value declared by a config's ``type`` Literal."""

    annotation = config_class.model_fields["type"].annotation
    values = get_args(annotation)
    if len(values) != 1 or not isinstance(values[0], str):
        raise TypeError(f"{config_class.__name__}.type must be a single string Literal")
    return values[0]
