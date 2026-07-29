"""Load YAML task configurations into registered Pydantic models."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Type, Union

import yaml

from anchor_bench.errors import ConfigParseError, UnknownConfigTypeError
from anchor_bench.schema.actuator_sys_id import ActuatorSysIdConfig
from anchor_bench.schema.base import BaseConfig

DATA_MODELS: Dict[str, Type[BaseConfig]] = {
    "actuator_sys_id": ActuatorSysIdConfig,
}


def parse(config_file: Union[str, Path]) -> BaseConfig:
    """Parse and validate one YAML task configuration file."""

    config_path = Path(config_file)

    try:
        raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ConfigParseError(
            f"Config file does not exist: {config_path}."
        ) from error
    except OSError as error:
        raise ConfigParseError(
            f"Could not read config file {config_path}: {error}."
        ) from error
    except yaml.YAMLError as error:
        raise ConfigParseError(
            f"Invalid YAML in config file {config_path}: {error}."
        ) from error

    if not isinstance(raw_config, dict):
        raise ConfigParseError(
            f"Config at {config_path} must contain a YAML object at its root."
        )

    config_type = raw_config.get("type")
    if not isinstance(config_type, str) or not config_type:
        raise UnknownConfigTypeError(
            f"Config at {config_path} must define a non-empty string field 'type'."
        )

    data_model = DATA_MODELS.get(config_type)
    if data_model is None:
        known_types = ", ".join(sorted(DATA_MODELS)) or "none"
        raise UnknownConfigTypeError(
            f"No parser is registered for config type {config_type!r} at "
            f"{config_path}. Registered config types: {known_types}."
        )

    return data_model.model_validate(raw_config)
