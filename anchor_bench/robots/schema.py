"""Strict robot identity and runtime-asset configuration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any, Literal, Self

from pydantic import Field, field_validator, model_validator

from anchor_bench.core.config_model import ConfigModel
from anchor_bench.core.hashing import require_relative_posix_path
from anchor_bench.core.profiles import require_logical_name


def _ordered_joint_names(value: Any) -> tuple[str, ...]:
    """Return a non-empty ordered tuple of unique physical joint names."""

    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError("joint_names must be an ordered list")
    names = tuple(value)
    if not names or any(
        not isinstance(name, str) or not name or name != name.strip()
        for name in names
    ):
        raise ValueError("joint_names must contain non-empty names without surrounding whitespace")
    duplicates = sorted(
        {name for index, name in enumerate(names) if name in names[:index]}
    )
    if duplicates:
        raise ValueError("duplicate joint_names: " + ", ".join(duplicates))
    return names


def _joint_aliases(value: Any) -> dict[str, str]:
    """Validate the raw alias mapping without resolving its targets yet."""

    if not isinstance(value, Mapping):
        raise ValueError("joint_aliases must be a mapping")
    aliases: dict[str, str] = {}
    for alias, target in value.items():
        if (
            not isinstance(alias, str)
            or not alias
            or alias != alias.strip()
            or not isinstance(target, str)
            or not target
            or target != target.strip()
        ):
            raise ValueError("joint_aliases must map non-empty names without surrounding whitespace")
        aliases[alias] = target
    return aliases


class RobotIdentity(ConfigModel):
    """Declarative robot identity; phase recipes and accepted state live elsewhere."""

    schema_version: Literal[1]
    name: str
    display_name: str
    fixed_base: bool = True
    asset_path: str
    runtime_usd: str
    joint_names: tuple[str, ...]
    joint_aliases: dict[str, str] = Field(default_factory=dict)
    ee_frame: str | None = None

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        """Require a filename-safe robot key."""

        return require_logical_name(value, label="robot name")

    @field_validator("display_name")
    @classmethod
    def _display_name(cls, value: str) -> str:
        """Require a readable non-empty display name."""

        if not value or value != value.strip():
            raise ValueError("display_name must be non-empty without surrounding whitespace")
        return value

    @field_validator("asset_path")
    @classmethod
    def _asset_path(cls, value: str) -> str:
        """Require a repository-relative asset-manifest path."""

        return require_relative_posix_path(value)

    @field_validator("runtime_usd")
    @classmethod
    def _runtime_usd(cls, value: str) -> str:
        """Require a relative USD filename below the robot asset root."""

        normalized = require_relative_posix_path(value)
        if PurePosixPath(normalized).suffix.lower() not in {".usd", ".usda", ".usdc"}:
            raise ValueError("runtime_usd must name a USD, USDA, or USDC asset")
        return normalized

    @field_validator("joint_names", mode="before")
    @classmethod
    def _joints(cls, value: Any) -> tuple[str, ...]:
        """Preserve canonical joint order while rejecting duplicates."""

        return _ordered_joint_names(value)

    @field_validator("joint_aliases", mode="before")
    @classmethod
    def _aliases(cls, value: Any) -> dict[str, str]:
        """Parse explicit alias-to-canonical mappings."""

        return _joint_aliases(value)

    @field_validator("ee_frame")
    @classmethod
    def _end_effector(cls, value: str | None) -> str | None:
        """Validate the optional end-effector frame label."""

        if value is not None and (not value or value != value.strip()):
            raise ValueError("ee_frame must be non-empty without surrounding whitespace")
        return value

    @model_validator(mode="after")
    def _identity_contract(self) -> Self:
        """Bind asset location and aliases to this robot's canonical identity."""

        expected_asset = f"assets/robots/{self.name}/manifest.yaml"
        if self.asset_path != expected_asset:
            raise ValueError(f"asset_path must be {expected_asset!r}")
        canonical = set(self.joint_names)
        colliding = sorted(set(self.joint_aliases).intersection(canonical))
        if colliding:
            raise ValueError("joint aliases collide with canonical names: " + ", ".join(colliding))
        unknown = sorted(set(self.joint_aliases.values()) - canonical)
        if unknown:
            raise ValueError("joint aliases target unknown canonical joints: " + ", ".join(unknown))
        return self


__all__ = ["RobotIdentity"]
