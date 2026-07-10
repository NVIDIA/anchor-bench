"""Strict, host-safe schemas for SysID profiles and parameter layouts."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import Field, FiniteFloat, field_validator, model_validator

from anchor_bench.core.config_model import ConfigModel


SUPPORTED_SYSID_PARAMETERS = (
    "stiffness",
    "damping",
    "armature",
    "dry_friction",
    "viscous_friction",
)

ParameterMode = Literal["per_joint", "shared"]


def _require_non_empty_text(value: str, *, name: str) -> str:
    """Return text after enforcing a non-empty, whitespace-stable value."""

    if not value or value != value.strip():
        raise ValueError(f"{name} must be non-empty and have no surrounding whitespace")
    return value


def _validate_joint_names(joints: Sequence[str], *, name: str = "joints") -> tuple[str, ...]:
    """Validate and freeze an ordered, non-empty sequence of unique joints."""

    result: list[str] = []
    for joint in joints:
        if not isinstance(joint, str) or not joint or joint != joint.strip():
            raise ValueError("joint names must be non-empty and have no surrounding whitespace")
        result.append(joint)
    duplicates = sorted({joint for joint in result if result.count(joint) > 1})
    if duplicates:
        raise ValueError(f"duplicate joint names in {name}: {', '.join(duplicates)}")
    if not result:
        raise ValueError(f"{name} must contain at least one joint")
    return tuple(result)


class ParameterBounds(ConfigModel):
    """Inclusive physical bounds for one optimizer variable."""

    lower: FiniteFloat = Field(ge=0.0)
    upper: FiniteFloat = Field(ge=0.0)

    @model_validator(mode="after")
    def _validate_order(self) -> "ParameterBounds":
        """Require the upper bound to be no smaller than the lower bound."""

        if self.upper < self.lower:
            raise ValueError(f"upper ({self.upper}) must be >= lower ({self.lower})")
        return self


class SysIdParameterSpec(ConfigModel):
    """Parameter bounds and explicit optimizer-variable tying policy."""

    mode: ParameterMode
    bounds: ParameterBounds


class SysIdSimulationConfig(ConfigModel):
    """Complete simulation settings used by one SysID recipe."""

    physics_freq: int = Field(ge=1)
    control_freq: int = Field(ge=1)
    render_freq: int = Field(ge=1)
    integrator: str
    solver_iterations: int = Field(ge=1)
    line_search_iterations: int = Field(ge=1)

    @field_validator("integrator")
    @classmethod
    def _validate_integrator(cls, value: str) -> str:
        """Validate the configured simulation integrator name."""

        return _require_non_empty_text(value, name="integrator")


class SysIdOptimizerConfig(ConfigModel):
    """CMA-ES settings used to optimize one SysID candidate."""

    type: Literal["cmaes"]
    sigma: FiniteFloat = Field(gt=0.0)
    max_iterations: int = Field(ge=1)
    epsilon: FiniteFloat = Field(ge=0.0)


class SysIdLossConfig(ConfigModel):
    """Non-negative state-signal weights for the SysID objective."""

    position: FiniteFloat = Field(ge=0.0)
    velocity: FiniteFloat = Field(ge=0.0)

    @model_validator(mode="after")
    def _require_signal(self) -> "SysIdLossConfig":
        """Require at least one state signal to contribute to the loss."""

        if self.position == 0.0 and self.velocity == 0.0:
            raise ValueError("at least one loss weight must be greater than zero")
        return self


class SysIdProfile(ConfigModel):
    """Human-authored SysID recipe, before typed CLI overrides are applied."""

    schema_version: Literal[1]
    data: str
    joints: list[str] = Field(min_length=1)
    simulation: SysIdSimulationConfig
    num_envs: int = Field(ge=1)
    buffer_time: FiniteFloat = Field(ge=0.0)
    seed: int
    optimizer: SysIdOptimizerConfig
    parameters: dict[str, SysIdParameterSpec] = Field(min_length=1)
    loss: SysIdLossConfig

    @field_validator("data")
    @classmethod
    def _validate_data(cls, value: str) -> str:
        """Validate the SysID dataset reference."""

        return _require_non_empty_text(value, name="data")

    @field_validator("joints")
    @classmethod
    def _validate_joints(cls, value: list[str]) -> list[str]:
        """Validate the profile's ordered target-joint list."""

        return list(_validate_joint_names(value))

    @field_validator("parameters", mode="before")
    @classmethod
    def _validate_parameters(cls, value: Any) -> Any:
        """Reject empty or unsupported SysID parameter specifications."""

        if not isinstance(value, dict):
            return value
        unsupported = sorted(set(value) - set(SUPPORTED_SYSID_PARAMETERS))
        if unsupported:
            raise ValueError(f"unsupported SysID parameter(s): {', '.join(unsupported)}")
        if not value:
            raise ValueError("must contain at least one SysID parameter")
        return value


@dataclass(frozen=True)
class ParameterLayoutEntry:
    """One optimizer variable and the canonical joint cells it controls."""

    parameter: str
    joints: tuple[str, ...]
    bounds: ParameterBounds

    @property
    def name(self) -> str:
        """Return the deterministic optimizer-variable name."""

        if len(self.joints) == 1:
            return f"{self.parameter}/{self.joints[0]}"
        return f"{self.parameter}/shared"


def parameter_layout(profile: SysIdProfile) -> tuple[ParameterLayoutEntry, ...]:
    """Return the deterministic optimizer layout for a resolved profile.

    ``per_joint`` contributes one variable per selected joint. ``shared``
    contributes one variable that expands to every selected joint. Parameter
    order is stable and independent of YAML mapping order.
    """

    entries: list[ParameterLayoutEntry] = []
    joints = tuple(profile.joints)
    for parameter in SUPPORTED_SYSID_PARAMETERS:
        spec = profile.parameters.get(parameter)
        if spec is None:
            continue
        if spec.mode == "shared":
            entries.append(ParameterLayoutEntry(parameter, joints, spec.bounds))
        else:
            entries.extend(
                ParameterLayoutEntry(parameter, (joint,), spec.bounds)
                for joint in joints
            )
    return tuple(entries)


def expand_parameter_values(
    profile: SysIdProfile,
    values: Sequence[float],
) -> dict[str, dict[str, float]]:
    """Expand optimizer variables into canonical per-joint fitted values."""

    layout = parameter_layout(profile)
    if len(values) != len(layout):
        raise ValueError(f"expected {len(layout)} optimizer values, got {len(values)}")

    expanded: dict[str, dict[str, float]] = {}
    for entry, raw_value in zip(layout, values, strict=True):
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"optimizer value for {entry.name} must be numeric") from exc
        if not math.isfinite(value):
            raise ValueError(f"optimizer value for {entry.name} must be finite")
        if value < entry.bounds.lower or value > entry.bounds.upper:
            raise ValueError(
                f"optimizer value for {entry.name} ({value}) is outside bounds "
                f"[{entry.bounds.lower}, {entry.bounds.upper}]"
            )
        parameter_values = expanded.setdefault(entry.parameter, {})
        for joint in entry.joints:
            parameter_values[joint] = value
    return expanded


def validate_profile_joints(
    profile: SysIdProfile,
    canonical_joint_names: Sequence[str],
) -> tuple[str, ...]:
    """Require the profile selection to contain canonical robot joint names."""

    canonical = _validate_joint_names(canonical_joint_names, name="canonical_joint_names")
    unknown = sorted(set(profile.joints) - set(canonical))
    if unknown:
        raise ValueError(f"non-canonical or unknown joints: {', '.join(unknown)}")
    return tuple(profile.joints)


# Compact aliases for consumers that do not need the SysID prefix at every use.
ParameterSpec = SysIdParameterSpec
SimulationConfig = SysIdSimulationConfig
OptimizerConfig = SysIdOptimizerConfig
LossConfig = SysIdLossConfig


__all__ = [
    "LossConfig",
    "OptimizerConfig",
    "ParameterBounds",
    "ParameterLayoutEntry",
    "ParameterMode",
    "ParameterSpec",
    "SUPPORTED_SYSID_PARAMETERS",
    "SimulationConfig",
    "SysIdLossConfig",
    "SysIdOptimizerConfig",
    "SysIdParameterSpec",
    "SysIdProfile",
    "SysIdSimulationConfig",
    "expand_parameter_values",
    "parameter_layout",
    "validate_profile_joints",
]
