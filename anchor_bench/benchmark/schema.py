"""Strict, host-safe schemas for benchmark profiles and candidate identity."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import PurePosixPath
from typing import Any, Literal, Self

from pydantic import Field, FiniteFloat, StrictBool, field_validator, model_validator

from anchor_bench.core.config_model import ConfigModel
from anchor_bench.core.hashing import require_relative_posix_path, validate_sha256
from anchor_bench.core.profiles import require_logical_name


BenchmarkSignal = Literal["position", "velocity"]
MotionSelection = Literal["*"] | tuple[str, ...]


def _require_text(value: str, *, label: str) -> str:
    """Return text after enforcing a non-empty, whitespace-stable value."""

    if not value or value != value.strip():
        raise ValueError(f"{label} must be non-empty and have no surrounding whitespace")
    return value


def _ordered_names(value: Any, *, label: str) -> tuple[str, ...]:
    """Parse an ordered, non-empty sequence of unique names."""

    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be an ordered list")
    names = tuple(value)
    if not names:
        raise ValueError(f"{label} must contain at least one value")
    if any(not isinstance(name, str) or not name or name != name.strip() for name in names):
        raise ValueError(f"{label} must contain non-empty names with no surrounding whitespace")
    duplicates = sorted({name for index, name in enumerate(names) if name in names[:index]})
    if duplicates:
        raise ValueError(f"duplicate {label}: {', '.join(duplicates)}")
    return names


class BenchmarkSimulationConfig(ConfigModel):
    """Complete Newton simulation settings used by one benchmark recipe."""

    physics_freq: int = Field(ge=1)
    control_freq: int = Field(ge=1)
    render_freq: int = Field(ge=1)
    integrator: str
    solver_iterations: int = Field(ge=1)
    line_search_iterations: int = Field(ge=1)

    @field_validator("integrator")
    @classmethod
    def _integrator(cls, value: str) -> str:
        """Validate the configured integrator name."""

        return _require_text(value, label="integrator")


class BenchmarkProfile(ConfigModel):
    """Human-authored held-out replay recipe, independent of its candidate."""

    schema_version: Literal[1]
    data: str
    motions: MotionSelection
    simulation: BenchmarkSimulationConfig
    num_envs: int = Field(ge=1)
    buffer_time: FiniteFloat = Field(ge=0.0)
    sync_initial_state: StrictBool
    signals: tuple[BenchmarkSignal, ...] = Field(min_length=1)
    output_root: str

    @field_validator("data", "output_root")
    @classmethod
    def _relative_paths(cls, value: str, info: Any) -> str:
        """Normalize benchmark data and output paths as relative POSIX paths."""

        _require_text(value, label=info.field_name)
        return require_relative_posix_path(value)

    @field_validator("motions", mode="before")
    @classmethod
    def _motions(cls, value: Any) -> MotionSelection:
        """Accept either every motion or an ordered set of named motions."""

        if value == "*":
            return "*"
        return _ordered_names(value, label="motions")

    @field_validator("signals", mode="before")
    @classmethod
    def _signals(cls, value: Any) -> tuple[str, ...]:
        """Validate the ordered state signals supported by benchmark replay."""

        signals = _ordered_names(value, label="signals")
        if "torque" in signals:
            raise ValueError(
                "torque benchmark signals are not supported until comparable torque semantics exist"
            )
        unsupported = sorted(set(signals) - {"position", "velocity"})
        if unsupported:
            raise ValueError(f"unsupported benchmark signal(s): {', '.join(unsupported)}")
        return signals


class BenchmarkCandidateReference(ConfigModel):
    """Resolved identity of one immutable SysID or training run directory."""

    phase: Literal["sysid", "train"]
    robot: str
    run: str
    run_input_sha256: str
    run_manifest_sha256: str

    @field_validator("robot")
    @classmethod
    def _robot(cls, value: str) -> str:
        """Validate the candidate robot's logical name."""

        return require_logical_name(value, label="robot name")

    @field_validator("run")
    @classmethod
    def _run(cls, value: str) -> str:
        """Normalize the candidate run reference as a relative POSIX path."""

        return require_relative_posix_path(value)

    @field_validator("run_input_sha256", "run_manifest_sha256")
    @classmethod
    def _hash(cls, value: str, info: Any) -> str:
        """Validate candidate identity fields as SHA-256 digests."""

        return validate_sha256(value, label=info.field_name)

    @model_validator(mode="after")
    def _immutable_run_layout(self) -> Self:
        """Require the candidate path to encode its phase, robot, seed, and digest."""

        parts = PurePosixPath(self.run).parts
        expected_prefix = ("outputs", self.phase, self.robot)
        if len(parts) != 6 or parts[:3] != expected_prefix:
            raise ValueError(
                "run must be an immutable directory under "
                f"outputs/{self.phase}/{self.robot}/<profile>/seed-<seed>/<run-input-sha256>"
            )
        require_logical_name(parts[3], label="source profile name")
        seed_text = parts[4].removeprefix("seed-")
        try:
            seed = int(seed_text)
        except ValueError as error:
            raise ValueError("run must contain a canonical seed-<integer> directory") from error
        if parts[4] != f"seed-{seed}":
            raise ValueError("run must contain a canonical seed-<integer> directory")
        if parts[-1] != self.run_input_sha256:
            raise ValueError("run directory digest must equal run_input_sha256")
        return self


__all__ = [
    "BenchmarkCandidateReference",
    "BenchmarkProfile",
    "BenchmarkSignal",
    "BenchmarkSimulationConfig",
    "MotionSelection",
]
