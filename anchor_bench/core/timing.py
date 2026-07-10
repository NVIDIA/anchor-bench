"""Simulation timing derived from complete authoritative phase profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class SimulationFrequencyConfig(Protocol):
    """Frequency fields shared by SysID, benchmark, and deploy profiles."""

    physics_freq: int
    control_freq: int
    render_freq: int


@dataclass(frozen=True)
class SimulationTiming:
    """Validated frequencies and their exact derived timing values."""

    physics_freq: int
    control_freq: int
    render_freq: int

    @property
    def physics_dt(self) -> float:
        """Return the physics-step duration in seconds."""

        return 1.0 / self.physics_freq

    @property
    def control_dt(self) -> float:
        """Return the control-step duration in seconds."""

        return 1.0 / self.control_freq

    @property
    def control_divisor(self) -> int:
        """Return physics steps per control step."""

        return self.physics_freq // self.control_freq

    @property
    def render_interval(self) -> int:
        """Return physics steps per rendered frame."""

        return self.physics_freq // self.render_freq


def _positive_frequency(name: str, value: int, context: str) -> int:
    """Return one positive integral frequency or raise a contextual error."""

    resolved = int(value)
    if resolved < 1:
        raise ValueError(f"{context}.{name} must be >= 1")
    return resolved


def resolve_simulation_timing(
    simulation: SimulationFrequencyConfig,
    context: str,
) -> SimulationTiming:
    """Validate exact phase frequencies and derive integer step ratios."""

    physics_freq = _positive_frequency(
        "physics_freq", simulation.physics_freq, context
    )
    control_freq = _positive_frequency(
        "control_freq", simulation.control_freq, context
    )
    render_freq = _positive_frequency(
        "render_freq", simulation.render_freq, context
    )
    if physics_freq % control_freq != 0:
        raise ValueError(f"{context} requires physics_freq divisible by control_freq")
    if physics_freq % render_freq != 0:
        raise ValueError(f"{context} requires physics_freq divisible by render_freq")
    return SimulationTiming(
        physics_freq=physics_freq,
        control_freq=control_freq,
        render_freq=render_freq,
    )


__all__ = [
    "SimulationFrequencyConfig",
    "SimulationTiming",
    "resolve_simulation_timing",
]
