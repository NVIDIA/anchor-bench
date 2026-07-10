from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anchor_bench.core.paths import repo_root
from anchor_bench.core.run_schema import SimulationDefaults


@dataclass(frozen=True)
class SimulationTiming:
    physics_freq: int
    control_freq: int
    render_freq: int

    @property
    def physics_dt(self) -> float:
        return 1.0 / self.physics_freq

    @property
    def control_dt(self) -> float:
        return 1.0 / self.control_freq

    @property
    def control_divisor(self) -> int:
        return self.physics_freq // self.control_freq

    @property
    def render_interval(self) -> int:
        return self.physics_freq // self.render_freq


def _positive_int(name: str, value: Any, context: str) -> int:
    resolved = int(value)
    if resolved < 1:
        raise ValueError(f"{context}.{name} must be >= 1")
    return resolved


def resolve_simulation_timing(sim_cfg: SimulationDefaults | None, context: str) -> SimulationTiming:
    cfg = sim_cfg or SimulationDefaults()
    physics_freq = _positive_int("physics_freq", cfg.physics_freq or 500, context)
    control_freq = _positive_int("control_freq", cfg.control_freq or physics_freq, context)
    render_freq = _positive_int("render_freq", cfg.render_freq or control_freq, context)
    if physics_freq % control_freq != 0:
        raise ValueError(f"{context} requires physics_freq divisible by control_freq")
    if physics_freq % render_freq != 0:
        raise ValueError(f"{context} requires physics_freq divisible by render_freq")
    return SimulationTiming(
        physics_freq=physics_freq,
        control_freq=control_freq,
        render_freq=render_freq,
    )


def resolve_value_with_source(
    explicit: Any,
    configured: Any,
    fallback: Any = None,
    *,
    explicit_label: str = "cli",
    configured_label: str = "run.yaml",
    fallback_label: str = "fallback",
) -> tuple[Any, str]:
    if explicit is not None:
        return explicit, explicit_label
    if configured is not None:
        return configured, configured_label
    return fallback, fallback_label


def config_source(
    explicit: Any,
    configured: Any,
    *,
    explicit_label: str = "cli",
    configured_label: str = "run.yaml",
    fallback_label: str = "fallback",
) -> str:
    return resolve_value_with_source(
        explicit,
        configured,
        explicit_label=explicit_label,
        configured_label=configured_label,
        fallback_label=fallback_label,
    )[1]


def defaulted_value(explicit: Any, default: Any, fallback: Any = None) -> Any:
    return explicit if explicit is not None else default if default is not None else fallback


def resolve_required_text(explicit: Any, default: Any, fallback: Any, label: str) -> str:
    value = defaulted_value(explicit, default, fallback)
    if value is None or str(value) == "":
        raise ValueError(f"{label} is required; pass it explicitly or set it in run.yaml")
    return str(value)


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return repo_root() / path


def resolve_repo_path_text(value: str | Path) -> str:
    return str(resolve_repo_path(value))
