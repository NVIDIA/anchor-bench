
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from anchor_bench.core.schemas import ActuatorFile, ResidualSpec


def load_actuator_file(path: str | Path) -> ActuatorFile:
    source = Path(path)
    data = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    return ActuatorFile.from_dict(data, source)


def _residual_to_dict(residual: ResidualSpec | None) -> dict[str, Any] | None:
    if residual is None:
        return None
    return {
        "type": residual.type,
        "network_file": residual.network_file,
        "stats_file": residual.stats_file,
        "input_columns": residual.input_columns,
        "output_column": residual.output_column,
        "source": residual.source,
        "parameters": residual.parameters,
    }


def build_newton_component_dicts(spec: ActuatorFile) -> list[dict[str, Any]]:
    """Return Newton 1.2 composable actuator component descriptions.

    Runtime construction is intentionally separated from schema validation so
    host-only tests do not import Isaac Sim or Newton.
    """
    components = []
    for actuator in spec.actuators:
        components.append(
            {
                "name": actuator.name,
                "joints": actuator.joints,
                "controller": {
                    "type": actuator.controller.type,
                    "parameters": actuator.controller.parameters,
                    "network_file": actuator.controller.network_file,
                    "source": actuator.controller.source,
                },
                "delay": {"duration_ms": actuator.delay.duration_ms},
                "clamps": [
                    {"type": clamp.type, "parameters": clamp.parameters}
                    for clamp in actuator.clamps
                ],
                "physical_parameters": actuator.physical_parameters,
                "residual": _residual_to_dict(actuator.residual),
            }
        )
    return components


def instantiate_newton_actuators(spec: ActuatorFile):
    """Instantiate Newton actuators lazily in an Isaac Sim runtime."""
    try:
        import newton.actuators as newton_actuators  # noqa: F401
    except Exception as exc:  # pragma: no cover - runtime-only guard
        raise RuntimeError(
            "Newton 1.2 actuator construction requires a runtime with newton.actuators available."
        ) from exc
    return build_newton_component_dicts(spec)
