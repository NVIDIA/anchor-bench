from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from anchor_bench.actuators.loader import load_actuator_file
from anchor_bench.core.schemas import ActuatorFile, ActuatorSpec, SysIdBoundsFile
from anchor_bench.robots.registry import RobotBundle


def _single_value_clamp(actuator, clamp_type: str, key: str) -> float | None:
    for clamp in actuator.clamps:
        if clamp.type == clamp_type and key in clamp.parameters:
            return float(clamp.parameters[key])
    return None


def _extract_implicit_params_from_actuator(actuator: ActuatorSpec) -> dict[str, Any]:
    params: dict[str, Any] = {
        "stiffness": deepcopy(actuator.controller.parameters.get("stiffness", {})),
        "damping": deepcopy(actuator.controller.parameters.get("damping", {})),
        "motor_lag_ms": actuator.delay.duration_ms,
    }
    effort = _single_value_clamp(actuator, "effort", "effort_limit")
    velocity = _single_value_clamp(actuator, "velocity", "velocity_limit")
    if effort is not None:
        params["effort_limit_sim"] = effort
    if velocity is not None:
        params["velocity_limit_sim"] = velocity
    for key in ("armature", "dry_friction", "viscous_friction"):
        if key in actuator.physical_parameters:
            params[key] = deepcopy(actuator.physical_parameters[key])
    return params


def build_implicit_actuator_params(bundle: RobotBundle) -> dict[str, Any]:
    actuator_file = load_actuator_file(bundle.paths.actuators)
    if not actuator_file.actuators:
        raise ValueError(f"{bundle.paths.actuators} has no actuator definitions")
    if len(actuator_file.actuators) != 1:
        raise ValueError("SO-101 runtime currently expects one actuator group")
    return _extract_implicit_params_from_actuator(actuator_file.actuators[0])


_FITTED_VECTOR_KEYS = ("stiffness", "damping", "armature", "dry_friction", "viscous_friction")
_FITTED_SCALAR_KEYS = ("motor_lag_ms", "effort_limit_sim", "velocity_limit_sim")
_FITTED_OVERLAY_KEYS = set(_FITTED_VECTOR_KEYS) | set(_FITTED_SCALAR_KEYS)
_FITTED_SUMMARY_KEYS = {"best_mse", "robot_name", "mode", "data_source", "provenance"}


def load_implicit_actuator_params(bundle: RobotBundle, params_path: str | Path | None = None) -> dict[str, Any]:
    if params_path is None:
        return build_implicit_actuator_params(bundle)

    path = Path(params_path)
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "actuators" in loaded or "backend" in loaded:
        actuator_file = ActuatorFile.from_dict(loaded, path)
        if not actuator_file.actuators:
            raise ValueError(f"{path} has no actuator definitions")
        if len(actuator_file.actuators) != 1:
            raise ValueError(f"{path} must contain exactly one actuator group")
        return _extract_implicit_params_from_actuator(actuator_file.actuators[0])

    params = build_implicit_actuator_params(bundle)
    unknown_overlay = sorted(set(loaded) - _FITTED_OVERLAY_KEYS - _FITTED_SUMMARY_KEYS)
    if unknown_overlay:
        raise ValueError(
            f"{path} has unknown actuator-param keys: {', '.join(unknown_overlay)}"
        )
    for key in _FITTED_VECTOR_KEYS:
        if key in loaded:
            existing = params.get(key)
            if isinstance(existing, dict) and isinstance(loaded[key], dict):
                updated = deepcopy(existing)
                updated.update(deepcopy(loaded[key]))
                params[key] = updated
            else:
                params[key] = deepcopy(loaded[key])
    for key in _FITTED_SCALAR_KEYS:
        if key in loaded:
            params[key] = loaded[key]
    return params


def load_sysid_bounds(bundle: RobotBundle) -> SysIdBoundsFile:
    data = yaml.safe_load(bundle.paths.bounds.read_text(encoding="utf-8")) or {}
    return SysIdBoundsFile.from_dict(data, bundle.paths.bounds)


def write_runtime_actuator_yaml(bundle: RobotBundle, output_path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(build_implicit_actuator_params(bundle), sort_keys=False),
        encoding="utf-8",
    )
