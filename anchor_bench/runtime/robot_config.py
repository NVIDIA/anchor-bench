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


def _per_joint_values(value: Any, joint_names: list[str]) -> dict[str, Any]:
    if isinstance(value, dict):
        return deepcopy(value)
    return {joint: deepcopy(value) for joint in joint_names}


def _extract_implicit_params_from_actuator(actuator: ActuatorSpec) -> dict[str, Any]:
    if actuator.controller.type != "pd":
        raise ValueError(
            f"implicit runtime requires a pd controller; actuator {actuator.name!r} "
            f"uses {actuator.controller.type!r}"
        )
    params: dict[str, Any] = {
        "stiffness": _per_joint_values(
            actuator.controller.parameters.get("stiffness", {}), actuator.joints
        ),
        "damping": _per_joint_values(
            actuator.controller.parameters.get("damping", {}), actuator.joints
        ),
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
            params[key] = _per_joint_values(
                actuator.physical_parameters[key], actuator.joints
            )
    return params


def build_implicit_actuator_params(bundle: RobotBundle) -> dict[str, Any]:
    actuator_file = load_actuator_file(bundle.paths.actuators)
    if not actuator_file.actuators:
        raise ValueError(f"{bundle.paths.actuators} has no actuator definitions")
    if len(actuator_file.actuators) != 1:
        raise ValueError(
            f"implicit runtime for robot {bundle.spec.name!r} expects exactly one actuator group"
        )
    return _extract_implicit_params_from_actuator(actuator_file.actuators[0])


def load_implicit_actuator_params(
    bundle: RobotBundle,
    params_path: str | Path | None = None,
) -> dict[str, Any]:
    actuator_file = load_actuator_file(params_path or bundle.paths.actuators)
    if not actuator_file.actuators:
        raise ValueError(f"{params_path or bundle.paths.actuators} has no actuator definitions")
    if len(actuator_file.actuators) != 1:
        raise ValueError(f"{params_path or bundle.paths.actuators} must contain exactly one actuator group")
    return _extract_implicit_params_from_actuator(actuator_file.actuators[0])


def build_fitted_actuator_document(
    bundle: RobotBundle,
    fitted: dict[str, dict[str, float]],
    *,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    document = load_actuator_file(bundle.paths.actuators).to_dict()
    actuators = document["actuators"]
    if len(actuators) != 1:
        raise ValueError(f"{bundle.paths.actuators} must contain exactly one actuator group")
    actuator = actuators[0]
    controller = actuator["controller"]["parameters"]
    physical = actuator["physical_parameters"]
    for name in ("stiffness", "damping"):
        if name in fitted:
            current = controller.get(name)
            if isinstance(current, dict):
                current.update(fitted[name])
            else:
                controller[name] = fitted[name]
    for name in ("armature", "dry_friction", "viscous_friction"):
        if name in fitted:
            current = physical.get(name)
            if isinstance(current, dict):
                current.update(fitted[name])
            else:
                physical[name] = fitted[name]
    document["metadata"] = metadata
    ActuatorFile.from_dict(document, "fitted actuator document")
    return document


def load_sysid_bounds(bundle: RobotBundle) -> SysIdBoundsFile:
    data = yaml.safe_load(bundle.paths.bounds.read_text(encoding="utf-8")) or {}
    return SysIdBoundsFile.from_dict(data, bundle.paths.bounds)
