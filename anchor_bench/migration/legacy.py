
from __future__ import annotations

from typing import Any


def migrate_legacy_actuator_config(
    legacy: dict[str, Any],
    *,
    robot_name: str,
    joint_names: list[str],
) -> dict[str, Any]:
    """Convert old model_type configs into Newton 1.2 component schema."""
    model_type = legacy.get("model_type", "implicit")
    params = dict(legacy.get("parameters", {}))
    if "dynamic_friction" in params and "dry_friction" not in params:
        params["dry_friction"] = params["dynamic_friction"]
    controller_type = {
        "implicit": "pd",
        "dcmotor": "pd",
        "lstm": "lstm",
        "gru": "gru",
        "lstm_perjoint": "lstm",
        "gru_perjoint": "gru",
        "fmu": "fmu",
        "hybrid_residual": "pd",
    }.get(model_type)
    if controller_type is None:
        raise ValueError(f"unsupported legacy model_type '{model_type}'")
    clamps = []
    if "effort_limit_sim" in params or "effort_limit" in params:
        clamps.append({"type": "effort", "parameters": {"effort_limit": params.get("effort_limit_sim", params.get("effort_limit"))}})
    if "velocity_limit_sim" in params or "velocity_limit" in params:
        clamps.append({"type": "velocity", "parameters": {"velocity_limit": params.get("velocity_limit_sim", params.get("velocity_limit"))}})
    actuator: dict[str, Any] = {
        "name": f"{robot_name}_actuator",
        "joints": joint_names,
        "controller": {
            "type": controller_type,
            "source": f"migrated from legacy {model_type}",
            "parameters": {
                "stiffness": params.get("stiffness"),
                "damping": params.get("damping"),
            },
        },
        "delay": {"duration_ms": float(legacy.get("motor_lag_ms", params.get("motor_lag_ms", 0.0)) or 0.0)},
        "clamps": clamps,
        "physical_parameters": {
            key: params[key]
            for key in ("armature", "dry_friction", "viscous_friction", "friction")
            if key in params
        },
    }
    if model_type == "hybrid_residual":
        actuator["residual"] = {"type": "gru", "network_files": legacy.get("network_files", {})}
    if "network_file" in legacy:
        actuator["controller"]["network_file"] = legacy["network_file"]
    if "fmu_path" in legacy:
        actuator["controller"]["parameters"]["fmu_path"] = legacy["fmu_path"]
        actuator["controller"]["parameters"]["step_size"] = legacy.get("fmu_step_size")
    return {"schema_version": 2, "backend": "newton", "actuators": [actuator]}
