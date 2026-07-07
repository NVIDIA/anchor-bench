
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

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
    """Return composable Newton actuator component descriptions.

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


def _collect_joint_prims(stage: Any, articulation_prim_path: str) -> dict[str, str]:
    from pxr import Usd  # noqa: PLC0415

    art_prim = stage.GetPrimAtPath(articulation_prim_path)
    if not art_prim.IsValid():
        raise ValueError(f"Articulation prim not found: {articulation_prim_path}")

    joint_types = {"PhysicsRevoluteJoint", "PhysicsPrismaticJoint"}
    joints: dict[str, str] = {}
    for prim in Usd.PrimRange(art_prim):
        if prim.GetTypeName() in joint_types:
            joints[prim.GetName()] = str(prim.GetPath())
    return joints


def _param_for_joint(value: Any, joint_name: str, default: float = 0.0) -> float:
    if isinstance(value, dict):
        if joint_name in value:
            return float(value[joint_name])
        return default
    if value is None:
        return default
    return float(value)


def _effort_limit(actuator) -> float | None:
    for clamp in actuator.clamps:
        if clamp.type == "effort":
            value = clamp.parameters.get("effort_limit", clamp.parameters.get("max_effort"))
            return None if value is None else float(value)
    return None


def _set_api_schemas(prim: Any, schemas: list[str]) -> None:
    from pxr import Sdf  # noqa: PLC0415

    existing = prim.GetMetadata("apiSchemas") or Sdf.TokenListOp()
    existing.prependedItems = list(schemas)
    prim.SetMetadata("apiSchemas", existing)


def _author_prim(
    *,
    stage: Any,
    act_prim_path: str,
    joint_prim_path: str,
    schemas: list[str],
    attrs: dict[str, float | int],
    model_path: str | None = None,
) -> None:
    from pxr import Sdf  # noqa: PLC0415

    act_prim = stage.DefinePrim(act_prim_path, "NewtonActuator")
    _set_api_schemas(act_prim, schemas)

    rel = act_prim.CreateRelationship("newton:targets")
    rel.SetTargets([Sdf.Path(joint_prim_path)])

    if model_path is not None:
        act_prim.CreateAttribute("newton:modelPath", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(model_path))

    for attr_name, attr_val in attrs.items():
        usd_name = f"newton:{attr_name}"
        if isinstance(attr_val, int):
            act_prim.CreateAttribute(usd_name, Sdf.ValueTypeNames.Int).Set(attr_val)
        else:
            act_prim.CreateAttribute(usd_name, Sdf.ValueTypeNames.Float).Set(float(attr_val))


def _author_pd_actuator(stage: Any, articulation_prim_path: str, actuator, joint_name: str, joint_prim_path: str) -> None:
    params = actuator.controller.parameters
    attrs: dict[str, float | int] = {
        "kp": _param_for_joint(params.get("stiffness"), joint_name),
        "kd": _param_for_joint(params.get("damping"), joint_name),
    }
    schemas = ["NewtonPDControlAPI"]
    effort = _effort_limit(actuator)
    if effort is not None:
        schemas.append("NewtonMaxEffortClampingAPI")
        attrs["maxEffort"] = effort
    _author_prim(
        stage=stage,
        act_prim_path=f"{articulation_prim_path}/{actuator.name}_{joint_name}_pd_actuator",
        joint_prim_path=joint_prim_path,
        schemas=schemas,
        attrs=attrs,
    )


GRU_RUNTIME_SHIM_MESSAGE = (
    "GRU actuators are not enabled through the pinned Isaac Lab adapter; "
    "use anchor_bench.runtime.actuator_net_gru for full-torque GRU and GRU residual runtime support."
)


def _ensure_newton_native_supported(spec: ActuatorFile) -> None:
    for actuator in spec.actuators:
        has_gru_residual = actuator.residual is not None and actuator.residual.type == "gru"
        if actuator.controller.type == "gru" or has_gru_residual:
            raise RuntimeError(GRU_RUNTIME_SHIM_MESSAGE)


def author_newton_actuator_prims(spec: ActuatorFile, *, stage: Any, articulation_prim_path: str) -> None:
    """Author ``NewtonActuator`` prims from an Anchor Bench actuator spec.

    The prims are intentionally loaded by Newton later via
    ``NewtonActuatorAdapter.from_usd`` or ``ModelBuilder.add_usd``. GRU
    controllers and residuals stay on ``anchor_bench.runtime.actuator_net_gru``
    until the pinned Isaac Lab adapter exposes Newton's native GRU controller
    through its USD loading path.
    """
    _ensure_newton_native_supported(spec)
    joint_prims = _collect_joint_prims(stage, articulation_prim_path)
    for actuator in spec.actuators:
        for joint_name in actuator.joints:
            if joint_name not in joint_prims:
                raise ValueError(f"joint {joint_name!r} not found under {articulation_prim_path}")
            joint_prim_path = joint_prims[joint_name]
            if actuator.controller.type == "pd":
                _author_pd_actuator(stage, articulation_prim_path, actuator, joint_name, joint_prim_path)
            else:
                raise ValueError(
                    f"Newton-native actuator loading does not support controller {actuator.controller.type!r}"
                )


def _missing_runtime_args(**kwargs: Any) -> list[str]:
    return [name for name, value in kwargs.items() if value is None]


def instantiate_newton_actuators(
    spec: ActuatorFile,
    *,
    stage: Any | None = None,
    articulation_prim_path: str | None = None,
    joint_names: Sequence[str] | None = None,
    num_envs: int | None = None,
    num_joints: int | None = None,
    device: str | None = None,
):
    """Author and load actuators through Newton's native actuator adapter.

    Use :func:`build_newton_component_dicts` for host-only dry-run inspection.
    Runtime instantiation needs a USD stage and articulation context so Newton
    can parse the authored ``NewtonActuator`` prims itself.
    """
    missing = _missing_runtime_args(
        stage=stage,
        articulation_prim_path=articulation_prim_path,
        joint_names=joint_names,
        num_envs=num_envs,
        num_joints=num_joints,
        device=device,
    )
    if missing:
        raise RuntimeError(
            "Newton actuator instantiation requires stage, articulation_prim_path, joint_names, "
            "num_envs, num_joints, and device; use build_newton_component_dicts for dry-run inspection."
        )
    _ensure_newton_native_supported(spec)

    try:
        from isaaclab_newton.actuators import NewtonActuatorAdapter  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - runtime-only guard
        raise RuntimeError(
            "Newton-native actuator construction requires isaaclab_newton.actuators in the full runtime."
        ) from exc

    assert stage is not None
    assert articulation_prim_path is not None
    assert joint_names is not None
    assert num_envs is not None
    assert num_joints is not None
    assert device is not None

    author_newton_actuator_prims(spec, stage=stage, articulation_prim_path=articulation_prim_path)
    return NewtonActuatorAdapter.from_usd(
        stage=stage,
        joint_names=list(joint_names),
        num_envs=num_envs,
        num_joints=num_joints,
        device=device,
        articulation_prim_path=articulation_prim_path,
    )
