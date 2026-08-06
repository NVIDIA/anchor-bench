"""USD export for fitted actuator system-identification parameters."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, Tuple, Union

from anchor_bench.tasks.actuator_sys_id.model import Candidate

_ANGULAR_DRIVES = frozenset({"angular", "rotX", "rotY", "rotZ"})
_LINEAR_DRIVES = frozenset({"linear", "transX", "transY", "transZ"})
_PARAMETER_NAMES = frozenset(
    {
        "stiffness",
        "damping",
        "armature",
        "dry_friction",
        "viscous_friction",
    }
)
_RADIANS_PER_DEGREE = math.pi / 180.0


def write_identified_usd_layer(
    source_usd: Union[str, Path],
    output_path: Union[str, Path],
    best_parameters: Candidate,
) -> Path:
    """Write a sparse USD layer containing the identified actuator parameters.

    The resulting layer sublayers ``source_usd`` and authors only stronger joint
    opinions. It therefore remains dependent on the source asset rather than
    duplicating or flattening it.
    """

    Sdf, Usd, UsdPhysics = _load_usd_modules()
    source_path = Path(source_usd).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"source USD does not exist: {source_path}")
    if source_path == destination:
        raise ValueError("identified USD output must differ from the source USD")
    if not best_parameters:
        raise ValueError("identified USD export requires fitted joint parameters")

    source_stage = Usd.Stage.Open(str(source_path))
    if source_stage is None:
        raise ValueError(f"could not open source USD: {source_path}")
    joints = {
        joint_name: _find_joint(source_stage, UsdPhysics, joint_name)
        for joint_name in best_parameters
    }
    drive_gain_scaling = _joint_drive_gain_scaling(source_stage, UsdPhysics)

    destination.parent.mkdir(parents=True, exist_ok=True)
    output_layer = Sdf.Layer.CreateNew(str(destination))
    if output_layer is None:
        raise RuntimeError(f"could not create identified USD layer: {destination}")
    output_layer.Clear()
    source_identifier = Path(
        os.path.relpath(source_path, start=destination.parent)
    ).as_posix()
    output_layer.subLayerPaths = [source_identifier]

    output_stage = Usd.Stage.Open(output_layer)
    if output_stage is None:
        raise RuntimeError(f"could not open identified USD layer: {destination}")
    source_default_prim = source_stage.GetDefaultPrim()
    if source_default_prim:
        default_prim = output_stage.GetPrimAtPath(source_default_prim.GetPath())
        if default_prim:
            output_stage.SetDefaultPrim(default_prim)

    for joint_name, parameters in best_parameters.items():
        _validate_parameters(joint_name, parameters)
        source_joint = joints[joint_name]
        drive_name, is_angular = _drive(source_joint, UsdPhysics)
        joint_override = output_stage.OverridePrim(source_joint.GetPath())
        joint_override.AddAppliedSchema("NewtonJointAPI")
        drive = UsdPhysics.DriveAPI.Apply(joint_override, drive_name)
        if not drive:
            raise RuntimeError(
                f"could not apply the {drive_name!r} drive to joint {joint_name!r}"
            )

        drive_scale = _drive_gain_export_scale(
            drive_name,
            is_angular=is_angular,
            joint_drive_gain_scaling=drive_gain_scaling,
        )
        drive.CreateStiffnessAttr(float(parameters["stiffness"]) * drive_scale)
        drive.CreateDampingAttr(float(parameters["damping"]) * drive_scale)
        _set_float(
            joint_override,
            Sdf,
            "newton:armature",
            parameters["armature"],
        )
        _set_float(
            joint_override,
            Sdf,
            "newton:friction",
            parameters["dry_friction"],
        )
        damping_scale = _RADIANS_PER_DEGREE if is_angular else 1.0
        _set_float(
            joint_override,
            Sdf,
            "newton:damping",
            float(parameters["viscous_friction"]) * damping_scale,
        )

    if not output_layer.Save():
        raise RuntimeError(f"could not save identified USD layer: {destination}")
    return destination


def _load_usd_modules() -> Tuple[Any, Any, Any]:
    try:
        from pxr import Sdf, Usd, UsdPhysics
    except ImportError as exc:
        raise RuntimeError(
            "USD export requires the simulation dependencies; "
            "install anchor-bench[simulation]"
        ) from exc
    return Sdf, Usd, UsdPhysics


def _find_joint(stage: Any, UsdPhysics: Any, requested: str) -> Any:
    matches = [
        prim
        for prim in stage.Traverse()
        if prim.IsA(UsdPhysics.Joint)
        and (str(prim.GetPath()) == requested or prim.GetName() == requested)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"source USD must contain exactly one joint matching {requested!r}"
        )
    return matches[0]


def _drive(joint: Any, UsdPhysics: Any) -> Tuple[str, bool]:
    if joint.IsA(UsdPhysics.RevoluteJoint):
        return "angular", True
    if joint.IsA(UsdPhysics.PrismaticJoint):
        return "linear", False

    drive_names = _applied_api_instances(joint, "PhysicsDriveAPI")
    if len(drive_names) == 1:
        drive_name = drive_names[0]
    elif not drive_names:
        free_axes = []
        for axis in _applied_api_instances(joint, "PhysicsLimitAPI"):
            limit = UsdPhysics.LimitAPI.Get(joint, axis)
            low = limit.GetLowAttr().Get()
            high = limit.GetHighAttr().Get()
            if low is not None and high is not None and float(low) < float(high):
                free_axes.append(axis)
        if len(free_axes) != 1:
            raise ValueError(
                f"joint {str(joint.GetPath())!r} must have exactly one driven axis"
            )
        drive_name = free_axes[0]
    else:
        raise ValueError(
            f"joint {str(joint.GetPath())!r} must have exactly one driven axis"
        )

    if drive_name in _ANGULAR_DRIVES:
        return drive_name, True
    if drive_name in _LINEAR_DRIVES:
        return drive_name, False
    raise ValueError(
        f"joint {str(joint.GetPath())!r} has unsupported drive {drive_name!r}"
    )


def _applied_api_instances(prim: Any, schema_name: str) -> list[str]:
    prefix = f"{schema_name}:"
    return [
        str(schema).removeprefix(prefix)
        for schema in prim.GetAppliedSchemas()
        if str(schema).startswith(prefix)
    ]


def _joint_drive_gain_scaling(stage: Any, UsdPhysics: Any) -> float:
    scenes = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Scene)]
    scaling = 1.0
    if scenes:
        attribute = scenes[0].GetAttribute("newton:joint_drive_gains_scaling")
        value = attribute.Get() if attribute else None
        if value is not None:
            scaling = float(value)
    if not math.isfinite(scaling) or scaling <= 0.0:
        raise ValueError("newton:joint_drive_gains_scaling must be positive and finite")
    return scaling


def _drive_gain_export_scale(
    drive_name: str,
    *,
    is_angular: bool,
    joint_drive_gain_scaling: float,
) -> float:
    if not is_angular:
        return 1.0
    if drive_name == "angular":
        return _RADIANS_PER_DEGREE / joint_drive_gain_scaling
    return _RADIANS_PER_DEGREE * joint_drive_gain_scaling


def _validate_parameters(joint_name: str, parameters: Dict[str, float]) -> None:
    names = set(parameters)
    if names != _PARAMETER_NAMES:
        missing = sorted(_PARAMETER_NAMES - names)
        extra = sorted(names - _PARAMETER_NAMES)
        details = []
        if missing:
            details.append(f"missing {missing}")
        if extra:
            details.append(f"unexpected {extra}")
        raise ValueError(
            f"invalid parameters for joint {joint_name!r}: " + ", ".join(details)
        )
    values = [float(parameters[name]) for name in _PARAMETER_NAMES]
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError(
            f"identified parameters for joint {joint_name!r} must be finite and nonnegative"
        )


def _set_float(prim: Any, Sdf: Any, name: str, value: float) -> None:
    prim.CreateAttribute(name, Sdf.ValueTypeNames.Float, custom=False).Set(float(value))
