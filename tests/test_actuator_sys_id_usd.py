# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

import json
import math

import pytest
from pxr import Sdf, Usd, UsdGeom, UsdPhysics

from anchor_bench.schema.actuator_sys_id import ControllerClamps
from anchor_bench.tasks.actuator_sys_id.usd import write_identified_usd_layer

# The OpenUSD extension modules do not publish complete Python type information.

_CONFIG_METADATA = '{"type":"actuator_sys_id","seed":7}'


def _parameters(**overrides):
    values = {
        "stiffness": 20.0,
        "damping": 5.0,
        "armature": 0.05,
        "dry_friction": 0.5,
        "viscous_friction": 1.0,
    }
    values.update(overrides)
    return values


def _clamps(
    effort_limit: float = 120.0,
    velocity_limit: float = 30.0,
) -> ControllerClamps:
    return ControllerClamps(
        effort_limit=effort_limit,
        velocity_limit=velocity_limit,
    )


def _create_source_usd(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 0.25)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 2.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.Cube.Define(stage, "/World/visual")

    scene = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
    scene.GetPrim().CreateAttribute(
        "newton:joint_drive_gains_scaling",
        Sdf.ValueTypeNames.Float,
        custom=True,
    ).Set(2.0)
    hip = UsdPhysics.RevoluteJoint.Define(stage, "/World/hip")
    for name, value in (
        ("mjc:armature", 0.028),
        ("mjc:damping", 0.6),
        ("mjc:frictionloss", 0.052),
    ):
        hip.GetPrim().CreateAttribute(
            name,
            Sdf.ValueTypeNames.Double,
            custom=False,
        ).Set(value)
    UsdPhysics.PrismaticJoint.Define(stage, "/World/slider")
    link = UsdGeom.Xform.Define(stage, "/World/link").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(link)
    link_mass = UsdPhysics.MassAPI.Apply(link)
    link_mass.CreateMassAttr(3.0)
    link_mass.CreateDiagonalInertiaAttr((1.0, 2.0, 3.0))
    link.CreateAttribute(
        "newton:inertia",
        Sdf.ValueTypeNames.DoubleArray,
        custom=False,
    ).Set([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])

    d6_joint = UsdPhysics.Joint.Define(stage, "/World/wrist")
    limit = UsdPhysics.LimitAPI.Apply(d6_joint.GetPrim(), "rotX")
    limit.CreateLowAttr(-45.0)
    limit.CreateHighAttr(45.0)

    hip_actuator = stage.DefinePrim("/World/hip_actuator", "MjcActuator")
    hip_actuator.CreateRelationship("mjc:target").SetTargets([hip.GetPath()])
    wrist_actuator = stage.DefinePrim("/World/wrist_actuator", "MjcActuator")
    wrist_actuator.CreateRelationship("mjc:target").SetTargets([d6_joint.GetPath()])
    stage.GetRootLayer().Save()


def test_writes_sparse_opinions_over_the_source_asset(tmp_path):
    source = tmp_path / "assets" / "robot.usda"
    output = tmp_path / "results" / "identified.usda"
    _create_source_usd(source)

    result = write_identified_usd_layer(
        source,
        output,
        {
            "hip": _parameters(),
            "/World/slider": _parameters(
                stiffness=30.0,
                damping=6.0,
                viscous_friction=2.0,
            ),
        },
        {
            "hip": _clamps(),
            "/World/slider": _clamps(effort_limit=80.0, velocity_limit=4.0),
        },
        _CONFIG_METADATA,
    )

    assert result == output.resolve()
    root_layer = Sdf.Layer.FindOrOpen(str(output))
    assert root_layer.subLayerPaths == ["../assets/robot.usda"]
    assert json.loads(root_layer.customLayerData["config"]) == {
        "type": "actuator_sys_id",
        "seed": 7,
    }
    assert root_layer.GetPrimAtPath("/World/visual") is None
    hip_spec = root_layer.GetPrimAtPath("/World/hip")
    assert "NewtonJointAPI" in hip_spec.GetInfo("apiSchemas").GetAddedOrExplicitItems()

    stage = Usd.Stage.Open(str(output))
    assert stage.GetDefaultPrim().GetPath() == Sdf.Path("/World")
    assert stage.GetPrimAtPath("/World/visual")
    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
    assert UsdGeom.GetStageMetersPerUnit(stage) == pytest.approx(0.25)
    assert UsdPhysics.GetStageKilogramsPerUnit(stage) == pytest.approx(2.0)

    hip = stage.GetPrimAtPath("/World/hip")
    hip_drive = UsdPhysics.DriveAPI.Get(hip, "angular")
    radians_per_degree = math.pi / 180.0
    assert hip_drive.GetStiffnessAttr().Get() == pytest.approx(
        20.0 * radians_per_degree / 2.0
    )
    assert hip_drive.GetDampingAttr().Get() == pytest.approx(
        5.0 * radians_per_degree / 2.0
    )
    assert hip_drive.GetMaxForceAttr().Get() == pytest.approx(120.0)
    assert hip.GetAttribute("newton:armature").Get() == pytest.approx(0.05)
    assert hip.GetAttribute("newton:friction").Get() == pytest.approx(0.5)
    assert hip.GetAttribute("newton:damping").Get() == pytest.approx(radians_per_degree)
    assert hip.GetAttribute("newton:velocityLimit").Get() == pytest.approx(
        30.0 / radians_per_degree
    )
    assert hip.GetAttribute("mjc:armature").Get() is None
    assert hip.GetAttribute("mjc:damping").Get() is None
    assert hip.GetAttribute("mjc:frictionloss").Get() is None
    assert not stage.GetPrimAtPath("/World/hip_actuator").IsActive()
    assert stage.GetPrimAtPath("/World/wrist_actuator").IsActive()

    slider = stage.GetPrimAtPath("/World/slider")
    slider_drive = UsdPhysics.DriveAPI.Get(slider, "linear")
    assert slider_drive.GetStiffnessAttr().Get() == pytest.approx(30.0)
    assert slider_drive.GetDampingAttr().Get() == pytest.approx(6.0)
    assert slider_drive.GetMaxForceAttr().Get() == pytest.approx(80.0)
    assert slider.GetAttribute("newton:damping").Get() == pytest.approx(2.0)
    assert slider.GetAttribute("newton:velocityLimit").Get() == pytest.approx(4.0)


def test_infers_a_scalar_d6_axis_and_uses_newton_angular_scaling(tmp_path):
    source = tmp_path / "robot.usda"
    output = tmp_path / "identified.usda"
    _create_source_usd(source)

    write_identified_usd_layer(
        source,
        output,
        {"wrist": _parameters()},
        {"wrist": _clamps()},
        _CONFIG_METADATA,
    )

    stage = Usd.Stage.Open(str(output))
    wrist = stage.GetPrimAtPath("/World/wrist")
    drive = UsdPhysics.DriveAPI.Get(wrist, "rotX")
    assert drive
    assert drive.GetStiffnessAttr().Get() == pytest.approx(
        20.0 * (math.pi / 180.0) * 2.0
    )


def test_writes_identified_mass_and_proportionally_scaled_inertia(tmp_path):
    source = tmp_path / "robot.usda"
    output = tmp_path / "identified.usda"
    _create_source_usd(source)

    write_identified_usd_layer(
        source,
        output,
        {"hip": _parameters()},
        {"hip": _clamps()},
        _CONFIG_METADATA,
        {"link": 12.0},
    )

    stage = Usd.Stage.Open(str(output))
    mass_api = UsdPhysics.MassAPI(stage.GetPrimAtPath("/World/link"))
    # The test stage uses two kilograms per authored mass unit. Its source mass
    # is therefore 6 kg, so the 12 kg result doubles the source inertia.
    assert mass_api.GetMassAttr().Get() == pytest.approx(6.0)
    assert mass_api.GetDiagonalInertiaAttr().Get() == pytest.approx((2.0, 4.0, 6.0))
    assert stage.GetPrimAtPath("/World/link").GetAttribute(
        "newton:inertia"
    ).Get() == pytest.approx((2.0, 4.0, 6.0, 0.2, 0.4, 0.6))


def test_preserves_a_supplied_symlink_as_the_sublayer_path(tmp_path):
    source = tmp_path / "cache" / "robot.usda"
    link = tmp_path / "assets" / "robot.usda"
    output = tmp_path / "assets" / "identified.usda"
    _create_source_usd(source)
    link.parent.mkdir()
    link.symlink_to(source)

    write_identified_usd_layer(
        link,
        output,
        {"hip": _parameters()},
        {"hip": _clamps()},
        _CONFIG_METADATA,
    )

    layer = Sdf.Layer.FindOrOpen(str(output))
    assert layer.subLayerPaths == ["robot.usda"]


def test_rejects_an_ambiguous_joint_name(tmp_path):
    source = tmp_path / "robot.usda"
    output = tmp_path / "identified.usda"
    _create_source_usd(source)
    stage = Usd.Stage.Open(str(source))
    UsdPhysics.RevoluteJoint.Define(stage, "/World/other/hip")
    stage.GetRootLayer().Save()

    with pytest.raises(ValueError, match="exactly one joint matching 'hip'"):
        write_identified_usd_layer(
            source,
            output,
            {"hip": _parameters()},
            {"hip": _clamps()},
            _CONFIG_METADATA,
        )


def test_rejects_controller_clamps_for_different_joints(tmp_path):
    source = tmp_path / "robot.usda"
    output = tmp_path / "identified.usda"
    _create_source_usd(source)

    with pytest.raises(ValueError, match="must exactly match"):
        write_identified_usd_layer(
            source,
            output,
            {"hip": _parameters()},
            {"slider": _clamps()},
            _CONFIG_METADATA,
        )
