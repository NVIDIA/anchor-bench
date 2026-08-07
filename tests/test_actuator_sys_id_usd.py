# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

import math

import pytest
from pxr import Sdf, Usd, UsdGeom, UsdPhysics

from anchor_bench.tasks.actuator_sys_id.usd import write_identified_usd_layer

# The OpenUSD extension modules do not publish complete Python type information.


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


def _create_source_usd(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.Cube.Define(stage, "/World/visual")

    scene = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
    scene.GetPrim().CreateAttribute(
        "newton:joint_drive_gains_scaling",
        Sdf.ValueTypeNames.Float,
        custom=True,
    ).Set(2.0)
    UsdPhysics.RevoluteJoint.Define(stage, "/World/hip")
    UsdPhysics.PrismaticJoint.Define(stage, "/World/slider")

    d6_joint = UsdPhysics.Joint.Define(stage, "/World/wrist")
    limit = UsdPhysics.LimitAPI.Apply(d6_joint.GetPrim(), "rotX")
    limit.CreateLowAttr(-45.0)
    limit.CreateHighAttr(45.0)
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
    )

    assert result == output.resolve()
    root_layer = Sdf.Layer.FindOrOpen(str(output))
    assert root_layer.subLayerPaths == ["../assets/robot.usda"]
    assert root_layer.GetPrimAtPath("/World/visual") is None
    hip_spec = root_layer.GetPrimAtPath("/World/hip")
    assert "NewtonJointAPI" in hip_spec.GetInfo("apiSchemas").GetAddedOrExplicitItems()

    stage = Usd.Stage.Open(str(output))
    assert stage.GetDefaultPrim().GetPath() == Sdf.Path("/World")
    assert stage.GetPrimAtPath("/World/visual")

    hip = stage.GetPrimAtPath("/World/hip")
    hip_drive = UsdPhysics.DriveAPI.Get(hip, "angular")
    radians_per_degree = math.pi / 180.0
    assert hip_drive.GetStiffnessAttr().Get() == pytest.approx(
        20.0 * radians_per_degree / 2.0
    )
    assert hip_drive.GetDampingAttr().Get() == pytest.approx(
        5.0 * radians_per_degree / 2.0
    )
    assert hip.GetAttribute("newton:armature").Get() == pytest.approx(0.05)
    assert hip.GetAttribute("newton:friction").Get() == pytest.approx(0.5)
    assert hip.GetAttribute("newton:damping").Get() == pytest.approx(radians_per_degree)

    slider = stage.GetPrimAtPath("/World/slider")
    slider_drive = UsdPhysics.DriveAPI.Get(slider, "linear")
    assert slider_drive.GetStiffnessAttr().Get() == pytest.approx(30.0)
    assert slider_drive.GetDampingAttr().Get() == pytest.approx(6.0)
    assert slider.GetAttribute("newton:damping").Get() == pytest.approx(2.0)


def test_infers_a_scalar_d6_axis_and_uses_newton_angular_scaling(tmp_path):
    source = tmp_path / "robot.usda"
    output = tmp_path / "identified.usda"
    _create_source_usd(source)

    write_identified_usd_layer(
        source,
        output,
        {"wrist": _parameters()},
    )

    stage = Usd.Stage.Open(str(output))
    wrist = stage.GetPrimAtPath("/World/wrist")
    drive = UsdPhysics.DriveAPI.Get(wrist, "rotX")
    assert drive
    assert drive.GetStiffnessAttr().Get() == pytest.approx(
        20.0 * (math.pi / 180.0) * 2.0
    )


def test_preserves_a_supplied_symlink_as_the_sublayer_path(tmp_path):
    source = tmp_path / "cache" / "robot.usda"
    link = tmp_path / "assets" / "robot.usda"
    output = tmp_path / "assets" / "identified.usda"
    _create_source_usd(source)
    link.parent.mkdir()
    link.symlink_to(source)

    write_identified_usd_layer(link, output, {"hip": _parameters()})

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
        write_identified_usd_layer(source, output, {"hip": _parameters()})
