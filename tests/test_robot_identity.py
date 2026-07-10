from __future__ import annotations

import pytest

from anchor_bench.robots.schema import RobotIdentity


def _robot() -> dict:
    """Return one valid minimal robot identity document."""

    return {
        "schema_version": 1,
        "name": "demo",
        "display_name": "Demo Robot",
        "fixed_base": True,
        "asset_path": "assets/robots/demo/manifest.yaml",
        "runtime_usd": "demo.usd",
        "joint_names": ["joint_a", "joint_b"],
        "joint_aliases": {"a": "joint_a"},
        "ee_frame": "tool",
    }


def test_robot_identity_contains_only_identity_and_runtime_asset_fields() -> None:
    """Accept the new robot surface and reject removed phase-layout fields."""

    robot = RobotIdentity.from_dict(_robot(), "robot.yaml")
    assert robot.joint_names == ("joint_a", "joint_b")
    assert robot.runtime_usd == "demo.usd"

    for legacy in ("actuator_file", "bounds_file", "run_file"):
        payload = {**_robot(), legacy: f"{legacy}.yaml"}
        with pytest.raises(ValueError, match=f"unknown keys: {legacy}"):
            RobotIdentity.from_dict(payload, "robot.yaml")


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"asset_path": "assets/robots/other/manifest.yaml"}, "asset_path must be"),
        ({"runtime_usd": "robot.xml"}, "must name a USD"),
        ({"joint_names": ["joint_a", "joint_a"]}, "duplicate joint_names"),
        ({"joint_aliases": {"joint_a": "joint_b"}}, "collide"),
        ({"joint_aliases": {"bad": "missing"}}, "target unknown"),
    ],
)
def test_robot_identity_rejects_ambiguous_or_cross_robot_configuration(
    update: dict,
    message: str,
) -> None:
    """Keep canonical axes and asset ownership unambiguous."""

    payload = {**_robot(), **update}
    with pytest.raises(ValueError, match=message):
        RobotIdentity.from_dict(payload, "robot.yaml")
