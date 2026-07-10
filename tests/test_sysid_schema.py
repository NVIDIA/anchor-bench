import math

import pytest


def _profile_payload() -> dict:
    return {
        "schema_version": 1,
        "data": "inputs/datasets/demo/csv",
        "joints": ["joint_a", "joint_b"],
        "simulation": {
            "physics_freq": 500,
            "control_freq": 500,
            "render_freq": 500,
            "integrator": "implicitfast",
            "solver_iterations": 10,
            "line_search_iterations": 10,
        },
        "num_envs": 64,
        "buffer_time": 2.0,
        "seed": 42,
        "optimizer": {
            "type": "cmaes",
            "sigma": 0.5,
            "max_iterations": 100,
            "epsilon": 0.001,
        },
        "parameters": {
            "stiffness": {
                "mode": "per_joint",
                "bounds": {"lower": 1.0, "upper": 80.0},
            },
            "damping": {
                "mode": "shared",
                "bounds": {"lower": 0.01, "upper": 5.0},
            },
        },
        "loss": {"position": 1.0, "velocity": 1.0},
    }


def test_sysid_profile_round_trip_is_strict_and_host_safe() -> None:
    from pydantic import BaseModel, ValidationError

    from anchor_bench.sysid.schema import SysIdProfile

    profile = SysIdProfile.from_dict(_profile_payload(), "profile.yaml")

    assert isinstance(profile, BaseModel)
    assert profile.parameters["stiffness"].mode == "per_joint"
    assert profile.parameters["damping"].mode == "shared"
    assert profile.to_dict() == _profile_payload()
    with pytest.raises(ValidationError):
        profile.seed = 7


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda data: data.update(typo=True), "unknown keys: typo"),
        (
            lambda data: data["parameters"].update(motor_lag={"mode": "shared", "bounds": {"lower": 0, "upper": 1}}),
            "unsupported SysID parameter",
        ),
        (
            lambda data: data["parameters"]["stiffness"]["bounds"].update(lower=5.0, upper=1.0),
            "upper.*must be >= lower",
        ),
        (
            lambda data: data["parameters"]["stiffness"]["bounds"].update(lower=math.inf),
            "finite number",
        ),
        (
            lambda data: data["parameters"]["stiffness"]["bounds"].update(lower=-1.0),
            "must be >= 0.0",
        ),
        (lambda data: data.update(joints=["joint_a", "joint_a"]), "duplicate joint"),
        (lambda data: data.update(joints=["joint_a", ""]), "joint names must be non-empty"),
        (lambda data: data.update(loss={"position": 0.0, "velocity": 0.0}), "at least one loss weight"),
    ],
)
def test_sysid_profile_rejects_invalid_contract(mutate, message: str) -> None:
    from anchor_bench.sysid.schema import SysIdProfile

    data = _profile_payload()
    mutate(data)

    with pytest.raises(ValueError, match=message):
        SysIdProfile.from_dict(data, "profile.yaml")


def test_parameter_layout_expands_per_joint_and_shared_values() -> None:
    from anchor_bench.sysid.schema import (
        SysIdProfile,
        expand_parameter_values,
        parameter_layout,
    )

    data = _profile_payload()
    # Input mapping order must not change the canonical optimizer layout.
    data["parameters"] = {
        "damping": data["parameters"]["damping"],
        "stiffness": data["parameters"]["stiffness"],
    }
    profile = SysIdProfile.from_dict(data, "profile.yaml")

    layout = parameter_layout(profile)
    assert [(entry.parameter, entry.joints) for entry in layout] == [
        ("stiffness", ("joint_a",)),
        ("stiffness", ("joint_b",)),
        ("damping", ("joint_a", "joint_b")),
    ]
    assert expand_parameter_values(profile, [10.0, 20.0, 0.5]) == {
        "stiffness": {"joint_a": 10.0, "joint_b": 20.0},
        "damping": {"joint_a": 0.5, "joint_b": 0.5},
    }


def test_parameter_expansion_validates_count_finiteness_and_bounds() -> None:
    from anchor_bench.sysid.schema import SysIdProfile, expand_parameter_values

    profile = SysIdProfile.from_dict(_profile_payload(), "profile.yaml")

    with pytest.raises(ValueError, match="expected 3 optimizer values"):
        expand_parameter_values(profile, [1.0, 2.0])
    with pytest.raises(ValueError, match="must be finite"):
        expand_parameter_values(profile, [1.0, 2.0, math.nan])
    with pytest.raises(ValueError, match="outside bounds"):
        expand_parameter_values(profile, [1.0, 81.0, 0.5])


def test_profile_joint_selection_must_already_be_canonical() -> None:
    from anchor_bench.sysid.schema import SysIdProfile, validate_profile_joints

    profile = SysIdProfile.from_dict(_profile_payload(), "profile.yaml")

    assert validate_profile_joints(profile, ["joint_b", "joint_a", "joint_c"]) == (
        "joint_a",
        "joint_b",
    )
    with pytest.raises(ValueError, match="non-canonical or unknown joints: joint_b"):
        validate_profile_joints(profile.model_copy(update={"joints": ["joint_a", "joint_b_alias"]}), ["joint_a", "joint_b"])
