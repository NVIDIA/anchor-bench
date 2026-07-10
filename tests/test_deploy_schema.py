from __future__ import annotations

import copy
import math

import pytest
from pydantic import ValidationError

from anchor_bench.deploy.schema import DeployProfile, FrozenDeployState


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64


def _simulation() -> dict:
    return {
        "physics_freq": 500,
        "control_freq": 500,
        "render_freq": 500,
        "integrator": "implicitfast",
        "solver_iterations": 10,
        "line_search_iterations": 10,
    }


def _profile_payload() -> dict:
    return {
        "schema_version": 1,
        "simulation": _simulation(),
        "composition": {
            "neural": [
                {
                    "name": "right_arm",
                    "model": "right_arm_gru",
                    "mappings": [
                        {
                            "index": 0,
                            "input_joints": ["context", "elbow"],
                            "output_joints": ["elbow"],
                        },
                        {
                            "index": 1,
                            "input_joints": ["context", "wrist"],
                            "output_joints": ["wrist"],
                        },
                    ],
                }
            ],
            "pd": [
                {
                    "name": "left_arm",
                    "joints": ["shoulder"],
                    "effort_limits": {"shoulder": 25.0},
                }
            ],
        },
    }


def _frozen_payload() -> dict:
    return {
        "schema_version": 1,
        "simulation": _simulation(),
        "ownership": {"elbow": "right_arm", "shoulder": "left_arm"},
        "composition": {
            "neural": [
                {
                    "name": "right_arm",
                    "model": "right_arm_gru",
                    "accepted_model_revision": 3,
                    "artifact": {
                        "path": f"outputs/train/demo/right_arm_gru/seed-42/{HASH_A}/network.pt",
                        "sha256": HASH_B,
                        "metadata_sha256": HASH_C,
                    },
                    "target": "torque_residual",
                    "sample_dt_s": 0.002,
                    "mappings": [
                        {
                            "index": 0,
                            "input_joints": ["context", "elbow"],
                            "output_joints": ["elbow"],
                            "pd_baseline": {
                                "kp": {"context": 10.0, "elbow": 20.0},
                                "kd": {"context": 0.5, "elbow": 0.75},
                                "velocity_target": "zero",
                            },
                            "friction_baseline": {
                                "dry": {"elbow": 0.1},
                                "viscous": {"elbow": 0.02},
                            },
                            "armature": {"elbow": 0.005},
                            "effort_limits": {"elbow": 25.0},
                            "clamping": "residual_output",
                        }
                    ],
                }
            ],
            "pd": [
                {
                    "name": "left_arm",
                    "joints": ["shoulder"],
                    "kp": {"shoulder": 30.0},
                    "kd": {"shoulder": 0.8},
                    "dry_friction": {"shoulder": 0.1},
                    "viscous_friction": {"shoulder": 0.02},
                    "armature": {"shoulder": 0.006},
                    "effort_limits": {"shoulder": 25.0},
                }
            ],
        },
        "source": {
            "profile": "default",
            "profile_sha256": HASH_A,
            "sysid_revision": 4,
            "sysid_state_sha256": HASH_B,
            "train_revision": 5,
            "train_state_sha256": HASH_C,
        },
    }


def test_deploy_profile_allows_context_overlap_but_has_disjoint_outputs() -> None:
    profile = DeployProfile.from_dict(_profile_payload(), "deploy/profile.yaml")

    mappings = profile.composition.neural[0].mappings
    assert mappings[0].input_joints[0] == mappings[1].input_joints[0] == "context"
    assert mappings[0].output_joints == ("elbow",)
    with pytest.raises(ValidationError):
        profile.simulation = profile.simulation


@pytest.mark.parametrize("selector", ["all", "remaining", "exclude"])
def test_deploy_profile_rejects_selection_keywords(selector: str) -> None:
    payload = _profile_payload()
    payload["composition"]["neural"][0]["mappings"][0]["input_joints"] = selector

    with pytest.raises(ValueError, match="selection keyword"):
        DeployProfile.from_dict(payload, "deploy/profile.yaml")


def test_deploy_profile_rejects_output_ownership_overlap() -> None:
    payload = _profile_payload()
    payload["composition"]["pd"][0] = {
        "name": "left_arm",
        "joints": ["elbow"],
        "effort_limits": {"elbow": 25.0},
    }

    with pytest.raises(ValueError, match="output joint 'elbow' is owned by both"):
        DeployProfile.from_dict(payload, "deploy/profile.yaml")


@pytest.mark.parametrize(
    ("limits", "message"),
    [
        ({"other": 25.0}, "keys must exactly match joints"),
        ({"shoulder": 0.0}, "must contain positive values"),
        ({"shoulder": math.inf}, "finite number"),
    ],
)
def test_deploy_profile_requires_exact_positive_pd_effort_limits(
    limits: dict[str, float],
    message: str,
) -> None:
    payload = _profile_payload()
    payload["composition"]["pd"][0]["effort_limits"] = limits

    with pytest.raises(ValueError, match=message):
        DeployProfile.from_dict(payload, "deploy/profile.yaml")


def test_deploy_profile_rejects_duplicate_names_indices_and_unknown_keys() -> None:
    payload = _profile_payload()
    payload["composition"]["pd"][0]["name"] = "right_arm"
    with pytest.raises(ValueError, match="duplicate actuator names"):
        DeployProfile.from_dict(payload, "deploy/profile.yaml")

    payload = _profile_payload()
    payload["composition"]["neural"][0]["mappings"][1]["index"] = 0
    with pytest.raises(ValueError, match="duplicate accepted-model indices"):
        DeployProfile.from_dict(payload, "deploy/profile.yaml")

    payload = _profile_payload()
    payload["composition"]["typo"] = []
    with pytest.raises(ValueError, match="unknown keys: typo"):
        DeployProfile.from_dict(payload, "deploy/profile.yaml")


def test_frozen_deploy_state_carries_resolved_runtime_contract() -> None:
    state = FrozenDeployState.from_dict(_frozen_payload(), "deploy/default.yaml")

    neural = state.composition.neural[0]
    assert neural.accepted_model_revision == 3
    assert neural.artifact.metadata_sha256 == HASH_C
    assert neural.mappings[0].effort_limits == {"elbow": 25.0}
    assert state.composition.pd[0].armature == {"shoulder": 0.006}
    assert state.ownership == {"elbow": "right_arm", "shoulder": "left_arm"}


def test_frozen_pd_armature_is_optional() -> None:
    payload = _frozen_payload()
    del payload["composition"]["pd"][0]["armature"]

    state = FrozenDeployState.from_dict(payload, "deploy/default.yaml")

    assert state.composition.pd[0].armature is None


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda data: data["ownership"].update(elbow="left_arm"),
            "ownership must exactly match resolved composition outputs",
        ),
        (
            lambda data: data["composition"]["neural"][0].update(sample_dt_s=0.004),
            "sample_dt_s must equal 1 / control_freq",
        ),
        (
            lambda data: data["composition"]["neural"][0]["mappings"][0].update(
                clamping="total_output"
            ),
            "requires clamping 'residual_output'",
        ),
        (
            lambda data: data["composition"]["neural"][0]["mappings"][0].update(
                pd_baseline=None
            ),
            "residual targets require a mapping PD baseline",
        ),
        (
            lambda data: data["composition"]["neural"][0]["mappings"][0]["pd_baseline"][
                "kp"
            ].pop("context"),
            "PD baseline kp and kd keys must match",
        ),
        (
            lambda data: data["composition"]["pd"][0]["kp"].update(shoulder=math.nan),
            "finite number",
        ),
        (
            lambda data: data["composition"]["neural"][0]["mappings"][0][
                "pd_baseline"
            ]["kp"].update(elbow=-1.0),
            "PD baseline kp must contain non-negative values",
        ),
        (
            lambda data: data["composition"]["neural"][0]["mappings"][0][
                "friction_baseline"
            ]["dry"].update(elbow=-1.0),
            "friction baseline dry must contain non-negative values",
        ),
        (
            lambda data: data["composition"]["neural"][0]["mappings"][0][
                "armature"
            ].update(elbow=-1.0),
            "armature must contain non-negative values",
        ),
        (
            lambda data: data["composition"]["pd"][0]["kd"].update(shoulder=-1.0),
            "kd must contain non-negative values",
        ),
        (
            lambda data: data["composition"]["pd"][0]["dry_friction"].update(
                shoulder=-1.0
            ),
            "dry_friction must contain non-negative values",
        ),
        (
            lambda data: data["composition"]["pd"][0]["armature"].update(
                shoulder=-1.0
            ),
            "armature must contain non-negative values",
        ),
        (
            lambda data: data["source"].update(train_state_sha256="bad"),
            "lowercase SHA-256",
        ),
    ],
)
def test_frozen_deploy_state_rejects_inconsistent_or_nonfinite_contracts(
    mutate,
    message: str,
) -> None:
    payload = copy.deepcopy(_frozen_payload())
    mutate(payload)

    with pytest.raises(ValueError, match=message):
        FrozenDeployState.from_dict(payload, "deploy/default.yaml")
