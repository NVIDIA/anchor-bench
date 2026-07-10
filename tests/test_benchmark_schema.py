from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from anchor_bench.benchmark.schema import BenchmarkCandidateReference, BenchmarkProfile


HASH_A = "a" * 64
HASH_B = "b" * 64


def _profile_payload() -> dict:
    return {
        "schema_version": 1,
        "data": "inputs/datasets/demo/held_out",
        "motions": "*",
        "simulation": {
            "physics_freq": 500,
            "control_freq": 500,
            "render_freq": 500,
            "integrator": "implicitfast",
            "solver_iterations": 10,
            "line_search_iterations": 10,
        },
        "num_envs": 16,
        "buffer_time": 2.0,
        "sync_initial_state": True,
        "signals": ["position", "velocity"],
        "output_root": "outputs/benchmark",
    }


def test_benchmark_profile_is_complete_strict_and_immutable() -> None:
    profile = BenchmarkProfile.from_dict(_profile_payload(), "benchmark/profile.yaml")

    assert profile.motions == "*"
    assert profile.signals == ("position", "velocity")
    assert profile.simulation.control_freq == 500
    with pytest.raises(ValidationError):
        profile.num_envs = 8

    payload = _profile_payload()
    payload["simulation"]["typo"] = 1
    with pytest.raises(ValueError, match="unknown keys: typo"):
        BenchmarkProfile.from_dict(payload, "benchmark/profile.yaml")


def test_benchmark_profile_accepts_unique_explicit_motion_names() -> None:
    payload = _profile_payload()
    payload["motions"] = ["slow_reach", "fast_reach"]

    profile = BenchmarkProfile.from_dict(payload, "benchmark/profile.yaml")

    assert profile.motions == ("slow_reach", "fast_reach")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("signals", ["position", "torque"], "torque benchmark signals are not supported"),
        ("signals", ["effort"], "unsupported benchmark signal"),
        ("signals", ["position", "position"], "duplicate signals"),
        ("motions", ["reach", "reach"], "duplicate motions"),
        ("buffer_time", math.inf, "finite number"),
        ("data", "../escape", "invalid relative POSIX path"),
    ],
)
def test_benchmark_profile_rejects_ambiguous_or_nonfinite_values(
    field: str,
    value: object,
    message: str,
) -> None:
    payload = _profile_payload()
    payload[field] = value

    with pytest.raises(ValueError, match=message):
        BenchmarkProfile.from_dict(payload, "benchmark/profile.yaml")


def test_benchmark_profile_requires_complete_simulation() -> None:
    payload = _profile_payload()
    del payload["simulation"]["line_search_iterations"]

    with pytest.raises(ValueError, match="missing required key 'line_search_iterations'"):
        BenchmarkProfile.from_dict(payload, "benchmark/profile.yaml")


def test_candidate_reference_names_an_exact_immutable_run_directory() -> None:
    candidate = BenchmarkCandidateReference.from_dict(
        {
            "phase": "train",
            "robot": "demo",
            "run": f"outputs/train/demo/right_arm/seed-42/{HASH_A}",
            "run_input_sha256": HASH_A,
            "run_manifest_sha256": HASH_B,
        },
        "candidate",
    )

    assert candidate.phase == "train"
    assert candidate.run.endswith(candidate.run_input_sha256)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"phase": "sysid"}, "run must be an immutable directory under outputs/sysid"),
        ({"run_input_sha256": HASH_B}, "run directory digest must equal run_input_sha256"),
        ({"run": f"outputs/train/demo/right_arm/seed-01/{HASH_A}"}, "canonical seed"),
        ({"run": f"outputs/train/demo/../other/seed-42/{HASH_A}"}, "invalid relative POSIX path"),
        ({"run_manifest_sha256": "not-a-hash"}, "lowercase SHA-256"),
    ],
)
def test_candidate_reference_rejects_mutable_or_mismatched_identity(
    updates: dict[str, str],
    message: str,
) -> None:
    payload = {
        "phase": "train",
        "robot": "demo",
        "run": f"outputs/train/demo/right_arm/seed-42/{HASH_A}",
        "run_input_sha256": HASH_A,
        "run_manifest_sha256": HASH_B,
    }
    payload.update(updates)

    with pytest.raises(ValueError, match=message):
        BenchmarkCandidateReference.from_dict(payload, "candidate")
