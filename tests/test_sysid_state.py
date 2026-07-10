from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def _parameters(*, stiffness_b: float = 20.0) -> dict:
    return {
        "stiffness": {"joint_a": 10.0, "joint_b": stiffness_b},
        "damping": {"joint_a": 1.0, "joint_b": 2.0},
        "armature": {"joint_a": 0.01, "joint_b": 0.02},
        "dry_friction": {"joint_a": 0.1, "joint_b": 0.2},
        "viscous_friction": {"joint_a": 0.3, "joint_b": 0.4},
    }


def _accepted_payload() -> dict:
    return {
        "schema_version": 1,
        "revision": 0,
        "parameters": _parameters(),
        "provenance": {"history": []},
    }


def _patch_payload() -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "sysid_fitted_patch",
        "robot": "demo",
        "parameters": {"stiffness": {"joint_b": 25.0}},
    }


def _candidate_payload() -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "sysid_state",
        "robot": "demo",
        "parameters": _parameters(stiffness_b=25.0),
        "source": {
            "parent_state_sha256": SHA_A,
            "fitted_patch_sha256": SHA_B,
            "run_input_sha256": SHA_C,
        },
    }


def _models():
    from anchor_bench.sysid.state import (
        AcceptedSysIdState,
        SysIdCandidateState,
        SysIdFittedPatch,
    )

    return (
        AcceptedSysIdState.from_dict(_accepted_payload(), "default.yaml"),
        SysIdFittedPatch.from_dict(_patch_payload(), "fitted_patch.yaml"),
        SysIdCandidateState.from_dict(_candidate_payload(), "sysid_state.yaml"),
    )


def test_revision_zero_accepted_state_and_artifacts_are_strict_models() -> None:
    from pydantic import BaseModel

    accepted, patch, candidate = _models()

    assert isinstance(accepted, BaseModel)
    assert accepted.revision == 0
    assert patch.parameters == {"stiffness": {"joint_b": 25.0}}
    assert candidate.parameters["stiffness"]["joint_b"] == 25.0


def test_accepted_state_records_typed_ordered_provenance() -> None:
    from anchor_bench.sysid.state import AcceptedSysIdState

    payload = _accepted_payload()
    payload["revision"] = 1
    payload["provenance"]["history"] = [
        {
            "revision": 1,
            "source_run": "outputs/sysid/demo/default/seed-1/run",
            "run_input_sha256": SHA_A,
            "run_manifest_sha256": SHA_B,
            "parent_state_sha256": SHA_C,
            "fitted_patch_sha256": SHA_D,
            "candidate_state_sha256": SHA_A,
            "promotion_parent_state_sha256": SHA_B,
            "updated_cells": [{"parameter": "stiffness", "joint": "joint_b"}],
            "benchmarks": [
                {
                    "run": "outputs/benchmark/demo/sysid/default/run",
                    "run_manifest_sha256": SHA_C,
                }
            ],
        }
    ]

    accepted = AcceptedSysIdState.from_dict(payload, "default.yaml")

    assert accepted.revision == 1
    assert accepted.provenance.history[0].updated_cells[0].joint == "joint_b"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda data: data["parameters"].update(
                unknown={"joint_a": 1.0, "joint_b": 1.0}
            ),
            "unsupported SysID parameter",
        ),
        (
            lambda data: data["parameters"]["stiffness"].update(
                joint_c=float("nan")
            ),
            "finite number",
        ),
        (
            lambda data: data["parameters"]["damping"].clear(),
            "must contain at least one joint cell",
        ),
        (
            lambda data: data.update(provenance={"history": [{"revision": 1}]}),
            "revision must equal provenance history length",
        ),
    ],
)
def test_accepted_state_rejects_invalid_complete_state(mutate, message: str) -> None:
    from anchor_bench.sysid.state import AcceptedSysIdState

    data = _accepted_payload()
    mutate(data)

    with pytest.raises(ValueError, match=message):
        AcceptedSysIdState.from_dict(data, "default.yaml")


@pytest.mark.parametrize(
    ("model_name", "payload"),
    [
        ("accepted", _accepted_payload),
        ("patch", _patch_payload),
        ("candidate", _candidate_payload),
    ],
)
def test_sysid_authority_artifacts_reject_negative_physical_values(
    model_name: str,
    payload,
) -> None:
    from anchor_bench.sysid.state import (
        AcceptedSysIdState,
        SysIdCandidateState,
        SysIdFittedPatch,
    )

    models = {
        "accepted": AcceptedSysIdState,
        "patch": SysIdFittedPatch,
        "candidate": SysIdCandidateState,
    }
    data = payload()
    data["parameters"]["stiffness"]["joint_b"] = -1.0

    with pytest.raises(ValueError, match="must be non-negative"):
        models[model_name].from_dict(data, f"{model_name}.yaml")


def test_complete_state_allows_parameter_specific_joint_subsets() -> None:
    from anchor_bench.sysid.state import AcceptedSysIdState, require_parameter_cells

    data = _accepted_payload()
    data["parameters"]["armature"].pop("joint_b")
    accepted = AcceptedSysIdState.from_dict(data, "default.yaml")

    require_parameter_cells(accepted, ["armature"], ["joint_a"])
    with pytest.raises(ValueError, match="missing required parameter cells: armature/joint_b"):
        require_parameter_cells(accepted, ["armature"], ["joint_b"])


def test_sparse_patch_application_preserves_unrelated_cells_and_parent() -> None:
    from anchor_bench.sysid.state import apply_fitted_patch

    accepted, patch, _ = _models()
    before = accepted.to_dict()

    result = apply_fitted_patch(accepted, patch)

    assert result["stiffness"] == {"joint_a": 10.0, "joint_b": 25.0}
    assert result["damping"] == accepted.parameters["damping"]
    assert accepted.to_dict() == before


def test_sparse_patch_rejects_empty_and_unknown_cells() -> None:
    from anchor_bench.sysid.state import SysIdFittedPatch, apply_fitted_patch

    accepted, _, _ = _models()
    empty = _patch_payload()
    empty["parameters"] = {}
    with pytest.raises(ValueError, match="must contain at least one parameter cell"):
        SysIdFittedPatch.from_dict(empty, "fitted_patch.yaml")

    unknown = _patch_payload()
    unknown["parameters"] = {"stiffness": {"joint_c": 25.0}}
    patch = SysIdFittedPatch.from_dict(unknown, "fitted_patch.yaml")
    with pytest.raises(ValueError, match="unknown parameter cell stiffness/joint_c"):
        apply_fitted_patch(accepted, patch)


def test_candidate_must_equal_parent_plus_patch_exactly() -> None:
    from anchor_bench.sysid.state import verify_candidate_state

    accepted, patch, candidate = _models()

    assert verify_candidate_state(accepted, patch, candidate) is candidate
    bad = candidate.model_copy(
        update={
            "parameters": {
                **candidate.parameters,
                "stiffness": {"joint_a": 11.0, "joint_b": 25.0},
            }
        }
    )
    with pytest.raises(
        ValueError,
        match="does not equal parent state plus fitted patch.*stiffness/joint_a",
    ):
        verify_candidate_state(accepted, patch, bad)


def test_build_candidate_state_applies_patch_and_validates_source_hashes() -> None:
    from anchor_bench.sysid.state import build_candidate_state

    accepted, patch, _ = _models()
    candidate = build_candidate_state(
        accepted,
        patch,
        {
            "parent_state_sha256": SHA_A,
            "fitted_patch_sha256": SHA_B,
            "run_input_sha256": SHA_C,
        },
    )

    assert candidate.parameters["stiffness"]["joint_b"] == 25.0
    with pytest.raises(ValueError, match="lowercase SHA-256 digest"):
        build_candidate_state(
            accepted,
            patch,
            {
                "parent_state_sha256": "not-a-hash",
                "fitted_patch_sha256": SHA_B,
                "run_input_sha256": SHA_C,
            },
        )


def test_candidate_robot_must_match_patch_and_candidate_parent() -> None:
    from anchor_bench.sysid.state import verify_candidate_state

    accepted, patch, candidate = _models()
    wrong_patch = patch.model_copy(update={"robot": "other"})
    with pytest.raises(ValueError, match="robot mismatch"):
        verify_candidate_state(accepted, wrong_patch, candidate)

    with pytest.raises(ValueError, match="robot mismatch"):
        verify_candidate_state(
            candidate.model_copy(update={"robot": "other"}),
            patch,
            candidate,
        )


def test_canonical_joint_maps_reject_aliases_and_preserve_robot_order() -> None:
    from anchor_bench.sysid.state import canonicalize_parameter_maps

    ordered = canonicalize_parameter_maps(_parameters(), ["joint_b", "joint_a"])
    assert list(ordered) == [
        "stiffness",
        "damping",
        "armature",
        "dry_friction",
        "viscous_friction",
    ]
    assert list(ordered["stiffness"]) == ["joint_b", "joint_a"]

    aliased = _parameters()
    aliased["stiffness"]["joint_a_alias"] = aliased["stiffness"].pop("joint_a")
    with pytest.raises(ValueError, match="non-canonical or unknown joints.*joint_a_alias"):
        canonicalize_parameter_maps(aliased, ["joint_a", "joint_b"])

    nonfinite = _parameters()
    nonfinite["stiffness"]["joint_a"] = float("inf")
    with pytest.raises(ValueError, match="stiffness/joint_a must be finite"):
        canonicalize_parameter_maps(nonfinite, ["joint_a", "joint_b"])


def test_required_parameter_cells_are_checked_for_initial_state() -> None:
    from anchor_bench.sysid.state import require_parameter_cells

    accepted, _, _ = _models()

    require_parameter_cells(accepted, ["stiffness", "damping"], ["joint_b"])
    with pytest.raises(
        ValueError,
        match="missing required parameter cells: rotor_inertia/joint_b",
    ):
        require_parameter_cells(accepted, ["rotor_inertia"], ["joint_b"])


def test_verified_candidate_loader_uses_manifest_roles_and_source_hashes(
    tmp_path: Path,
) -> None:
    from anchor_bench.sysid.state import load_verified_sysid_candidate

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "parent_state.yaml").write_text(
        yaml.safe_dump(_accepted_payload()), encoding="utf-8"
    )
    (run_dir / "fitted_patch.yaml").write_text(
        yaml.safe_dump(_patch_payload()), encoding="utf-8"
    )
    (run_dir / "sysid_state.yaml").write_text(
        yaml.safe_dump(_candidate_payload()), encoding="utf-8"
    )
    manifest = SimpleNamespace(
        phase="sysid",
        robot="demo",
        run_input_sha256=SHA_C,
        generated_files=[
            SimpleNamespace(path="parent_state.yaml", role="parent_state", sha256=SHA_A),
            SimpleNamespace(path="fitted_patch.yaml", role="fitted_patch", sha256=SHA_B),
            SimpleNamespace(path="sysid_state.yaml", role="sysid_state", sha256=SHA_D),
        ],
    )

    loaded = load_verified_sysid_candidate(run_dir, manifest, expected_robot="demo")

    assert loaded.run_dir == run_dir
    assert loaded.manifest is manifest
    assert loaded.state.parameters["stiffness"]["joint_b"] == 25.0


def test_verified_candidate_loader_rejects_bad_manifest_relationships(
    tmp_path: Path,
) -> None:
    from anchor_bench.sysid.state import load_verified_sysid_candidate

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    for name, payload in (
        ("parent_state.yaml", _accepted_payload()),
        ("fitted_patch.yaml", _patch_payload()),
        ("sysid_state.yaml", _candidate_payload()),
    ):
        (run_dir / name).write_text(yaml.safe_dump(payload), encoding="utf-8")
    entries = [
        SimpleNamespace(path="parent_state.yaml", role="parent_state", sha256=SHA_A),
        SimpleNamespace(path="fitted_patch.yaml", role="fitted_patch", sha256=SHA_B),
        SimpleNamespace(path="sysid_state.yaml", role="sysid_state", sha256=SHA_D),
    ]

    with pytest.raises(ValueError, match="expected a sysid run manifest"):
        load_verified_sysid_candidate(
            run_dir,
            SimpleNamespace(
                phase="train",
                robot="demo",
                run_input_sha256=SHA_C,
                generated_files=entries,
            ),
        )
    with pytest.raises(
        ValueError,
        match="missing required generated-file role: fitted_patch",
    ):
        load_verified_sysid_candidate(
            run_dir,
            SimpleNamespace(
                phase="sysid",
                robot="demo",
                run_input_sha256=SHA_C,
                generated_files=entries[:1] + entries[2:],
            ),
        )
    bad_state = _candidate_payload()
    bad_state["source"]["parent_state_sha256"] = "e" * 64
    (run_dir / "sysid_state.yaml").write_text(
        yaml.safe_dump(bad_state), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="parent_state_sha256 does not match"):
        load_verified_sysid_candidate(
            run_dir,
            SimpleNamespace(
                phase="sysid",
                robot="demo",
                run_input_sha256=SHA_C,
                generated_files=entries,
            ),
        )
