from __future__ import annotations

from copy import deepcopy
import math

import pytest

from anchor_bench.core.hashing import UnsafePathError, canonical_sha256
from anchor_bench.core.manifest_schema import (
    GeneratedFile,
    RunManifest,
    build_profile_snapshot,
    manifest_sha256,
)


def _profile(raw_yaml: str = "schema_version: 1\nvalue: 2\n"):
    return build_profile_snapshot(
        name="default",
        path="configs/robots/g1/sysid/profiles/default.yaml",
        raw_yaml=raw_yaml,
        content={"schema_version": 1, "value": 2},
    )


def _manifest(*, raw_yaml: str = "schema_version: 1\nvalue: 2\n") -> RunManifest:
    profile = _profile(raw_yaml)
    resolved_config = {"schema_version": 1, "seed": 42}
    fingerprints = {"dataset": {"tree_sha256": "a" * 64}}
    return RunManifest.build(
        phase="sysid",
        robot="g1",
        profile=profile,
        resolved_config=resolved_config,
        cli_overrides={"seed": 42},
        seed=42,
        replicate=None,
        run_input={
            "phase": "sysid",
            "robot": "g1",
            "profile": {
                "name": profile.name,
                "content": profile.content,
                "content_sha256": profile.content_sha256,
            },
            "resolved_config": resolved_config,
            "seed": 42,
            "replicate": None,
            "fingerprints": fingerprints,
        },
        fingerprints=fingerprints,
        upstream={},
        generated_files=[
            GeneratedFile(path="z.yaml", role="metrics", sha256="b" * 64),
            GeneratedFile(path="a.yaml", role="resolved_config", sha256="c" * 64),
        ],
    )


def test_builder_computes_distinct_input_and_manifest_hashes() -> None:
    manifest = _manifest()
    assert manifest.run_input_sha256 == canonical_sha256(manifest.run_input)
    assert manifest.run_manifest_sha256 == manifest_sha256(manifest)
    assert manifest.run_input_sha256 != manifest.run_manifest_sha256
    assert [entry.path for entry in manifest.generated_files] == ["a.yaml", "z.yaml"]


def test_profile_formatting_is_nonsemantic_but_remains_manifest_provenance() -> None:
    first = _manifest()
    second = _manifest(raw_yaml="# comment\nvalue: 2\nschema_version: 1\n")
    assert first.profile.content_sha256 == second.profile.content_sha256
    assert first.profile.raw_sha256 != second.profile.raw_sha256
    assert first.run_input_sha256 == second.run_input_sha256
    assert first.run_manifest_sha256 != second.run_manifest_sha256


def test_input_resolved_config_and_self_hash_mismatches_are_rejected() -> None:
    manifest = _manifest()
    data = manifest.model_dump(mode="json")

    bad_input = dict(data)
    bad_input["run_input_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="run_input_sha256 mismatch"):
        RunManifest.from_dict(bad_input, "manifest")

    bad_resolved = dict(data)
    bad_resolved["resolved_config_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="resolved_config_sha256 mismatch"):
        RunManifest.from_dict(bad_resolved, "manifest")

    bad_self = dict(data)
    bad_self["run_manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="run_manifest_sha256 mismatch"):
        RunManifest.from_dict(bad_self, "manifest")


@pytest.mark.parametrize(
    "path",
    ["../escape", "/absolute", "a//b", "./a", "a\\b", "manifest.yaml"],
)
def test_generated_file_paths_must_be_safe_run_relative_content(path: str) -> None:
    with pytest.raises((UnsafePathError, ValueError)):
        GeneratedFile(path=path, role="metrics", sha256="a" * 64)


def test_duplicate_generated_paths_and_unknown_fields_are_rejected() -> None:
    entry = GeneratedFile(path="metrics.yaml", role="metrics", sha256="a" * 64)
    profile = _profile()
    run_input = {
        "phase": "sysid",
        "robot": "g1",
        "profile": {
            "name": profile.name,
            "content": profile.content,
            "content_sha256": profile.content_sha256,
        },
        "resolved_config": {},
        "seed": None,
        "replicate": None,
        "fingerprints": {},
    }
    with pytest.raises(ValueError, match="duplicate paths"):
        RunManifest.build(
            phase="sysid",
            robot="g1",
            profile=profile,
            resolved_config={},
            run_input=run_input,
            generated_files=[entry, entry],
        )

    data = _manifest().model_dump(mode="json")
    data["unknown"] = True
    with pytest.raises(ValueError, match="unknown keys: unknown"):
        RunManifest.from_dict(data, "manifest")


def test_nonfinite_phase_specific_mapping_is_rejected() -> None:
    profile = _profile()
    with pytest.raises(ValueError, match="non-finite"):
        RunManifest.build(
            phase="train",
            robot="g1",
            profile=profile,
            resolved_config={},
            run_input={
                "phase": "train",
                "robot": "g1",
                "profile": {
                    "name": profile.name,
                    "content": profile.content,
                    "content_sha256": profile.content_sha256,
                },
                "resolved_config": {},
                "seed": None,
                "replicate": None,
                "fingerprints": {"loss": math.nan},
            },
            fingerprints={"loss": math.nan},
            generated_files=[],
        )


def _rehash_manifest_document(document: dict) -> dict:
    """Recompute a tampered document's canonical manifest self-hash."""

    result = deepcopy(document)
    payload = {
        key: value
        for key, value in result.items()
        if key != "run_manifest_sha256" and value is not None
    }
    result["run_manifest_sha256"] = canonical_sha256(payload)
    return result


def test_self_consistent_top_level_identity_tampering_is_rejected() -> None:
    """Reject top-level changes even after their local and self hashes are repaired."""

    original = _manifest().model_dump(mode="json", exclude_none=True)
    RunManifest.from_dict(
        _rehash_manifest_document(original),
        "untampered rehashed manifest",
    )
    cases: list[tuple[str, dict]] = []

    phase = deepcopy(original)
    phase["phase"] = "train"
    cases.append(("phase", phase))

    robot = deepcopy(original)
    robot["robot"] = "h1"
    cases.append(("robot", robot))

    profile_name = deepcopy(original)
    profile_name["profile"]["name"] = "other"
    cases.append(("profile.name", profile_name))

    profile_content = deepcopy(original)
    profile_content["profile"]["content"]["value"] = 3
    profile_content["profile"]["content_sha256"] = canonical_sha256(
        profile_content["profile"]["content"]
    )
    cases.append(("profile.content", profile_content))

    resolved = deepcopy(original)
    resolved["resolved_config"]["seed"] = 7
    resolved["resolved_config_sha256"] = canonical_sha256(
        resolved["resolved_config"]
    )
    cases.append(("resolved_config", resolved))

    seed = deepcopy(original)
    seed["seed"] = 7
    cases.append(("seed", seed))

    replicate = deepcopy(original)
    replicate["replicate"] = "copied"
    cases.append(("replicate", replicate))

    fingerprints = deepcopy(original)
    fingerprints["fingerprints"] = {
        "dataset": {"tree_sha256": "d" * 64}
    }
    cases.append(("fingerprints", fingerprints))

    for label, document in cases:
        with pytest.raises(ValueError, match="does not match"):
            RunManifest.from_dict(
                _rehash_manifest_document(document),
                f"tampered {label}",
            )


def test_self_consistent_benchmark_upstream_tampering_is_rejected() -> None:
    """Bind benchmark upstream.source to its duplicated candidate identity."""

    profile = _profile()
    source = {
        "phase": "sysid",
        "robot": "g1",
        "run": "outputs/sysid/g1/default/seed-42/" + "a" * 64,
        "run_input_sha256": "a" * 64,
        "run_manifest_sha256": "b" * 64,
    }
    resolved_config = {"schema_version": 1}
    run_input = {
        "phase": "benchmark",
        "robot": "g1",
        "profile": {
            "name": profile.name,
            "content": profile.content,
            "content_sha256": profile.content_sha256,
        },
        "resolved_config": resolved_config,
        "seed": None,
        "replicate": None,
        "fingerprints": {},
        "candidate": {"source": source},
    }
    manifest = RunManifest.build(
        phase="benchmark",
        robot="g1",
        profile=profile,
        resolved_config=resolved_config,
        run_input=run_input,
        upstream={"source": source},
        generated_files=[],
    )
    document = manifest.model_dump(mode="json", exclude_none=True)
    document["upstream"]["source"]["run_manifest_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="upstream.source"):
        RunManifest.from_dict(
            _rehash_manifest_document(document),
            "tampered benchmark upstream",
        )
