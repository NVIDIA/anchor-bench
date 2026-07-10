from __future__ import annotations

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
    return RunManifest.build(
        phase="sysid",
        robot="g1",
        profile=_profile(raw_yaml),
        resolved_config={"schema_version": 1, "seed": 42},
        cli_overrides={"seed": 42},
        seed=42,
        replicate=None,
        run_input={"robot": "g1", "profile": "default", "seed": 42},
        fingerprints={"dataset": {"tree_sha256": "a" * 64}},
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
    with pytest.raises(ValueError, match="duplicate paths"):
        RunManifest.build(
            phase="sysid",
            robot="g1",
            profile=_profile(),
            resolved_config={},
            run_input={},
            generated_files=[entry, entry],
        )

    data = _manifest().model_dump(mode="json")
    data["unknown"] = True
    with pytest.raises(ValueError, match="unknown keys: unknown"):
        RunManifest.from_dict(data, "manifest")


def test_nonfinite_phase_specific_mapping_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        RunManifest.build(
            phase="train",
            robot="g1",
            profile=_profile(),
            resolved_config={},
            run_input={},
            fingerprints={"loss": math.nan},
            generated_files=[],
        )
