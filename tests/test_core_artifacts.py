from __future__ import annotations

import fcntl
import os
from pathlib import Path

import pytest

from anchor_bench.core.artifacts import (
    RunVerificationError,
    atomic_write_yaml,
    generated_file,
    load_manifest,
    run_slot,
    sibling_lock,
    sibling_lock_path,
    verify_run_directory,
    write_manifest,
)
from anchor_bench.core.hashing import canonical_sha256
from anchor_bench.core.manifest_schema import RunManifest, build_profile_snapshot
from anchor_bench.core.profiles import load_yaml_mapping


def _profile():
    return build_profile_snapshot(
        name="default",
        path="configs/robots/g1/sysid/profiles/default.yaml",
        raw_yaml="schema_version: 1\n",
        content={"schema_version": 1},
    )


def _run_input() -> dict[str, object]:
    """Return one complete duplicated manifest identity envelope."""

    profile = _profile()
    return {
        "phase": "sysid",
        "robot": "g1",
        "profile": {
            "name": profile.name,
            "content": profile.content,
            "content_sha256": profile.content_sha256,
        },
        "resolved_config": {"schema_version": 1, "seed": 1},
        "seed": 1,
        "replicate": None,
        "fingerprints": {},
    }


def _finish(run_dir: Path, run_input: dict[str, object]) -> RunManifest:
    payload = run_dir / "metrics.yaml"
    payload.write_text("loss: 0.5\n", encoding="utf-8")
    manifest = RunManifest.build(
        phase="sysid",
        robot="g1",
        profile=_profile(),
        resolved_config={"schema_version": 1, "seed": 1},
        run_input=run_input,
        seed=1,
        generated_files=[generated_file(run_dir, "metrics.yaml", "metrics")],
    )
    write_manifest(run_dir, manifest)
    return manifest


def test_fresh_run_installs_and_second_identical_request_is_noop(tmp_path: Path) -> None:
    final = tmp_path / "runs" / "abc"
    run_input = _run_input()
    input_hash = canonical_sha256(run_input)
    with run_slot(final, input_hash, containment_root=tmp_path) as slot:
        assert slot.noop is False
        assert slot.tmp_dir is not None
        manifest = _finish(slot.tmp_dir, run_input)

    assert (final / "manifest.yaml").is_file()
    assert (final / "metrics.yaml").is_file()
    with run_slot(final, input_hash, containment_root=tmp_path) as slot:
        assert slot.noop is True
        assert slot.tmp_dir is None
        assert slot.manifest == manifest


def test_incomplete_corrupt_and_identity_mismatched_outputs_are_hard_errors(tmp_path: Path) -> None:
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    (incomplete / "partial.txt").write_text("partial", encoding="utf-8")
    with pytest.raises(RunVerificationError, match="missing regular manifest.yaml"):
        with run_slot(incomplete, containment_root=tmp_path):
            pass

    final = tmp_path / "complete"
    run_input = _run_input()
    with run_slot(
        final,
        canonical_sha256(run_input),
        containment_root=tmp_path,
    ) as slot:
        assert slot.tmp_dir is not None
        _finish(slot.tmp_dir, run_input)

    (final / "metrics.yaml").write_text("loss: 999\n", encoding="utf-8")
    with pytest.raises(RunVerificationError, match="hash mismatch"):
        verify_run_directory(final)

    (final / "metrics.yaml").write_text("loss: 0.5\n", encoding="utf-8")
    with pytest.raises(RunVerificationError, match="run input identity mismatch"):
        verify_run_directory(final, "0" * 64)


def test_unrecorded_file_and_symlink_are_rejected(tmp_path: Path) -> None:
    final = tmp_path / "run"
    run_input = _run_input()
    with run_slot(
        final,
        canonical_sha256(run_input),
        containment_root=tmp_path,
    ) as slot:
        assert slot.tmp_dir is not None
        _finish(slot.tmp_dir, run_input)

    (final / "extra.txt").write_text("not authenticated", encoding="utf-8")
    with pytest.raises(RunVerificationError, match="unrecorded generated files"):
        verify_run_directory(final)
    (final / "extra.txt").unlink()

    outside = tmp_path / "outside.txt"
    outside.write_text("loss: 0.5\n", encoding="utf-8")
    (final / "metrics.yaml").unlink()
    (final / "metrics.yaml").symlink_to(outside)
    with pytest.raises(RunVerificationError, match="symlink"):
        verify_run_directory(final)


def test_manifest_is_written_last_and_cannot_be_replaced(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    run_input = _run_input()
    (run_dir / "metrics.yaml").write_text("loss: 0.5\n", encoding="utf-8")
    manifest = RunManifest.build(
        phase="sysid",
        robot="g1",
        profile=_profile(),
        resolved_config={"schema_version": 1, "seed": 1},
        seed=1,
        run_input=run_input,
        generated_files=[generated_file(run_dir, "metrics.yaml", "metrics")],
    )
    assert not (run_dir / "manifest.yaml").exists()
    written_hash = write_manifest(run_dir, manifest)
    loaded, loaded_hash = load_manifest(run_dir)
    assert loaded == manifest
    assert loaded_hash == written_hash == manifest.run_manifest_sha256
    with pytest.raises(RunVerificationError, match="already exists"):
        write_manifest(run_dir, manifest)


def test_manifest_formatting_does_not_change_canonical_hash(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest = _finish(run_dir, _run_input())
    original_hash = load_manifest(run_dir)[1]
    document = load_yaml_mapping(run_dir / "manifest.yaml")
    (run_dir / "manifest.yaml").write_text(
        "# a nonsemantic YAML comment\n" + __import__("yaml").safe_dump(document, sort_keys=True),
        encoding="utf-8",
    )
    loaded, reformatted_hash = load_manifest(run_dir)
    assert loaded == manifest
    assert reformatted_hash == original_hash


def test_failed_run_cleans_temp_and_never_installs_final(tmp_path: Path) -> None:
    final = tmp_path / "runs" / "abc"
    with pytest.raises(RuntimeError, match="boom"):
        with run_slot(final, containment_root=tmp_path) as slot:
            assert slot.tmp_dir is not None
            (slot.tmp_dir / "partial.csv").write_text("x", encoding="utf-8")
            raise RuntimeError("boom")
    assert not final.exists()
    assert not any(path.is_dir() for path in final.parent.iterdir())


def test_run_slot_rejects_escape_and_symlinked_parent_before_writing(
    tmp_path: Path,
) -> None:
    """Reject escaped and symlink-routed targets before creating locks or temps."""

    root = tmp_path / "repository"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()

    escaped = outside / "escaped-run"
    with pytest.raises(ValueError, match="escapes containment root"):
        with run_slot(escaped, containment_root=root):
            pass
    assert not escaped.exists()
    assert not escaped.with_name("escaped-run.lock").exists()

    outputs = root / "outputs"
    outputs.symlink_to(outside, target_is_directory=True)
    routed = outputs / "sysid" / "g1" / "run"
    with pytest.raises(ValueError, match="symlinked parent"):
        with run_slot(routed, containment_root=root):
            pass
    assert list(outside.iterdir()) == []


def test_stable_sibling_lock_is_exclusive_and_survives_target_replacement(tmp_path: Path) -> None:
    target = tmp_path / "default.yaml"
    target.write_text("revision: 1\n", encoding="utf-8")
    lock_path = sibling_lock_path(target)
    with sibling_lock(target):
        descriptor = os.open(lock_path, os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(descriptor)
        replacement = tmp_path / "replacement.yaml"
        replacement.write_text("revision: 2\n", encoding="utf-8")
        os.replace(replacement, target)
        assert lock_path.is_file()


def test_atomic_yaml_failure_preserves_existing_state(tmp_path: Path) -> None:
    target = tmp_path / "default.yaml"
    atomic_write_yaml(target, {"revision": 1})
    with pytest.raises(ValueError):
        atomic_write_yaml(target, {"bad": object()})
    assert load_yaml_mapping(target) == {"revision": 1}
    assert not any(".tmp-" in path.name for path in tmp_path.iterdir())
