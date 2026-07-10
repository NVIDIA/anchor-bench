"""Host-safe tests for robot.yaml and runtime asset-package identities."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from anchor_bench.robots.fingerprints import (
    load_robot_identity_fingerprint,
    preview_asset_fingerprint,
    runtime_asset_fingerprint,
    verify_runtime_asset_fingerprint,
)
from anchor_bench.robots.registry import get_robot_from_root
from anchor_bench.runtime.assets import resolve_runtime_asset


def _robot_repository(root: Path) -> tuple[Path, Path]:
    """Create one declarative robot and a two-file local asset package."""

    robot_path = root / "configs/robots/demo/robot.yaml"
    robot_path.parent.mkdir(parents=True)
    robot_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "demo",
                "display_name": "Demo",
                "fixed_base": True,
                "asset_path": "assets/robots/demo/manifest.yaml",
                "runtime_usd": "demo.usd",
                "joint_names": ["joint_a", "joint_b"],
                "joint_aliases": {"A": "joint_a"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    manifest = root / "assets/robots/demo/manifest.yaml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("name: demo\n", encoding="utf-8")
    runtime_usd = manifest.parent / "files/demo.usd"
    runtime_usd.parent.mkdir()
    runtime_usd.write_bytes(b"usd")
    (runtime_usd.parent / "mesh.bin").write_bytes(b"mesh")
    return robot_path, runtime_usd


def test_robot_identity_tracks_raw_and_semantic_changes(tmp_path: Path) -> None:
    """Distinguish formatting changes from semantic robot contract changes."""

    robot_path, _asset = _robot_repository(tmp_path)
    _spec, first = load_robot_identity_fingerprint(
        robot_path,
        expected_name="demo",
        repository_root=tmp_path,
    )
    document = yaml.safe_load(robot_path.read_text(encoding="utf-8"))
    robot_path.write_text(yaml.safe_dump(document, sort_keys=True), encoding="utf-8")
    _spec, reformatted = load_robot_identity_fingerprint(
        robot_path,
        expected_name="demo",
        repository_root=tmp_path,
    )
    assert reformatted["raw_sha256"] != first["raw_sha256"]
    assert reformatted["canonical_sha256"] == first["canonical_sha256"]

    document["fixed_base"] = False
    robot_path.write_text(yaml.safe_dump(document, sort_keys=True), encoding="utf-8")
    _spec, changed = load_robot_identity_fingerprint(
        robot_path,
        expected_name="demo",
        repository_root=tmp_path,
    )
    assert changed["canonical_sha256"] != first["canonical_sha256"]


def test_asset_identity_authenticates_non_root_package_files(tmp_path: Path) -> None:
    """Make referenced-package changes invalidate identity even when USD is stable."""

    _robot_path, asset = _robot_repository(tmp_path)
    first = runtime_asset_fingerprint(asset, runtime_usd="demo.usd")
    assert first["root_usd_sha256"] == first["asset_sha256"]
    assert set(first["package_files"]) == {"demo.usd", "mesh.bin"}

    (asset.parent / "mesh.bin").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="asset package changed after resolution"):
        verify_runtime_asset_fingerprint(first, asset, runtime_usd="demo.usd")


def test_asset_preview_has_no_authoritative_asset_digest(tmp_path: Path) -> None:
    """Label manifest-only dry-run material solely as a preview identity."""

    _robot_path, asset = _robot_repository(tmp_path)
    manifest = asset.parents[1] / "manifest.yaml"
    preview = preview_asset_fingerprint(
        manifest,
        runtime_usd="demo.usd",
        repository_root=tmp_path,
    )
    assert preview["kind"] == "runtime_asset_preview"
    assert "asset_sha256" not in preview
    assert "root_usd_sha256" not in preview
    assert len(preview["preview_sha256"]) == 64


def test_external_asset_resolution_prefers_robot_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prefer a robot-scoped private package over a shared external root."""

    _robot_path, _asset = _robot_repository(tmp_path)
    external = tmp_path / "private"
    external.mkdir()
    (external / "demo.usd").write_bytes(b"shared")
    robot_asset = external / "demo/demo.usd"
    robot_asset.parent.mkdir()
    robot_asset.write_bytes(b"scoped")
    monkeypatch.setenv("ANCHOR_BENCH_ASSET_DIR", str(external))

    bundle = get_robot_from_root("demo", tmp_path)
    assert resolve_runtime_asset(bundle) == robot_asset
