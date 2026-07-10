"""Tests for the host-safe robot identity registry."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from anchor_bench.robots.registry import get_robot, list_robots, validate_robot


def _write_identity(root: Path, name: str, *, document_name: str | None = None) -> Path:
    """Write one minimal identity and its tracked asset manifest."""

    config_root = root / "configs/robots"
    robot_dir = config_root / name
    robot_dir.mkdir(parents=True)
    asset_dir = root / "assets/robots" / name
    asset_dir.mkdir(parents=True)
    (asset_dir / "manifest.yaml").write_text(
        f"name: {name}\nexpected_assets: [{name}.usd]\n",
        encoding="utf-8",
    )
    (robot_dir / "robot.yaml").write_text(
        "\n".join(
            [
                "schema_version: 1",
                f"name: {document_name or name}",
                f"display_name: {name.title()}",
                "fixed_base: true",
                f"asset_path: assets/robots/{document_name or name}/manifest.yaml",
                f"runtime_usd: {name}.usd",
                "joint_names: [elbow]",
                "joint_aliases: {}",
                "ee_frame: elbow",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return config_root


def test_registry_loads_all_reference_robot_identities() -> None:
    """Keep the checked-in robot set deliberate and identity-only."""

    assert list_robots() == ["g1", "h1", "so101", "teststand", "ur10e"]
    for name in list_robots():
        bundle = get_robot(name)
        assert bundle.spec.name == name
        assert bundle.paths.asset.is_file()
        assert validate_robot(name) == []
        assert not hasattr(bundle, "run_config")
        assert not hasattr(bundle.paths, "actuators")
        assert not hasattr(bundle.paths, "bounds")


def test_registry_does_not_require_external_runtime_usd_bytes() -> None:
    """Allow host-only inspection when the manifest exists but the USD does not."""

    bundle = get_robot("g1")
    assert bundle.spec.runtime_usd == "g1.usd"
    assert not (bundle.paths.asset.parent / "files" / bundle.spec.runtime_usd).exists()


def test_registry_rejects_directory_identity_mismatch(tmp_path: Path) -> None:
    """Bind the identity name to its configuration directory."""

    config_root = _write_identity(tmp_path, "badbot", document_name="other")
    with (
        patch("anchor_bench.robots.registry.robot_config_root", return_value=config_root),
        patch("anchor_bench.robots.registry.repo_root", return_value=tmp_path),
    ):
        with pytest.raises(ValueError, match="must match its configuration directory"):
            get_robot("badbot")


def test_registry_reports_missing_asset_manifest_without_resolving_usd(tmp_path: Path) -> None:
    """Require the tracked manifest while leaving external asset resolution to runtime."""

    config_root = _write_identity(tmp_path, "demo")
    (tmp_path / "assets/robots/demo/manifest.yaml").unlink()
    with (
        patch("anchor_bench.robots.registry.robot_config_root", return_value=config_root),
        patch("anchor_bench.robots.registry.repo_root", return_value=tmp_path),
    ):
        assert "missing asset manifest" in "\n".join(validate_robot("demo"))
        with pytest.raises(ValueError, match="missing asset manifest"):
            get_robot("demo")
