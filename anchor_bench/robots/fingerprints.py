"""Host-safe robot.yaml and runtime asset-package identities."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from anchor_bench.core.hashing import (
    canonical_object,
    canonical_sha256,
    repo_relative_posix,
    require_relative_posix_path,
    sha256_file,
    tree_fingerprint,
    validate_sha256,
)
from anchor_bench.core.profiles import load_yaml_mapping
from anchor_bench.robots.schema import RobotIdentity


ROBOT_IDENTITY_KIND = "robot_yaml_identity"
ASSET_PACKAGE_KIND = "runtime_asset_package"
ASSET_PREVIEW_KIND = "runtime_asset_preview"


def load_robot_identity_fingerprint(
    path: str | Path,
    *,
    expected_name: str,
    repository_root: str | Path,
) -> tuple[RobotIdentity, dict[str, Any]]:
    """Load one strict robot.yaml and return its raw and canonical identity."""

    source = Path(path)
    raw_sha256 = sha256_file(source)
    spec = RobotIdentity.from_dict(load_yaml_mapping(source), source)
    if spec.name != expected_name:
        raise ValueError(
            f"robot identity mismatch: requested {expected_name!r}, robot.yaml names {spec.name!r}"
        )
    if sha256_file(source) != raw_sha256:
        raise RuntimeError(f"robot identity changed while it was being loaded: {source}")
    fingerprint = {
        "kind": ROBOT_IDENTITY_KIND,
        "path": repo_relative_posix(
            source,
            repository_root=repository_root,
            must_exist=True,
        ),
        "raw_sha256": raw_sha256,
        "canonical_sha256": canonical_sha256(spec.to_dict()),
    }
    canonical_object(fingerprint)
    return spec, fingerprint


def robot_identity_fingerprint(
    bundle: Any,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Reauthenticate a registry bundle against its exact robot.yaml bytes."""

    loaded, fingerprint = load_robot_identity_fingerprint(
        bundle.paths.robot,
        expected_name=bundle.spec.name,
        repository_root=repository_root,
    )
    if loaded != bundle.spec:
        raise RuntimeError(f"robot identity changed after registry loading: {bundle.paths.robot}")
    return fingerprint


def verify_robot_identity_fingerprint(
    expected: Mapping[str, Any],
    bundle: Any,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Require current robot.yaml raw and canonical identities to match."""

    actual = robot_identity_fingerprint(bundle, repository_root=repository_root)
    if dict(expected) != actual:
        raise RuntimeError(
            "robot identity changed after resolution: "
            f"expected {dict(expected)}, got {actual}"
        )
    return actual


def asset_package_root(runtime_asset: str | Path, runtime_usd: str) -> Path:
    """Recover the selected asset candidate root from its relative USD name."""

    relative = require_relative_posix_path(runtime_usd)
    asset = Path(runtime_asset).resolve(strict=True)
    root = asset
    for _part in PurePosixPath(relative).parts:
        root = root.parent
    expected = root.joinpath(*PurePosixPath(relative).parts).resolve(strict=True)
    if expected != asset:
        raise ValueError(
            f"runtime asset {asset} does not occupy configured package path {relative!r}"
        )
    return root


def runtime_asset_fingerprint(
    runtime_asset: str | Path,
    *,
    runtime_usd: str,
) -> dict[str, Any]:
    """Hash the root USD and every regular file in its selected package root."""

    asset = Path(runtime_asset).resolve(strict=True)
    package_root = asset_package_root(asset, runtime_usd)
    package = tree_fingerprint(package_root)
    root_sha256 = sha256_file(asset)
    files = dict(package["files"])
    if files.get(runtime_usd) != root_sha256:
        raise RuntimeError("asset package fingerprint does not contain the configured root USD")
    fingerprint = {
        "kind": ASSET_PACKAGE_KIND,
        "runtime_usd": runtime_usd,
        "asset_sha256": root_sha256,
        "root_usd_sha256": root_sha256,
        "package_tree_sha256": package["tree_sha256"],
        "package_files": files,
    }
    canonical_object(fingerprint)
    return fingerprint


def runtime_asset_package_sha256(fingerprint: Mapping[str, Any]) -> str:
    """Return the authoritative package-tree digest from a runtime asset identity."""

    if fingerprint.get("kind") != ASSET_PACKAGE_KIND:
        raise ValueError("runtime asset fingerprint is not an authoritative package identity")
    value = fingerprint.get("package_tree_sha256")
    if not isinstance(value, str):
        raise ValueError(
            "authoritative runtime asset fingerprint is missing package_tree_sha256"
        )
    return validate_sha256(value, label="runtime asset package_tree_sha256")


def preview_asset_fingerprint(
    manifest_path: str | Path,
    *,
    runtime_usd: str,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Build a clearly non-authoritative dry-run identity without resolving USDs."""

    document = {
        "kind": ASSET_PREVIEW_KIND,
        "runtime_usd": require_relative_posix_path(runtime_usd),
        "manifest_path": repo_relative_posix(
            manifest_path,
            repository_root=repository_root,
            must_exist=True,
        ),
        "manifest_sha256": sha256_file(manifest_path),
    }
    return {**document, "preview_sha256": canonical_sha256(document)}


def fingerprint_contains_preview(value: Any) -> bool:
    """Return whether a nested fingerprint envelope contains preview material."""

    if isinstance(value, Mapping):
        kind = value.get("kind")
        if isinstance(kind, str) and ("preview" in kind or kind.startswith("dry_run_")):
            return True
        return any(fingerprint_contains_preview(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(fingerprint_contains_preview(item) for item in value)
    return False


def verify_runtime_asset_fingerprint(
    expected: Mapping[str, Any],
    runtime_asset: str | Path,
    *,
    runtime_usd: str,
) -> dict[str, Any]:
    """Verify a package identity, with root-only support for older manifests."""

    actual = runtime_asset_fingerprint(runtime_asset, runtime_usd=runtime_usd)
    if expected.get("kind") == ASSET_PACKAGE_KIND:
        if dict(expected) != actual:
            raise RuntimeError("runtime asset package changed after resolution")
        return actual
    legacy = expected.get("asset_sha256", expected.get("sha256"))
    if not isinstance(legacy, str):
        raise ValueError("resolved asset fingerprint is not authoritative")
    if validate_sha256(legacy, label="asset fingerprint") != actual["root_usd_sha256"]:
        raise RuntimeError("runtime asset changed after resolution")
    return actual


__all__ = [
    "ASSET_PACKAGE_KIND",
    "ASSET_PREVIEW_KIND",
    "ROBOT_IDENTITY_KIND",
    "asset_package_root",
    "fingerprint_contains_preview",
    "load_robot_identity_fingerprint",
    "preview_asset_fingerprint",
    "robot_identity_fingerprint",
    "runtime_asset_fingerprint",
    "runtime_asset_package_sha256",
    "verify_robot_identity_fingerprint",
    "verify_runtime_asset_fingerprint",
]
