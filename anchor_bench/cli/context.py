"""Host-safe construction of public CLI resolver dependencies."""

from __future__ import annotations

import importlib.metadata
import platform
from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

from anchor_bench.benchmark.phase import BenchmarkResolverInputs
from anchor_bench.core.fingerprints import source_fingerprint
from anchor_bench.core.hashing import canonical_sha256, sha256_file, tree_fingerprint
from anchor_bench.core.paths import accepted_state_path, phase_profiles_dir, repo_root
from anchor_bench.core.versions import load_external_versions
from anchor_bench.deploy.runner import DeployResolverInputs
from anchor_bench.robots.registry import RobotBundle, get_robot
from anchor_bench.sysid.runner import SysIdResolverInputs
from anchor_bench.training.runner import TrainingResolverInputs


Phase = Literal["sysid", "train", "benchmark"]
DatasetFingerprintResolver = Callable[[Path], Mapping[str, Any]]

_PHASE_DISTRIBUTIONS: dict[Phase, tuple[str, ...]] = {
    "sysid": ("numpy", "cmaes", "mujoco", "mujoco-warp"),
    "train": ("numpy", "torch", "optuna", "mujoco", "mujoco-warp"),
    "benchmark": ("numpy", "torch", "mujoco", "mujoco-warp"),
}


def _distribution_version(name: str) -> str | None:
    """Return installed distribution metadata without importing its package."""

    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def dependency_fingerprints(phase: Phase) -> dict[str, Any]:
    """Freeze pinned runtimes and installed phase dependencies."""

    versions = load_external_versions()
    return {
        "python": platform.python_version(),
        "compatibility": asdict(versions),
        "installed": {
            name: _distribution_version(name)
            for name in _PHASE_DISTRIBUTIONS[phase]
        },
    }


def _asset_fingerprint(bundle: RobotBundle, *, dry_run: bool) -> dict[str, Any]:
    """Hash the runtime USD for execution or its manifest for a dry preview."""

    if dry_run:
        manifest_sha256 = sha256_file(bundle.paths.asset)
        return {
            "kind": "dry_run_asset_manifest",
            "runtime_usd": bundle.spec.runtime_usd,
            "manifest_sha256": manifest_sha256,
            "sha256": manifest_sha256,
        }

    from anchor_bench.runtime.assets import resolve_runtime_asset

    runtime_asset = resolve_runtime_asset(bundle)
    return {
        "kind": "runtime_usd",
        "runtime_usd": bundle.spec.runtime_usd,
        "sha256": sha256_file(runtime_asset),
    }


def _dataset_fingerprint_resolver(
    repository_root: Path,
    *,
    dry_run: bool,
) -> DatasetFingerprintResolver:
    """Return exact tree hashes, with declared-path previews only when absent."""

    def resolve(path: Path) -> Mapping[str, Any]:
        """Fingerprint one dataset without importing simulator dependencies."""

        if path.exists():
            if not path.is_dir():
                raise ValueError(f"dataset path must be a directory: {path}")
            return tree_fingerprint(path)
        if not dry_run:
            raise FileNotFoundError(f"dataset directory does not exist: {path}")
        try:
            declared_path = path.resolve(strict=False).relative_to(repository_root).as_posix()
        except ValueError as error:
            raise ValueError(f"dataset path escapes repository: {path}") from error
        preview = {
            "kind": "dry_run_declared_dataset",
            "path": declared_path,
        }
        return {**preview, "tree_sha256": canonical_sha256(preview)}

    return resolve


def _common_context(robot: str, phase: Phase, *, dry_run: bool) -> tuple[Path, RobotBundle, dict[str, Any]]:
    """Load robot identity and common semantic fingerprints once."""

    root = repo_root().resolve(strict=True)
    bundle = get_robot(robot)
    common = {
        "asset": _asset_fingerprint(bundle, dry_run=dry_run),
        "source": source_fingerprint(root),
        "dependencies": dependency_fingerprints(phase),
    }
    return root, bundle, common


def sysid_resolver_inputs(robot: str, *, dry_run: bool) -> SysIdResolverInputs:
    """Build repository-owned dependencies for one SysID resolution."""

    root, bundle, common = _common_context(robot, "sysid", dry_run=dry_run)
    return SysIdResolverInputs(
        repository_root=root,
        profiles_directory=phase_profiles_dir(robot, "sysid"),
        accepted_state=accepted_state_path(robot, "sysid"),
        canonical_joint_names=bundle.spec.joint_names,
        joint_aliases=bundle.spec.joint_aliases,
        asset_fingerprint=common["asset"],
        source_fingerprint=common["source"],
        dependency_fingerprints=common["dependencies"],
        dataset_fingerprint_resolver=_dataset_fingerprint_resolver(
            root,
            dry_run=dry_run,
        ),
    )


def training_resolver_inputs(robot: str, *, dry_run: bool) -> TrainingResolverInputs:
    """Build repository-owned dependencies for one training resolution."""

    root, bundle, common = _common_context(robot, "train", dry_run=dry_run)
    return TrainingResolverInputs(
        repository_root=root,
        profiles_directory=phase_profiles_dir(robot, "train"),
        accepted_sysid_state=accepted_state_path(robot, "sysid"),
        canonical_joint_names=bundle.spec.joint_names,
        joint_aliases=bundle.spec.joint_aliases,
        asset_fingerprint=common["asset"],
        source_fingerprint=common["source"],
        dependency_fingerprints=common["dependencies"],
        dataset_fingerprint_resolver=_dataset_fingerprint_resolver(
            root,
            dry_run=dry_run,
        ),
    )


def benchmark_resolver_inputs(robot: str, *, dry_run: bool) -> BenchmarkResolverInputs:
    """Build repository-owned dependencies for one benchmark resolution."""

    root, bundle, common = _common_context(robot, "benchmark", dry_run=dry_run)
    return BenchmarkResolverInputs(
        repository_root=root,
        profiles_directory=phase_profiles_dir(robot, "benchmark"),
        canonical_joint_names=bundle.spec.joint_names,
        joint_aliases=bundle.spec.joint_aliases,
        asset_fingerprint=common["asset"],
        source_fingerprint=common["source"],
        dependency_fingerprints=common["dependencies"],
    )


def deploy_resolver_inputs(robot: str) -> DeployResolverInputs:
    """Build exact accepted-state paths for deployment promotion."""

    root = repo_root().resolve(strict=True)
    bundle = get_robot(robot)
    return DeployResolverInputs(
        repository_root=root,
        robot=robot,
        profiles_directory=phase_profiles_dir(robot, "deploy"),
        accepted_training_state=accepted_state_path(robot, "train"),
        accepted_sysid_state=accepted_state_path(robot, "sysid"),
        deployed_state=accepted_state_path(robot, "deploy"),
        canonical_joint_names=bundle.spec.joint_names,
        joint_aliases=bundle.spec.joint_aliases,
    )


__all__ = [
    "benchmark_resolver_inputs",
    "dependency_fingerprints",
    "deploy_resolver_inputs",
    "sysid_resolver_inputs",
    "training_resolver_inputs",
]
