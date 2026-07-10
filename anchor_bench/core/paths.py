"""Canonical repository paths for configs, assets, and immutable runs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from anchor_bench.core.hashing import validate_sha256
from anchor_bench.core.profiles import require_logical_name


PhaseName = Literal["sysid", "train", "benchmark", "deploy"]


def safe_run_path(
    run_dir: str | Path,
    *,
    containment_root: str | Path,
    must_exist: bool,
) -> Path:
    """Return a contained run directory after rejecting symlinked ancestors."""

    supplied_root = Path(containment_root)
    if supplied_root.is_symlink() or not supplied_root.is_dir():
        raise ValueError(
            f"containment_root must be a real directory: {supplied_root}"
        )
    root = Path(os.path.abspath(supplied_root))
    candidate = Path(run_dir)
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = Path(os.path.abspath(candidate))
    try:
        relative = candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(
            f"run directory escapes containment root {root}: {candidate}"
        ) from error
    if not relative.parts:
        raise ValueError("run directory must be below containment_root")

    current = root
    for part in relative.parts[:-1]:
        current /= part
        if not os.path.lexists(current):
            break
        if current.is_symlink():
            raise ValueError(f"run directory has a symlinked parent: {current}")
        if not current.is_dir():
            raise ValueError(f"run directory parent is not a directory: {current}")

    if os.path.lexists(candidate):
        if candidate.is_symlink():
            raise ValueError(f"run directory must not be a symlink: {candidate}")
        if not candidate.is_dir():
            raise ValueError(f"run path is not a directory: {candidate}")
    elif must_exist:
        raise ValueError(f"run directory does not exist: {candidate}")
    return candidate


def _phase_run_path(
    output_directory: Path,
    phase: Literal["sysid", "train"],
    *,
    robot: str,
    profile: str,
    seed: int,
    run_input_sha256: str,
) -> Path:
    """Build one validated content-addressed phase path below an output root."""

    digest = validate_sha256(run_input_sha256, label="run_input_sha256")
    profile_name = require_logical_name(profile, label="profile name")
    robot_name = require_logical_name(robot, label="robot name")
    return (
        output_directory
        / phase
        / robot_name
        / profile_name
        / f"seed-{seed}"
        / digest
    )


def repo_root() -> Path:
    """Return the Anchor Bench repository root."""

    return Path(__file__).resolve().parents[2]


def config_root() -> Path:
    """Return the repository's declarative configuration root."""

    return repo_root() / "configs"


def robot_config_root() -> Path:
    """Return the root containing per-robot configuration directories."""

    return config_root() / "robots"


def asset_root() -> Path:
    """Return the repository's tracked asset root."""

    return repo_root() / "assets"


def output_root() -> Path:
    """Return the root for generated phase outputs."""

    return repo_root() / "outputs"


def third_party_root() -> Path:
    """Return the root for pinned third-party source trees."""

    return repo_root() / "third_party"


def robot_config_dir(robot: str) -> Path:
    """Return one robot's declarative configuration directory."""

    return robot_config_root() / require_logical_name(robot, label="robot name")


def phase_config_dir(robot: str, phase: PhaseName) -> Path:
    """Return one phase's configuration directory for a robot."""

    return robot_config_dir(robot) / phase


def phase_profiles_dir(robot: str, phase: PhaseName) -> Path:
    """Return the directory containing complete phase recipes."""

    return phase_config_dir(robot, phase) / "profiles"


def accepted_state_path(robot: str, phase: Literal["sysid", "train", "deploy"]) -> Path:
    """Return the machine-managed accepted state for a phase."""

    return phase_config_dir(robot, phase) / "default.yaml"


def phase_run_dir(
    phase: Literal["sysid", "train"],
    *,
    robot: str,
    profile: str,
    seed: int,
    run_input_sha256: str,
) -> Path:
    """Derive a contained immutable SysID or training output directory."""

    candidate = _phase_run_path(
        output_root(),
        phase,
        robot=robot,
        profile=profile,
        seed=seed,
        run_input_sha256=run_input_sha256,
    )
    return safe_run_path(
        candidate,
        containment_root=repo_root(),
        must_exist=False,
    )


def require_canonical_phase_run_dir(
    run_dir: str | Path,
    *,
    repository_root: str | Path,
    phase: Literal["sysid", "train"],
    robot: str,
    profile: str,
    seed: int | None,
    run_input_sha256: str,
) -> Path:
    """Require an existing run at its exact canonical content-addressed path."""

    if seed is None:
        raise ValueError(f"completed {phase} run manifest must contain a seed")
    root = Path(os.path.abspath(repository_root))
    actual = safe_run_path(
        run_dir,
        containment_root=root,
        must_exist=True,
    )
    expected = _phase_run_path(
        root / "outputs",
        phase,
        robot=robot,
        profile=profile,
        seed=seed,
        run_input_sha256=run_input_sha256,
    )
    if actual != expected:
        raise ValueError(
            f"{phase} promotion source must be the canonical run directory "
            f"{expected}"
        )
    return actual


def benchmark_run_dir(
    *,
    output_directory: str | Path,
    robot: str,
    source_phase: Literal["sysid", "train"],
    profile: str,
    source_manifest_sha256: str,
    benchmark_input_sha256: str,
) -> Path:
    """Derive a benchmark output; its caller-selected root is non-semantic."""

    robot_name = require_logical_name(robot, label="robot name")
    profile_name = require_logical_name(profile, label="profile name")
    source_hash = validate_sha256(source_manifest_sha256, label="source_manifest_sha256")
    benchmark_hash = validate_sha256(benchmark_input_sha256, label="benchmark_input_sha256")
    return Path(output_directory) / robot_name / source_phase / profile_name / source_hash / benchmark_hash
