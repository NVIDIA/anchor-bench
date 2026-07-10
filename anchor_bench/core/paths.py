
from __future__ import annotations

from pathlib import Path
from typing import Literal

from anchor_bench.core.hashing import validate_sha256
from anchor_bench.core.profiles import require_logical_name


PhaseName = Literal["sysid", "train", "benchmark", "deploy"]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def config_root() -> Path:
    return repo_root() / "configs"


def robot_config_root() -> Path:
    return config_root() / "robots"


def asset_root() -> Path:
    return repo_root() / "assets"


def output_root() -> Path:
    return repo_root() / "outputs"


def third_party_root() -> Path:
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
    """Derive an immutable SysID or training output directory."""

    digest = validate_sha256(run_input_sha256, label="run_input_sha256")
    profile_name = require_logical_name(profile, label="profile name")
    robot_name = require_logical_name(robot, label="robot name")
    return output_root() / phase / robot_name / profile_name / f"seed-{seed}" / digest


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
