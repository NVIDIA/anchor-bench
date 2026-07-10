"""Resolve declarative robot assets without importing the simulator."""

from __future__ import annotations

import os
from pathlib import Path

from anchor_bench.robots.registry import RobotBundle


class AssetNotFoundError(FileNotFoundError):
    """Raised when an external robot asset is missing."""


def _candidate_roots(bundle: RobotBundle) -> list[Path]:
    """Return configured and repository-local asset roots in precedence order."""

    roots: list[Path] = [bundle.paths.asset.parent / "files"]
    env_root = os.environ.get("ANCHOR_BENCH_ASSET_DIR")
    if env_root:
        external = Path(env_root)
        roots = [external / bundle.spec.name, external, *roots]
    return roots


def resolve_robot_asset(bundle: RobotBundle, filename: str) -> Path:
    """Resolve one robot asset or report every supported location."""

    candidates = [root / filename for root in _candidate_roots(bundle)]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise AssetNotFoundError(
        f"Missing asset {filename!r} for robot {bundle.spec.name}.\n"
        "Looked in:\n"
        "- $ANCHOR_BENCH_ASSET_DIR\n"
        f"- assets/robots/{bundle.spec.name}/files\n"
        "\n"
        f"Set ANCHOR_BENCH_ASSET_DIR or place assets under assets/robots/{bundle.spec.name}/files/."
    )


def resolve_runtime_asset(bundle: RobotBundle) -> Path:
    """Resolve the runtime USD named by a declarative robot identity."""

    filename = bundle.spec.runtime_usd
    if not filename:
        raise AssetNotFoundError(
            f"Robot {bundle.spec.name!r} has no runtime_usd configured in robot.yaml"
        )
    return resolve_robot_asset(bundle, filename)
