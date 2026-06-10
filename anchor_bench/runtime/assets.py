from __future__ import annotations

import os
from pathlib import Path

from anchor_bench.robots.registry import RobotBundle


class AssetNotFoundError(FileNotFoundError):
    """Raised when an external robot asset is missing."""


def _candidate_roots(bundle: RobotBundle) -> list[Path]:
    roots: list[Path] = [bundle.paths.asset.parent / "files"]
    env_root = os.environ.get("ANCHOR_BENCH_ASSET_DIR")
    if env_root:
        roots.insert(0, Path(env_root) / bundle.spec.name)
        roots.insert(0, Path(env_root))
    return roots


def resolve_robot_asset(bundle: RobotBundle, filename: str) -> Path:
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
    filename = bundle.spec.runtime_usd
    if not filename:
        raise AssetNotFoundError(
            f"Robot {bundle.spec.name!r} has no runtime_usd configured in robot.yaml"
        )
    return resolve_robot_asset(bundle, filename)
