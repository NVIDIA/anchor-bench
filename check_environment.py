
#!/usr/bin/env python3
"""Check the Anchor Bench runtime environment."""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

from anchor_bench.core.paths import repo_root
from anchor_bench.core.versions import load_external_versions
from anchor_bench.robots.registry import list_robots, validate_robot


def _module_version(name: str) -> str | None:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return None
    module = __import__(name)
    return getattr(module, "__version__", "installed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Report checks without requiring heavy runtime installs.")
    args = parser.parse_args(argv)

    versions = load_external_versions()
    print(f"repo: {repo_root()}")
    print(f"required python: {versions.python_version}")
    print(f"current python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    print(f"required isaac sim: {versions.isaac_sim_version}")
    print(f"required newton: {versions.newton_version} ({versions.newton_tag})")
    print(f"required isaac lab: {versions.isaac_lab_branch}@{versions.isaac_lab_commit}")

    errors: list[str] = []
    if not args.dry_run and (sys.version_info.major, sys.version_info.minor) != (3, 12):
        errors.append("Python 3.12 is required for the full Isaac Sim 6.0 runtime.")

    for robot in list_robots():
        errors.extend(validate_robot(robot))
    print(f"robots: {', '.join(list_robots())}")

    newton_version = _module_version("newton")
    isaacsim_version = _module_version("isaacsim")
    isaaclab_version = _module_version("isaaclab")
    isaaclab_newton_version = _module_version("isaaclab_newton")
    print(f"newton installed: {newton_version or 'not installed'}")
    print(f"isaacsim installed: {isaacsim_version or 'not installed'}")
    print(f"isaaclab installed: {isaaclab_version or 'not installed'}")
    print(f"isaaclab_newton installed: {isaaclab_newton_version or 'not installed'}")

    if not args.dry_run:
        if newton_version is None:
            errors.append("newton is not installed")
        if isaacsim_version is None:
            errors.append("isaacsim is not installed")
        if isaaclab_version is None:
            errors.append("isaaclab is not installed")
        if isaaclab_newton_version is None:
            errors.append("isaaclab_newton is not installed")
        lab_dir = Path(os.environ.get("ANCHOR_BENCH_ISAACLAB_DIR", repo_root() / "third_party" / "IsaacLab"))
        if not (lab_dir / ".git").exists():
            errors.append(f"Isaac Lab checkout is missing at {lab_dir}")
        else:
            result = subprocess.run(
                ["git", "-c", f"safe.directory={lab_dir}", "-C", str(lab_dir), "rev-parse", "HEAD"],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if result.stdout.strip() != versions.isaac_lab_commit:
                errors.append(f"Isaac Lab checkout is not pinned to {versions.isaac_lab_commit}")

    if errors:
        print("environment check failed:")
        for err in errors:
            print(f"- {err}")
        return 1
    print("environment check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
