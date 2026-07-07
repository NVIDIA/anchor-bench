#!/usr/bin/env python3
"""Check the Anchor Bench runtime environment."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, distribution
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


def _distribution_version(name: str) -> str | None:
    try:
        return distribution(name).version
    except PackageNotFoundError:
        return None


def _distribution_vcs_commit(name: str) -> str | None:
    try:
        installed = distribution(name)
    except PackageNotFoundError:
        return None
    direct_url = installed.read_text("direct_url.json")
    if not direct_url:
        return None
    try:
        payload = json.loads(direct_url)
    except json.JSONDecodeError:
        return None
    commit = payload.get("vcs_info", {}).get("commit_id")
    return str(commit) if commit else None


def _newton_capabilities() -> tuple[bool, bool, str | None]:
    if importlib.util.find_spec("newton") is None:
        return False, False, None
    try:
        import newton
    except Exception as exc:  # pragma: no cover - depends on a broken external install
        return False, False, f"{type(exc).__name__}: {exc}"

    solver_type = getattr(getattr(newton, "solvers", None), "SolverMuJoCo", None)
    actuator_type = getattr(
        getattr(newton, "actuators", None), "ControllerNeuralGRU", None
    )
    return (
        actuator_type is not None,
        solver_type is not None and hasattr(solver_type, "eval_dynamic_bias"),
        None,
    )


def _newton_environment_errors(
    installed_version: str | None,
    installed_commit: str | None,
    installed_warp_version: str | None,
    has_gru_controller: bool,
    has_dynamic_bias: bool,
    import_error: str | None,
    *,
    required_version: str,
    required_commit: str,
    required_warp_version: str,
) -> list[str]:
    errors: list[str] = []
    if installed_version is None:
        errors.append("newton is not installed")
    else:
        if installed_version != required_version:
            errors.append(
                f"newton {required_version} is required; found {installed_version}"
            )
        if (installed_commit or "").lower() != required_commit.lower():
            errors.append(
                f"newton is not pinned to commit {required_commit}; "
                f"found {installed_commit or 'no VCS commit metadata'}"
            )
        if import_error:
            errors.append(f"newton import failed: {import_error}")
        else:
            if not has_gru_controller:
                errors.append("newton ControllerNeuralGRU is unavailable")
            if not has_dynamic_bias:
                errors.append("newton SolverMuJoCo.eval_dynamic_bias is unavailable")

    if installed_warp_version is None:
        errors.append("warp-lang is not installed")
    elif installed_warp_version != required_warp_version:
        errors.append(
            f"warp-lang {required_warp_version} is required; "
            f"found {installed_warp_version}"
        )
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=("kitless", "full"),
        default="full",
        help="Select lightweight Newton checks or the complete Isaac runtime checks.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate manifests and robot configs without importing runtime packages.",
    )
    args = parser.parse_args(argv)

    versions = load_external_versions()
    print(f"repo: {repo_root()}")
    print(f"required python: {versions.python_version}")
    print(
        f"current python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    print(f"required isaac sim: {versions.isaac_sim_version}")
    print(f"required newton: {versions.newton_version}@{versions.newton_commit}")
    print(f"required warp-lang: {versions.warp_version}")
    print(
        f"required isaac lab: {versions.isaac_lab_branch}@{versions.isaac_lab_commit}"
    )

    errors: list[str] = []
    for robot in list_robots():
        errors.extend(validate_robot(robot))
    print(f"robots: {', '.join(list_robots())}")

    if args.dry_run:
        print("runtime packages: skipped (--dry-run)")
    else:
        if (sys.version_info.major, sys.version_info.minor) != (3, 12):
            errors.append("Python 3.12 is required.")

        newton_version = _distribution_version(versions.newton_package)
        newton_commit = _distribution_vcs_commit(versions.newton_package)
        warp_version = _distribution_version(versions.warp_package)
        has_gru_controller, has_dynamic_bias, newton_import_error = (
            _newton_capabilities()
        )
        print(f"newton installed: {newton_version or 'not installed'}")
        print(f"newton commit: {newton_commit or 'not available'}")
        print(f"warp-lang installed: {warp_version or 'not installed'}")
        print(
            f"newton GRU controller: {'available' if has_gru_controller else 'unavailable'}"
        )
        print(
            f"newton dynamic bias API: {'available' if has_dynamic_bias else 'unavailable'}"
        )
        errors.extend(
            _newton_environment_errors(
                newton_version,
                newton_commit,
                warp_version,
                has_gru_controller,
                has_dynamic_bias,
                newton_import_error,
                required_version=versions.newton_version,
                required_commit=versions.newton_commit,
                required_warp_version=versions.warp_version,
            )
        )

    if not args.dry_run and args.profile == "full":
        isaacsim_version = _module_version("isaacsim")
        isaaclab_version = _module_version("isaaclab")
        isaaclab_newton_version = _module_version("isaaclab_newton")
        print(f"isaacsim installed: {isaacsim_version or 'not installed'}")
        print(f"isaaclab installed: {isaaclab_version or 'not installed'}")
        print(
            f"isaaclab_newton installed: {isaaclab_newton_version or 'not installed'}"
        )
        if isaacsim_version is None:
            errors.append("isaacsim is not installed")
        if isaaclab_version is None:
            errors.append("isaaclab is not installed")
        if isaaclab_newton_version is None:
            errors.append("isaaclab_newton is not installed")
        lab_dir = Path(
            os.environ.get(
                "ANCHOR_BENCH_ISAACLAB_DIR", repo_root() / "third_party" / "IsaacLab"
            )
        )
        if not (lab_dir / ".git").exists():
            errors.append(f"Isaac Lab checkout is missing at {lab_dir}")
        else:
            result = subprocess.run(
                [
                    "git",
                    "-c",
                    f"safe.directory={lab_dir}",
                    "-C",
                    str(lab_dir),
                    "rev-parse",
                    "HEAD",
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if result.stdout.strip() != versions.isaac_lab_commit:
                errors.append(
                    f"Isaac Lab checkout is not pinned to {versions.isaac_lab_commit}"
                )

    if errors:
        print("environment check failed:")
        for err in errors:
            print(f"- {err}")
        return 1
    print("environment check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
