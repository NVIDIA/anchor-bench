"""Manifest-backed writers for one immutable benchmark result."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from anchor_bench.core.artifacts import generated_files, write_manifest_last
from anchor_bench.core.hashing import require_relative_posix_path
from anchor_bench.core.manifest_schema import RunManifest

if TYPE_CHECKING:
    from anchor_bench.benchmark.phase import (
        BenchmarkExecutionResult,
        BenchmarkTrajectoryResult,
        ResolvedBenchmarkRun,
    )


def _write_yaml(path: Path, document: Mapping[str, object]) -> None:
    """Write one deterministic UTF-8 YAML mapping into the staging directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(dict(document), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _write_payload(path: Path, payload: str | bytes) -> None:
    """Write one backend-provided CSV payload without changing its bytes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        if not payload:
            raise ValueError(f"trajectory payload for {path} must not be empty")
        path.write_text(payload, encoding="utf-8")
    elif isinstance(payload, bytes):
        if not payload:
            raise ValueError(f"trajectory payload for {path} must not be empty")
        path.write_bytes(payload)
    else:
        raise TypeError(f"trajectory payload for {path} must be str or bytes")


def _validated_trajectories(
    resolved: "ResolvedBenchmarkRun",
    result: "BenchmarkExecutionResult",
) -> dict[str, "BenchmarkTrajectoryResult"]:
    """Require exactly one real/simulated pair for every selected motion."""

    if not isinstance(result.trajectories, Mapping):
        raise TypeError("benchmark trajectories must be a mapping")
    actual = set(result.trajectories)
    expected = {motion.name for motion in resolved.motions}
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append("unexpected " + ", ".join(extra))
        raise ValueError("benchmark trajectory result mismatch: " + "; ".join(details))
    ordered: dict[str, BenchmarkTrajectoryResult] = {}
    for motion in resolved.motions:
        value = result.trajectories[motion.name]
        from anchor_bench.benchmark.phase import BenchmarkTrajectoryResult

        if not isinstance(value, BenchmarkTrajectoryResult):
            raise TypeError(
                f"trajectory result {motion.name!r} must be BenchmarkTrajectoryResult"
            )
        ordered[motion.name] = value
    return ordered


def write_benchmark_run(
    run_dir: Path,
    resolved: "ResolvedBenchmarkRun",
    result: "BenchmarkExecutionResult",
) -> RunManifest:
    """Write metrics and trajectories, then finalize the benchmark manifest."""

    metrics = result.validated_metrics()
    trajectories = _validated_trajectories(resolved, result)
    _write_yaml(run_dir / "metrics.yaml", metrics)
    roles: dict[str, str] = {"metrics.yaml": "metrics"}
    for motion_name, trajectory in trajectories.items():
        safe_name = require_relative_posix_path(motion_name)
        real_path = f"trajectories/{safe_name}/real.csv"
        simulated_path = f"trajectories/{safe_name}/simulated.csv"
        _write_payload(run_dir / real_path, trajectory.real_csv)
        _write_payload(run_dir / simulated_path, trajectory.simulated_csv)
        roles[real_path] = "real_trajectory"
        roles[simulated_path] = "simulated_trajectory"

    manifest = RunManifest.build(
        phase="benchmark",
        robot=resolved.robot,
        profile=resolved.profile_snapshot,
        resolved_config=resolved.resolved_config,
        cli_overrides=resolved.cli_overrides,
        run_input=resolved.run_input,
        run_input_sha256=resolved.run_input_sha256,
        fingerprints=resolved.fingerprints,
        upstream={"source": resolved.source_reference.to_dict()},
        generated_files=generated_files(run_dir, roles),
    )
    write_manifest_last(run_dir, manifest)
    return manifest


__all__ = ["write_benchmark_run"]
