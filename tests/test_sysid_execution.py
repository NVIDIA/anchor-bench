from __future__ import annotations

from pathlib import Path

import pytest
import yaml


def _profile() -> dict:
    return {
        "schema_version": 1,
        "data": "inputs/demo",
        "joints": ["a", "B"],
        "simulation": {
            "physics_freq": 500,
            "control_freq": 500,
            "render_freq": 500,
            "integrator": "implicitfast",
            "solver_iterations": 10,
            "line_search_iterations": 10,
        },
        "num_envs": 4,
        "buffer_time": 1.0,
        "seed": 7,
        "optimizer": {
            "type": "cmaes",
            "sigma": 0.5,
            "max_iterations": 8,
            "epsilon": 0.001,
        },
        "parameters": {
            "stiffness": {
                "mode": "per_joint",
                "bounds": {"lower": 0.0, "upper": 100.0},
            },
            "damping": {
                "mode": "shared",
                "bounds": {"lower": 0.0, "upper": 10.0},
            },
        },
        "loss": {"position": 1.0, "velocity": 1.0},
    }


def _accepted() -> dict:
    return {
        "schema_version": 1,
        "revision": 0,
        "parameters": {
            "stiffness": {"joint_a": 10.0, "joint_b": 20.0},
            "damping": {"joint_a": 1.0, "joint_b": 3.0},
            "armature": {"joint_a": 0.01, "joint_b": 0.02},
        },
        "provenance": {"history": []},
    }


def _setup(tmp_path: Path):
    from anchor_bench.sysid.runner import SysIdResolverInputs

    profiles = tmp_path / "configs/robots/demo/sysid/profiles"
    profiles.mkdir(parents=True)
    (profiles / "default.yaml").write_text(
        yaml.safe_dump(_profile(), sort_keys=False), encoding="utf-8"
    )
    accepted = profiles.parent / "default.yaml"
    accepted.write_text(yaml.safe_dump(_accepted(), sort_keys=False), encoding="utf-8")
    dataset = tmp_path / "inputs/demo"
    dataset.mkdir(parents=True)
    (dataset / "episode.csv").write_text("t,q\n0,0\n", encoding="utf-8")

    def run_dir(_phase: str, **kwargs) -> Path:
        return (
            tmp_path
            / "outputs/sysid/demo"
            / kwargs["profile"]
            / f"seed-{kwargs['seed']}"
            / kwargs["run_input_sha256"]
        )

    inputs = SysIdResolverInputs(
        repository_root=tmp_path,
        profiles_directory=profiles,
        accepted_state=accepted,
        canonical_joint_names=("joint_a", "joint_b"),
        joint_aliases={"a": "joint_a", "B": "joint_b"},
        asset_fingerprint={"sha256": "a" * 64},
        source_fingerprint={"git": "b" * 40},
        dependency_fingerprints={"newton": "1.4.0.dev0"},
        run_directory_resolver=run_dir,
    )
    return inputs, accepted


def _execution_result():
    from anchor_bench.sysid.runner import SysIdExecutionResult

    return SysIdExecutionResult(
        fitted_parameters={
            "stiffness": {"joint_a": 12.0, "joint_b": 25.0},
            "damping": {"joint_a": 2.0, "joint_b": 2.0},
        },
        metrics={"best_loss": 0.25},
        optimizer_artifacts={"history.csv": "generation,loss\n0,0.25\n"},
    )


def test_resolution_replaces_profile_fields_and_canonicalizes_aliases(tmp_path: Path) -> None:
    from anchor_bench.sysid.runner import SysIdInvocation, resolve_sysid_run

    inputs, _ = _setup(tmp_path)
    resolved = resolve_sysid_run(
        SysIdInvocation(robot="demo", num_envs=8, seed=11), inputs
    )

    assert resolved.resolved_config.joints == ["joint_a", "joint_b"]
    assert resolved.resolved_config.num_envs == 8
    assert resolved.resolved_config.seed == 11
    assert resolved.cli_overrides == {"num_envs": 8, "seed": 11}
    assert resolved.run_dir.name == resolved.run_input_sha256
    assert resolved.run_input["parent_state"]["content_sha256"] == resolved.parent_state_sha256

    with pytest.raises(ValueError, match="duplicate SysID joints after alias"):
        resolve_sysid_run(
            SysIdInvocation(robot="demo", joints=("a", "joint_a")), inputs
        )


def test_dry_run_writes_nothing_and_execution_is_manifested_noop(tmp_path: Path) -> None:
    from anchor_bench.core.artifacts import verify_run_directory
    from anchor_bench.sysid.runner import (
        SysIdInvocation,
        execute_sysid_run,
        resolve_sysid_run,
    )

    inputs, _ = _setup(tmp_path)
    dry = resolve_sysid_run(SysIdInvocation(robot="demo"), inputs)
    outcome = execute_sysid_run(dry)
    assert outcome.dry_run is True
    assert not dry.run_dir.exists()

    resolved = resolve_sysid_run(
        SysIdInvocation(robot="demo", dry_run=False), inputs
    )
    calls = 0

    def backend(_resolved):
        nonlocal calls
        calls += 1
        return _execution_result()

    first = execute_sysid_run(resolved, backend)
    assert first.no_op is False
    assert calls == 1
    manifest = verify_run_directory(resolved.run_dir, resolved.run_input_sha256)
    assert manifest.phase == "sysid"
    assert {entry.path for entry in manifest.generated_files} == {
        "resolved_config.yaml",
        "parent_state.yaml",
        "fitted_patch.yaml",
        "sysid_state.yaml",
        "metrics.yaml",
        "optimizer/history.csv",
    }

    second = execute_sysid_run(
        resolved,
        lambda _: (_ for _ in ()).throw(AssertionError("backend called")),
    )
    assert second.no_op is True
    assert calls == 1


def _write_benchmark(tmp_path: Path, source_run: Path) -> Path:
    from anchor_bench.core.artifacts import generated_file, write_manifest
    from anchor_bench.core.manifest_schema import RunManifest
    from anchor_bench.core.artifacts import load_manifest

    source, source_hash = load_manifest(source_run)
    run_dir = tmp_path / "outputs/benchmark/demo/evidence"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.yaml").write_text("position_rmse: 0.1\n", encoding="utf-8")
    source_relative = source_run.relative_to(tmp_path).as_posix()
    manifest = RunManifest.build(
        phase="benchmark",
        robot="demo",
        profile=source.profile,
        resolved_config={"schema_version": 1},
        run_input={"source_manifest_sha256": source_hash},
        upstream={
            "source": {
                "phase": "sysid",
                "robot": "demo",
                "run": source_relative,
                "run_input_sha256": source.run_input_sha256,
                "run_manifest_sha256": source_hash,
            }
        },
        generated_files=[generated_file(run_dir, "metrics.yaml", "metrics")],
    )
    write_manifest(run_dir, manifest)
    return run_dir


def test_sparse_promotion_preserves_cells_and_accepts_later_evidence(tmp_path: Path) -> None:
    from anchor_bench.core.profiles import load_yaml_mapping
    from anchor_bench.sysid.promotion import promote_sysid_run
    from anchor_bench.sysid.runner import SysIdInvocation, execute_sysid_run, resolve_sysid_run

    inputs, accepted = _setup(tmp_path)
    resolved = resolve_sysid_run(
        SysIdInvocation(robot="demo", dry_run=False), inputs
    )
    execute_sysid_run(resolved, lambda _: _execution_result())

    first = promote_sysid_run(
        resolved.run_dir, accepted, repository_root=tmp_path
    )
    assert first.no_op is False
    assert first.state.revision == 1
    assert first.state.parameters["stiffness"]["joint_b"] == 25.0
    assert first.state.parameters["armature"] == _accepted()["parameters"]["armature"]

    repeat = promote_sysid_run(
        resolved.run_dir, accepted, repository_root=tmp_path
    )
    assert repeat.no_op is True
    benchmark = _write_benchmark(tmp_path, resolved.run_dir)
    evidence = promote_sysid_run(
        resolved.run_dir,
        accepted,
        repository_root=tmp_path,
        benchmark_runs=(benchmark,),
    )
    assert evidence.no_op is False
    assert evidence.state.revision == 2
    assert evidence.state.provenance.history[-1].updated_cells == []
    assert len(evidence.state.provenance.history[-1].benchmarks) == 1
    assert load_yaml_mapping(accepted)["revision"] == 2


def test_optimizer_uses_parent_mean_and_shared_layout(tmp_path: Path) -> None:
    from anchor_bench.sysid.optimizer import CMAESOptimizer
    from anchor_bench.sysid.runner import SysIdInvocation, resolve_sysid_run

    inputs, _ = _setup(tmp_path)
    resolved = resolve_sysid_run(SysIdInvocation(robot="demo"), inputs)
    optimizer = CMAESOptimizer(
        resolved.resolved_config,
        num_envs=3,
        cmaes=resolved.resolved_config.optimizer,
        seed=5,
        allow_random_fallback=True,
        parent_parameters=resolved.parent_state.parameters,
    )

    assert optimizer.param_names == [
        "stiffness/joint_a",
        "stiffness/joint_b",
        "damping/shared",
    ]
    assert optimizer.normalized_mean == pytest.approx([-0.8, -0.6, -0.6])
    population = optimizer.sample_population()
    assert all(item["damping"]["joint_a"] == item["damping"]["joint_b"] for item in population)


def test_promotion_rejects_another_robots_accepted_state(tmp_path: Path) -> None:
    """Bind promotion to the robot encoded by the accepted-state path."""

    from anchor_bench.sysid.promotion import promote_sysid_run
    from anchor_bench.sysid.runner import SysIdInvocation, execute_sysid_run, resolve_sysid_run

    inputs, _ = _setup(tmp_path)
    resolved = resolve_sysid_run(SysIdInvocation(robot="demo", dry_run=False), inputs)
    execute_sysid_run(resolved, lambda _: _execution_result())
    other = tmp_path / "configs/robots/other/sysid/default.yaml"
    other.parent.mkdir(parents=True)
    other.write_text(yaml.safe_dump(_accepted(), sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="promotion robot mismatch"):
        promote_sysid_run(resolved.run_dir, other, repository_root=tmp_path)
