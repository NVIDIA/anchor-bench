"""Host-safe tests for immutable candidate benchmark orchestration."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import pytest
import yaml

from anchor_bench.benchmark.phase import (
    BenchmarkExecutionResult,
    BenchmarkInvocation,
    BenchmarkResolverInputs,
    BenchmarkTrajectoryResult,
    LoadedRuntimeMetadata,
    TrainingBenchmarkCandidate,
    execute_benchmark_run,
    load_embedded_runtime_metadata,
    resolve_benchmark_run,
)
from anchor_bench.core.artifacts import generated_files, verify_run_directory, write_manifest_last
from anchor_bench.core.hashing import canonical_sha256, sha256_file, sha256_hex
from anchor_bench.core.manifest_schema import RunManifest, build_profile_snapshot
from anchor_bench.sysid.state import (
    AcceptedSysIdState,
    SysIdCandidateSource,
    SysIdFittedPatch,
    build_candidate_state,
)
from anchor_bench.training.metadata import RuntimeGRUMetadata


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _write_yaml(path: Path, document: dict[str, Any]) -> None:
    """Write one deterministic test YAML document."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


def _write_motion(root: Path, joints: tuple[str, ...] = ("joint_a",)) -> None:
    """Write a minimal fingerprintable SAGE motion with state velocity."""

    root.mkdir(parents=True, exist_ok=True)
    (root / "joint_list.txt").write_text("\n".join(joints) + "\n", encoding="utf-8")
    (root / "control.csv").write_text(
        'type,timestamp,positions\nCONTROL,0,"[0.0]"\n', encoding="utf-8"
    )
    (root / "state_motor.csv").write_text(
        'type,timestamp,positions,velocities,torques\nSTATE_MOTOR,0,"[0.0]","[0.0]","[0.0]"\n',
        encoding="utf-8",
    )
    _write_yaml(
        root / "metadata.yaml",
        {
            "time_unit": "seconds",
            "position_unit": "rad",
            "velocity_unit": "rad/s",
            "torque_unit": "N*m",
        },
    )


def _write_benchmark_profile(
    repo: Path,
    *,
    control_freq: int = 500,
    output_root: str = "outputs/benchmark",
) -> Path:
    """Write the selected complete benchmark profile."""

    path = repo / "configs/robots/demo/benchmark/profiles/default.yaml"
    _write_yaml(
        path,
        {
            "schema_version": 1,
            "data": "inputs/held_out",
            "motions": ["reach"],
            "simulation": {
                "physics_freq": 500,
                "control_freq": control_freq,
                "render_freq": 500,
                "integrator": "implicitfast",
                "solver_iterations": 10,
                "line_search_iterations": 10,
            },
            "num_envs": 4,
            "buffer_time": 0.25,
            "sync_initial_state": True,
            "signals": ["position", "velocity"],
            "output_root": output_root,
        },
    )
    return path.parent


def _source_profile(repo: Path, phase: str, content: dict[str, Any]) -> Any:
    """Create a source profile snapshot whose bytes exist inside the test repo."""

    path = repo / f"configs/robots/demo/{phase}/profiles/candidate.yaml"
    raw = yaml.safe_dump(content, sort_keys=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(raw, encoding="utf-8")
    return build_profile_snapshot(
        name="candidate",
        path=path.relative_to(repo).as_posix(),
        raw_yaml=raw,
        content=content,
    )


def _install_sysid_source(repo: Path, *, control_freq: int = 500) -> Path:
    """Install one valid immutable SysID candidate under the canonical layout."""

    run_input = {"phase": "sysid", "case": "candidate"}
    run_hash = canonical_sha256(run_input)
    run_dir = repo / "outputs/sysid/demo/candidate/seed-42" / run_hash
    run_dir.mkdir(parents=True)
    parent = AcceptedSysIdState(
        schema_version=1,
        revision=0,
        parameters={
            "stiffness": {"joint_a": 10.0},
            "damping": {"joint_a": 1.0},
            "armature": {"joint_a": 0.01},
            "dry_friction": {"joint_a": 0.1},
            "viscous_friction": {"joint_a": 0.2},
        },
        provenance={"history": []},
    )
    patch = SysIdFittedPatch(
        schema_version=1,
        artifact_type="sysid_fitted_patch",
        robot="demo",
        parameters={"stiffness": {"joint_a": 12.0}},
    )
    _write_yaml(run_dir / "parent_state.yaml", parent.to_dict())
    _write_yaml(run_dir / "fitted_patch.yaml", patch.to_dict())
    state = build_candidate_state(
        parent,
        patch,
        SysIdCandidateSource(
            parent_state_sha256=sha256_file(run_dir / "parent_state.yaml"),
            fitted_patch_sha256=sha256_file(run_dir / "fitted_patch.yaml"),
            run_input_sha256=run_hash,
        ),
    )
    _write_yaml(run_dir / "sysid_state.yaml", state.to_dict())
    _write_yaml(run_dir / "metrics.yaml", {"loss": 0.1})
    resolved_config = {
        "joints": ["joint_a"],
        "simulation": {"control_freq": control_freq},
    }
    profile = _source_profile(repo, "sysid", {"schema_version": 1})
    manifest = RunManifest.build(
        phase="sysid",
        robot="demo",
        profile=profile,
        resolved_config=resolved_config,
        seed=42,
        run_input=run_input,
        generated_files=generated_files(
            run_dir,
            {
                "parent_state.yaml": "parent_state",
                "fitted_patch.yaml": "fitted_patch",
                "sysid_state.yaml": "sysid_state",
                "metrics.yaml": "metrics",
            },
        ),
    )
    write_manifest_last(run_dir, manifest)
    return run_dir


def _runtime_metadata(
    *,
    profile_sha256: str,
    resolved_config_sha256: str,
    sysid_state_sha256: str,
    sample_dt_s: float = 0.002,
) -> RuntimeGRUMetadata:
    """Build one valid scalar full-torque neural runtime contract."""

    return RuntimeGRUMetadata.from_dict(
        {
            "schema_version": 3,
            "model_type": "gru",
            "runtime_dtype": "float32",
            "input_size": 1,
            "output_size": 1,
            "recurrent": {"hidden_size": 8, "num_layers": 1, "dropout": 0.0},
            "input_features": [
                {"name": "position", "domain": "input", "components": ["value"], "width": 1}
            ],
            "flattening_order": ["feature", "joint", "component"],
            "mappings": [
                {
                    "input_joints": ["joint_a"],
                    "output_joints": ["joint_a"],
                    "effort_limits": 20.0,
                    "clamping": "total_output",
                }
            ],
            "target_columns": ["torque"],
            "sample_dt_s": sample_dt_s,
            "normalization": {
                "inputs": {"mean": {"position": 0.0}, "std": {"position": 1.0}},
                "targets": {"mean": {"torque": 0.0}, "std": {"torque": 1.0}},
            },
            "delay": {"handling": "learned", "external_delay_s": 0.0},
            "hashes": {
                "profile_sha256": profile_sha256,
                "resolved_config_sha256": resolved_config_sha256,
                "sysid_state_sha256": sysid_state_sha256,
                "asset_sha256": SHA_A,
                "source_sha256": SHA_B,
            },
        },
        "runtime metadata",
    )


def _install_training_source(
    repo: Path,
    *,
    sample_dt_s: float = 0.002,
) -> tuple[Path, LoadedRuntimeMetadata]:
    """Install a manifest-authenticated training run with injected metadata."""

    run_input = {"phase": "train", "case": "candidate"}
    run_hash = canonical_sha256(run_input)
    run_dir = repo / "outputs/train/demo/candidate/seed-42" / run_hash
    run_dir.mkdir(parents=True)
    (run_dir / "network.pt").write_bytes(b"authenticated-network")
    (run_dir / "sysid_state.yaml").write_text("frozen: state\n", encoding="utf-8")
    profile = _source_profile(repo, "train", {"schema_version": 1, "target": "torque"})
    resolved_config = {"target": "torque", "sample_dt_s": sample_dt_s}
    metadata = _runtime_metadata(
        profile_sha256=profile.content_sha256,
        resolved_config_sha256=canonical_sha256(resolved_config),
        sysid_state_sha256=sha256_file(run_dir / "sysid_state.yaml"),
        sample_dt_s=sample_dt_s,
    )
    raw = json.dumps(
        metadata.to_dict(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    manifest = RunManifest.build(
        phase="train",
        robot="demo",
        profile=profile,
        resolved_config=resolved_config,
        seed=42,
        run_input=run_input,
        fingerprints={
            "asset": {"asset_sha256": SHA_A},
            "dataset": {"tree_sha256": SHA_B},
        },
        generated_files=generated_files(
            run_dir,
            {"network.pt": "network", "sysid_state.yaml": "sysid_state"},
        ),
    )
    write_manifest_last(run_dir, manifest)
    return run_dir, LoadedRuntimeMetadata(metadata, raw, sha256_hex(raw))


def _resolver_inputs(
    repo: Path,
    profiles: Path,
    *,
    metadata: LoadedRuntimeMetadata | None = None,
) -> BenchmarkResolverInputs:
    """Build deterministic host-only resolver dependencies."""

    loader = (lambda _path: metadata) if metadata is not None else None
    return BenchmarkResolverInputs(
        repository_root=repo,
        profiles_directory=profiles,
        canonical_joint_names=("joint_a", "joint_b"),
        joint_aliases={"A": "joint_a"},
        asset_fingerprint={"asset_sha256": SHA_A},
        source_fingerprint={"source_sha256": SHA_B},
        dependency_fingerprints={"newton": "1.4.0.dev0"},
        metadata_loader=loader,
    )


def _setup_sysid_case(tmp_path: Path, *, control_freq: int = 500) -> tuple[Path, Path, BenchmarkResolverInputs]:
    """Create the common profile, motion, source, and resolver fixture."""

    profiles = _write_benchmark_profile(tmp_path, control_freq=control_freq)
    _write_motion(tmp_path / "inputs/held_out/reach")
    source = _install_sysid_source(tmp_path, control_freq=control_freq)
    return source, profiles, _resolver_inputs(tmp_path, profiles)


def test_resolver_authenticates_sysid_candidate_and_freezes_evidence(tmp_path: Path) -> None:
    """Resolution reconstructs SysID and emits the exact evidence envelope."""

    source, _profiles, inputs = _setup_sysid_case(tmp_path)
    resolved = resolve_benchmark_run(
        BenchmarkInvocation(robot="demo", run=source), inputs
    )

    assert resolved.source_reference.to_dict() == {
        "phase": "sysid",
        "robot": "demo",
        "run": source.relative_to(tmp_path).as_posix(),
        "run_input_sha256": source.name,
        "run_manifest_sha256": resolved.source_manifest.run_manifest_sha256,
    }
    assert resolved.motions[0].name == "reach"
    assert resolved.run_dir.name == resolved.run_input_sha256


def test_output_root_override_does_not_change_semantic_identity(tmp_path: Path) -> None:
    """Storage location changes the target path but not benchmark input hash."""

    source, _profiles, inputs = _setup_sysid_case(tmp_path)
    first = resolve_benchmark_run(
        BenchmarkInvocation(robot="demo", run=source, output_root="outputs/one"), inputs
    )
    second = resolve_benchmark_run(
        BenchmarkInvocation(robot="demo", run=source, output_root="outputs/two"), inputs
    )

    assert first.run_input_sha256 == second.run_input_sha256
    assert first.run_dir != second.run_dir
    assert "output_root" not in first.resolved_config
    assert "output_root" not in first.cli_overrides


def test_fresh_execution_writes_evidence_and_existing_run_is_noop(tmp_path: Path) -> None:
    """A second identical request verifies bytes without invoking replay."""

    source, _profiles, inputs = _setup_sysid_case(tmp_path)
    resolved = resolve_benchmark_run(
        BenchmarkInvocation(robot="demo", run=source, dry_run=False), inputs
    )
    calls = 0

    def backend(_resolved: Any) -> BenchmarkExecutionResult:
        """Return deterministic non-placeholder replay evidence."""

        nonlocal calls
        calls += 1
        return BenchmarkExecutionResult(
            metrics={"position_rmse": 0.125, "velocity_rmse": 0.25},
            trajectories={
                "reach": BenchmarkTrajectoryResult(
                    real_csv="time,position\n0,0\n",
                    simulated_csv="time,position\n0,0.1\n",
                )
            },
        )

    first = execute_benchmark_run(resolved, backend)
    second = execute_benchmark_run(resolved)

    assert not first.no_op
    assert second.no_op
    assert calls == 1
    manifest = verify_run_directory(resolved.run_dir, resolved.run_input_sha256)
    assert manifest.upstream == {"source": resolved.source_reference.to_dict()}
    assert (resolved.run_dir / "trajectories/reach/real.csv").is_file()


def test_fresh_run_requires_real_backend_and_writes_nothing(tmp_path: Path) -> None:
    """Host orchestration never invents placeholder metrics or trajectories."""

    source, _profiles, inputs = _setup_sysid_case(tmp_path)
    resolved = resolve_benchmark_run(
        BenchmarkInvocation(robot="demo", run=source, dry_run=False), inputs
    )

    with pytest.raises(ValueError, match="replay backend is required"):
        execute_benchmark_run(resolved)
    assert not resolved.run_dir.exists()


def test_training_candidate_uses_only_authenticated_network_metadata_contract(tmp_path: Path) -> None:
    """Training resolution derives axes and timing from embedded metadata."""

    profiles = _write_benchmark_profile(tmp_path)
    _write_motion(tmp_path / "inputs/held_out/reach")
    source, loaded = _install_training_source(tmp_path)
    resolved = resolve_benchmark_run(
        BenchmarkInvocation(robot="demo", run=source),
        _resolver_inputs(tmp_path, profiles, metadata=loaded),
    )

    assert isinstance(resolved.candidate, TrainingBenchmarkCandidate)
    assert resolved.candidate.input_joints == ("joint_a",)
    assert resolved.candidate.output_joints == ("joint_a",)
    assert resolved.candidate.metadata_sha256 == loaded.sha256
    assert set(resolved.candidate.artifact_hashes) == {"network", "metadata"}


def test_source_tamper_fails_before_neural_metadata_loader(tmp_path: Path) -> None:
    """Whole-run verification precedes any archive metadata or runtime loading."""

    profiles = _write_benchmark_profile(tmp_path)
    _write_motion(tmp_path / "inputs/held_out/reach")
    source, _loaded = _install_training_source(tmp_path)
    (source / "network.pt").write_bytes(b"tampered")
    called = False

    def loader(_path: Path) -> LoadedRuntimeMetadata:
        """Record an invalid early metadata-loader invocation."""

        nonlocal called
        called = True
        raise AssertionError("loader should not run")

    inputs = _resolver_inputs(tmp_path, profiles)
    inputs = BenchmarkResolverInputs(**{**inputs.__dict__, "metadata_loader": loader})
    with pytest.raises(RuntimeError, match="hash mismatch"):
        resolve_benchmark_run(BenchmarkInvocation(robot="demo", run=source), inputs)
    assert not called


def test_training_metadata_sysid_hash_must_match_authenticated_manifest(tmp_path: Path) -> None:
    """Benchmark binds SysID provenance by file hash without parsing its snapshot."""

    profiles = _write_benchmark_profile(tmp_path)
    _write_motion(tmp_path / "inputs/held_out/reach")
    source, loaded = _install_training_source(tmp_path)
    bad_hashes = loaded.metadata.hashes.model_copy(
        update={"sysid_state_sha256": SHA_A}
    )
    bad_metadata = loaded.metadata.model_copy(update={"hashes": bad_hashes})
    bad_loaded = LoadedRuntimeMetadata(bad_metadata, loaded.raw_bytes, loaded.sha256)

    with pytest.raises(ValueError, match="sysid_state_sha256 does not match"):
        resolve_benchmark_run(
            BenchmarkInvocation(robot="demo", run=source),
            _resolver_inputs(tmp_path, profiles, metadata=bad_loaded),
        )


@pytest.mark.parametrize("phase", ["sysid", "train"])
def test_timing_mismatch_is_a_host_side_hard_error(tmp_path: Path, phase: str) -> None:
    """Both candidate types must use the benchmark control timestep."""

    profiles = _write_benchmark_profile(tmp_path, control_freq=500)
    _write_motion(tmp_path / "inputs/held_out/reach")
    if phase == "sysid":
        source = _install_sysid_source(tmp_path, control_freq=250)
        inputs = _resolver_inputs(tmp_path, profiles)
    else:
        source, metadata = _install_training_source(tmp_path, sample_dt_s=0.004)
        inputs = _resolver_inputs(tmp_path, profiles, metadata=metadata)

    with pytest.raises(ValueError, match="candidate timestep .* does not match"):
        resolve_benchmark_run(BenchmarkInvocation(robot="demo", run=source), inputs)


def test_required_candidate_joints_are_checked_before_runtime(tmp_path: Path) -> None:
    """A held-out trajectory missing one owned axis fails during resolution."""

    profiles = _write_benchmark_profile(tmp_path)
    _write_motion(tmp_path / "inputs/held_out/reach", joints=("joint_b",))
    source = _install_sysid_source(tmp_path)

    with pytest.raises(ValueError, match="missing candidate joints: joint_a"):
        resolve_benchmark_run(
            BenchmarkInvocation(robot="demo", run=source),
            _resolver_inputs(tmp_path, profiles),
        )


def test_default_metadata_loader_reads_torchscript_extra_file_without_torch(tmp_path: Path) -> None:
    """The default loader reads deterministic ZIP extra files using stdlib."""

    profile_hash = canonical_sha256({"profile": 1})
    resolved_hash = canonical_sha256({"resolved": 1})
    metadata = _runtime_metadata(
        profile_sha256=profile_hash,
        resolved_config_sha256=resolved_hash,
        sysid_state_sha256=SHA_C,
    )
    raw = json.dumps(metadata.to_dict(), sort_keys=True, separators=(",", ":")).encode()
    archive = tmp_path / "network.pt"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("network/extra/metadata.json", raw)

    loaded = load_embedded_runtime_metadata(archive)

    assert loaded.metadata == metadata
    assert loaded.sha256 == sha256_hex(raw)
