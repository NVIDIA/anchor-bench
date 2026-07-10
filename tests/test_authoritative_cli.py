"""Focused tests for the authoritative phase-oriented public CLI."""

from __future__ import annotations

import builtins
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from anchor_bench.cli.main import main
from anchor_bench.cli.parser import build_parser
from anchor_bench.core.plan import CommandPlan


_HEAVY_PACKAGES = {"torch", "newton", "warp", "isaacsim", "isaaclab"}


def _guard_heavy_imports(monkeypatch: Any) -> None:
    """Fail immediately if a dry-run handler imports a simulator package."""

    real_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        """Delegate every import except a forbidden heavy runtime."""

        if name.split(".", maxsplit=1)[0] in _HEAVY_PACKAGES:
            raise AssertionError(f"dry run imported heavy runtime {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def test_parser_exposes_authoritative_command_contract() -> None:
    """Parse complete replacements, mapping shorthand, and explicit promotion."""

    sysid = build_parser().parse_args(
        [
            "sysid",
            "--robot",
            "so101",
            "--profile",
            "default",
            "--joints",
            "Rotation,Pitch",
            "--optimizer",
            '{"type":"cmaes","sigma":0.25,"max_iterations":12,"epsilon":0.01}',
            "--dry-run",
        ]
    )
    assert sysid.joints == ("Rotation", "Pitch")
    assert sysid.optimizer["max_iterations"] == 12

    training = build_parser().parse_args(
        [
            "train-actuator",
            "--robot",
            "g1",
            "--input-joints",
            "right_shoulder_pitch,right_elbow",
            "--output-joints",
            "right_elbow",
            "--no-skip-optuna",
            "--dry-run",
        ]
    )
    assert training.input_joints == ("right_shoulder_pitch", "right_elbow")
    assert training.output_joints == ("right_elbow",)
    assert training.skip_optuna is False

    promotion = build_parser().parse_args(
        [
            "promote",
            "train",
            "--robot",
            "g1",
            "--run",
            "outputs/train/g1/right-arm/seed-42/hash",
            "--benchmark-run",
            "outputs/benchmark/g1/train/default/source/hash",
            "--dry-run",
        ]
    )
    assert promotion.promote_phase == "train"
    assert promotion.dry_run is True


def test_unknown_robot_is_a_clean_user_error(capsys: Any) -> None:
    """Return status 2 instead of exposing a registry traceback."""

    assert main(["sysid", "--robot", "missing-robot", "--dry-run"]) == 2
    assert "unknown robot" in capsys.readouterr().err


def test_sysid_and_training_dry_runs_use_profiles_without_runtime_imports(
    monkeypatch: Any,
    capsys: Any,
) -> None:
    """Resolve real profiles and structured overrides on a host-only Python."""

    _guard_heavy_imports(monkeypatch)
    assert main(["sysid", "--robot", "so101", "--seed", "7", "--dry-run"]) == 0
    sysid = yaml.safe_load(capsys.readouterr().out)
    assert sysid["workflow"] == "sysid"
    assert sysid["metadata"]["seed"] == 7
    assert "run_input_sha256" not in sysid["metadata"]
    assert "run_dir" not in sysid["metadata"]
    assert "preview_input_sha256" in sysid["metadata"]
    assert sysid["result"]["status"] == "dry-run"
    assert "run_input_sha256" not in sysid["result"]
    assert "run_dir" not in sysid["result"]

    assert main(
        [
            "train-actuator",
            "--robot",
            "so101",
            "--input-joints",
            "Rotation,Pitch",
            "--output-joints",
            "Pitch",
            "--seed",
            "11",
            "--dry-run",
        ]
    ) == 0
    training = yaml.safe_load(capsys.readouterr().out)
    config = training["metadata"]["resolved_config"]
    assert training["workflow"] == "train-actuator"
    assert config["fit"]["seed"] == 11
    assert config["mappings"] == [
        {
            "input_joints": ["Rotation", "Pitch"],
            "output_joints": ["Pitch"],
        }
    ]
    assert training["result"]["status"] == "dry-run"


def test_dry_run_context_uses_manifest_preview_and_exact_source(monkeypatch: Any) -> None:
    """Avoid resolving a USD while still binding the current Anchor source."""

    from anchor_bench.cli.context import sysid_resolver_inputs

    _guard_heavy_imports(monkeypatch)
    inputs = sysid_resolver_inputs("g1", dry_run=True)
    assert inputs.asset_fingerprint["kind"] == "runtime_asset_preview"
    assert inputs.asset_fingerprint["runtime_usd"] == "g1.usd"
    assert len(inputs.asset_fingerprint["preview_sha256"]) == 64
    assert "sha256" not in inputs.asset_fingerprint
    assert len(inputs.robot_identity_fingerprint["raw_sha256"]) == 64
    assert len(inputs.robot_identity_fingerprint["canonical_sha256"]) == 64
    assert len(inputs.source_fingerprint["source_sha256"]) == 64
    assert len(inputs.source_fingerprint["git_commit"]) == 40


def test_deploy_promotion_dry_run_does_not_write(monkeypatch: Any, capsys: Any) -> None:
    """Materialize the default PD composition without creating deploy state."""

    _guard_heavy_imports(monkeypatch)
    path = Path("configs/robots/so101/deploy/default.yaml")
    before = path.read_bytes() if path.exists() else None
    assert main(["promote", "deploy", "--robot", "so101", "--dry-run"]) == 0
    document = yaml.safe_load(capsys.readouterr().out)
    after = path.read_bytes() if path.exists() else None
    assert before == after
    assert document["workflow"] == "promote-deploy"
    assert document["dry_run"] is True
    assert document["ownership"]["Rotation"] == "so101_all"


def test_benchmark_handler_forwards_exact_candidate_and_overrides(
    tmp_path: Path,
    monkeypatch: Any,
    capsys: Any,
) -> None:
    """Wire benchmark CLI fields to the immutable resolver and executor APIs."""

    from anchor_bench.benchmark import phase as benchmark_phase
    from anchor_bench.cli import commands, context

    source = tmp_path / "outputs/sysid/demo/default/seed-1/hash"
    source.mkdir(parents=True)
    captured: dict[str, Any] = {}

    def fake_resolve(invocation: Any, inputs: Any) -> Any:
        """Capture the typed invocation and return a sentinel resolved run."""

        captured["invocation"] = invocation
        captured["inputs"] = inputs
        return "resolved"

    def fake_execute(resolved: Any, backend: Any) -> Any:
        """Capture the lazy backend and return a printable dry-run outcome."""

        captured["resolved"] = resolved
        captured["backend"] = backend
        return SimpleNamespace(
            plan=CommandPlan(
                workflow="benchmark",
                robot="demo",
                command=["anchor-bench", "benchmark"],
                dry_run=True,
                notes=[],
            ),
            run_dir=tmp_path / "outputs/benchmark/result",
            run_input_sha256="a" * 64,
            dry_run=True,
            no_op=False,
            manifest=None,
        )

    monkeypatch.setattr(commands, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(context, "benchmark_resolver_inputs", lambda robot, dry_run: "inputs")
    monkeypatch.setattr(benchmark_phase, "resolve_benchmark_run", fake_resolve)
    monkeypatch.setattr(benchmark_phase, "execute_benchmark_run", fake_execute)
    args = build_parser().parse_args(
        [
            "benchmark",
            "--robot",
            "demo",
            "--run",
            str(source),
            "--motion",
            "held-out-a",
            "--motion",
            "held-out-b",
            "--output-root",
            "outputs/custom-benchmarks",
            "--headless",
            "--dry-run",
        ]
    )
    assert commands.dispatch(args) == 0
    capsys.readouterr()
    invocation = captured["invocation"]
    assert invocation.run == source
    assert invocation.motions == ["held-out-a", "held-out-b"]
    assert invocation.output_root == "outputs/custom-benchmarks"
    assert invocation.headless is True
    assert captured["resolved"] == "resolved"
    assert captured["backend"] is commands._benchmark_backend


def test_public_phase_exports_are_host_safe() -> None:
    """Expose authoritative APIs without loading heavy runtime packages."""

    import anchor_bench.benchmark as benchmark
    import anchor_bench.deploy as deploy
    import anchor_bench.sysid as sysid
    import anchor_bench.training as training

    assert sysid.resolve_sysid_run
    assert training.resolve_training_run
    assert benchmark.resolve_benchmark_run
    assert deploy.promote_deploy
    assert not _HEAVY_PACKAGES.intersection(sys.modules)


def test_sysid_promotion_dry_run_returns_proposal_without_writing(tmp_path: Path) -> None:
    """Preview a sparse accepted-state update while preserving exact bytes."""

    from anchor_bench.sysid.promotion import promote_sysid_run
    from anchor_bench.sysid.runner import SysIdInvocation, execute_sysid_run, resolve_sysid_run
    from tests.test_sysid_execution import _execution_result, _setup

    inputs, accepted = _setup(tmp_path)
    resolved = resolve_sysid_run(
        SysIdInvocation(robot="demo", dry_run=False),
        inputs,
    )
    execute_sysid_run(resolved, lambda _: _execution_result())
    before = accepted.read_bytes()
    outcome = promote_sysid_run(
        resolved.run_dir,
        accepted,
        repository_root=tmp_path,
        dry_run=True,
    )
    assert outcome.dry_run is True
    assert outcome.no_op is False
    assert outcome.state.revision == 1
    assert accepted.read_bytes() == before
