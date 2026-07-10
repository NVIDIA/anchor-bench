"""Public command handlers for immutable phase runs and promotion."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar

import yaml

from anchor_bench.core.hashing import contained_path
from anchor_bench.core.paths import accepted_state_path, repo_root


ConfigT = TypeVar("ConfigT")


def _print_run_outcome(outcome: Any) -> int:
    """Print a command plan followed by immutable execution status."""

    print(outcome.plan.to_yaml().rstrip())
    status = "dry-run" if outcome.dry_run else "no-op" if outcome.no_op else "completed"
    result: dict[str, Any] = {
        "status": status,
        "run_dir": str(outcome.run_dir),
        "run_input_sha256": outcome.run_input_sha256,
    }
    if outcome.dry_run:
        result["identity"] = "preview"
        result["identity_note"] = (
            "runtime asset and unavailable dataset bytes are fingerprinted only "
            "for real execution"
        )
    if outcome.manifest is not None:
        result["run_manifest_sha256"] = outcome.manifest.run_manifest_sha256
    print(yaml.safe_dump({"result": result}, sort_keys=False).rstrip())
    return 0


def _print_promotion(document: Mapping[str, Any]) -> int:
    """Print one explicit promotion result as stable YAML."""

    print(yaml.safe_dump(dict(document), sort_keys=False).rstrip())
    return 0


def _typed_config(model: type[ConfigT], value: Mapping[str, Any] | None, label: str) -> ConfigT | None:
    """Validate one optional JSON object through a strict config model."""

    if value is None:
        return None
    parser = getattr(model, "from_dict")
    return parser(value, label)


def _contained_input(value: str, *, label: str) -> Path:
    """Resolve one existing user path inside the repository."""

    try:
        return contained_path(repo_root(), value, must_exist=True)
    except ValueError as error:
        raise ValueError(f"{label} is invalid: {error}") from error


def _optional_state(value: str | None, *, default_literal: bool) -> str | Path | None:
    """Normalize an omitted/default state selector or authenticate its path."""

    if value is None:
        return None
    if value == "default":
        return "default" if default_literal else None
    return _contained_input(value, label="state path")


def _benchmark_paths(values: Sequence[str]) -> tuple[Path, ...]:
    """Contain optional benchmark evidence paths in caller order."""

    return tuple(_contained_input(value, label="--benchmark-run") for value in values)


def _sysid_parameters(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Validate a complete SysID parameter-spec replacement."""

    if value is None:
        return None
    from anchor_bench.sysid.schema import SysIdParameterSpec

    return {
        str(name): SysIdParameterSpec.from_dict(spec, f"--parameters.{name}")
        for name, spec in value.items()
    }


def _training_mappings(value: Sequence[Any] | None) -> tuple[Any, ...] | None:
    """Validate a complete ordered training-mapping replacement."""

    if value is None:
        return None
    from anchor_bench.training.schema import TrainingMapping

    return tuple(
        TrainingMapping.from_dict(item, f"--mappings[{index}]")
        for index, item in enumerate(value)
    )


def _training_features(value: Sequence[Any] | None) -> tuple[Any, ...] | None:
    """Validate a complete ordered feature-block replacement."""

    if value is None:
        return None
    from anchor_bench.training.schema import TrainingFeature

    return tuple(
        TrainingFeature.from_dict(item, f"--input-features[{index}]")
        for index, item in enumerate(value)
    )


def _handle_robots(args: argparse.Namespace) -> int:
    """List or describe declarative robot identities."""

    from anchor_bench.robots.registry import get_robot, list_robots, validate_robot

    if args.robot_command == "list":
        for robot in list_robots():
            print(robot)
        return 0
    bundle = get_robot(args.robot)
    errors = validate_robot(args.robot)
    print(
        json.dumps(
            {"robot": bundle.spec.to_dict(), "errors": errors},
            indent=2,
        )
    )
    return 0 if not errors else 1


def _handle_sysid(args: argparse.Namespace) -> int:
    """Resolve and optionally execute one immutable SysID candidate."""

    from anchor_bench.cli.context import sysid_resolver_inputs
    from anchor_bench.sysid.runner import (
        SysIdInvocation,
        execute_sysid_run,
        resolve_sysid_run,
    )
    from anchor_bench.sysid.schema import (
        SysIdLossConfig,
        SysIdOptimizerConfig,
        SysIdSimulationConfig,
    )

    invocation = SysIdInvocation(
        robot=args.robot,
        profile=args.profile,
        initial_state=_optional_state(args.initial_state, default_literal=False),
        replicate=args.replicate,
        dry_run=args.dry_run,
        data=args.data,
        joints=args.joints,
        simulation=_typed_config(SysIdSimulationConfig, args.simulation, "--simulation"),
        num_envs=args.num_envs,
        buffer_time=args.buffer_time,
        seed=args.seed,
        optimizer=_typed_config(SysIdOptimizerConfig, args.optimizer, "--optimizer"),
        parameters=_sysid_parameters(args.parameters),
        loss=_typed_config(SysIdLossConfig, args.loss, "--loss"),
    )
    resolved = resolve_sysid_run(
        invocation,
        sysid_resolver_inputs(args.robot, dry_run=args.dry_run),
    )
    backend = None
    if not args.dry_run:
        from anchor_bench.sysid.runtime import optimize_resolved_sysid

        backend = optimize_resolved_sysid
    return _print_run_outcome(execute_sysid_run(resolved, backend))


def _handle_train(args: argparse.Namespace) -> int:
    """Resolve and optionally execute one immutable GRU training candidate."""

    from anchor_bench.cli.context import training_resolver_inputs
    from anchor_bench.training.runner import (
        TrainingInvocation,
        execute_training_run,
        resolve_training_run,
    )
    from anchor_bench.training.schema import TrainingFit, TrainingNetwork

    invocation = TrainingInvocation(
        robot=args.robot,
        profile=args.profile,
        sysid_state=_optional_state(args.sysid_state, default_literal=True),
        replicate=args.replicate,
        dry_run=args.dry_run,
        data=args.data,
        sample_dt_s=args.sample_dt_s,
        network=_typed_config(TrainingNetwork, args.network, "--network"),
        target=args.target,
        mappings=_training_mappings(args.mappings),
        input_joints=args.input_joints,
        output_joints=args.output_joints,
        input_features=_training_features(args.input_features),
        fit=_typed_config(TrainingFit, args.fit, "--fit"),
        seed=args.seed,
        skip_optuna=args.skip_optuna,
        trials=args.trials,
        trial_epochs=args.trial_epochs,
        final_epochs=args.final_epochs,
        noise_std=args.noise_std,
    )
    resolved = resolve_training_run(
        invocation,
        training_resolver_inputs(args.robot, dry_run=args.dry_run),
    )
    return _print_run_outcome(execute_training_run(resolved))


def _benchmark_backend(resolved: Any) -> Any:
    """Cross into the lazy Newton replay adapter for a fresh benchmark."""

    from anchor_bench.benchmark.replay import run_benchmark_replay

    return run_benchmark_replay(resolved)


def _handle_benchmark(args: argparse.Namespace) -> int:
    """Resolve and optionally replay one exact immutable candidate."""

    from anchor_bench.benchmark.phase import (
        BenchmarkInvocation,
        execute_benchmark_run,
        resolve_benchmark_run,
    )
    from anchor_bench.benchmark.schema import BenchmarkSimulationConfig
    from anchor_bench.cli.context import benchmark_resolver_inputs

    invocation = BenchmarkInvocation(
        robot=args.robot,
        run=_contained_input(args.run, label="--run"),
        profile=args.profile,
        dry_run=args.dry_run,
        headless=args.headless,
        data=args.data,
        motions=args.motions,
        simulation=_typed_config(
            BenchmarkSimulationConfig,
            args.simulation,
            "--simulation",
        ),
        num_envs=args.num_envs,
        buffer_time=args.buffer_time,
        sync_initial_state=args.sync_initial_state,
        signals=args.signals,
        output_root=args.output_root,
    )
    resolved = resolve_benchmark_run(
        invocation,
        benchmark_resolver_inputs(args.robot, dry_run=args.dry_run),
    )
    return _print_run_outcome(
        execute_benchmark_run(resolved, _benchmark_backend)
    )


def _handle_promote_sysid(args: argparse.Namespace) -> int:
    """Validate and explicitly promote one immutable SysID candidate."""

    from anchor_bench.sysid.promotion import promote_sysid_run

    root = repo_root()
    outcome = promote_sysid_run(
        _contained_input(args.run, label="--run"),
        accepted_state_path(args.robot, "sysid"),
        repository_root=root,
        benchmark_runs=_benchmark_paths(args.benchmark_run),
        dry_run=args.dry_run,
    )
    return _print_promotion(
        {
            "workflow": "promote-sysid",
            "robot": args.robot,
            "dry_run": args.dry_run,
            "no_op": outcome.no_op,
            "accepted_state": str(outcome.accepted_state),
            "revision": outcome.state.revision,
            "source_manifest_sha256": outcome.source_manifest_sha256,
        }
    )


def _handle_promote_train(args: argparse.Namespace) -> int:
    """Validate and explicitly promote one immutable training candidate."""

    from anchor_bench.training.promotion import promote_training_run

    root = repo_root()
    outcome = promote_training_run(
        _contained_input(args.run, label="--run"),
        accepted_state_path(args.robot, "train"),
        repository_root=root,
        benchmark_runs=_benchmark_paths(args.benchmark_run),
        dry_run=args.dry_run,
    )
    return _print_promotion(
        {
            "workflow": "promote-train",
            "robot": args.robot,
            "dry_run": args.dry_run,
            "no_op": outcome.no_op,
            "accepted_catalog": str(outcome.accepted_catalog),
            "revision": outcome.catalog.revision,
            "model": outcome.model_key,
            "source_manifest_sha256": outcome.source_manifest_sha256,
        }
    )


def _handle_promote_deploy(args: argparse.Namespace) -> int:
    """Freeze accepted sources into the selected deployment composition."""

    from anchor_bench.cli.context import deploy_resolver_inputs
    from anchor_bench.deploy.promotion import promote_deploy

    outcome = promote_deploy(
        deploy_resolver_inputs(args.robot),
        profile_name=args.profile,
        dry_run=args.dry_run,
    )
    return _print_promotion(
        {
            "workflow": "promote-deploy",
            "robot": args.robot,
            "profile": args.profile,
            "dry_run": outcome.dry_run,
            "no_op": outcome.no_op,
            "accepted_state": str(outcome.accepted_state),
            "ownership": outcome.state.ownership,
        }
    )


def _handle_promote(args: argparse.Namespace) -> int:
    """Dispatch one explicit phase promotion without executing phase work."""

    handlers = {
        "sysid": _handle_promote_sysid,
        "train": _handle_promote_train,
        "deploy": _handle_promote_deploy,
    }
    return handlers[args.promote_phase](args)


_HANDLERS: dict[str, Callable[[argparse.Namespace], int]] = {
    "robots": _handle_robots,
    "sysid": _handle_sysid,
    "train-actuator": _handle_train,
    "benchmark": _handle_benchmark,
    "promote": _handle_promote,
}


def dispatch(args: argparse.Namespace) -> int:
    """Dispatch one parsed top-level command."""

    return _HANDLERS[args.command](args)


__all__ = ["dispatch"]
