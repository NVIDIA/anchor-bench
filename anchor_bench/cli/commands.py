from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any


def _print_plan(plan: Any) -> int:
    print(plan.to_yaml().rstrip())
    return 0


def _require_existing_path(label: str, value: str | None, kind: str = "path") -> None:
    if value is not None and not Path(value).exists():
        raise FileNotFoundError(f"{label} {kind} does not exist: {value}")


def _handle_robots(args: argparse.Namespace) -> int:
    from anchor_bench.robots.registry import get_robot, list_robots, validate_robot

    if args.robot_command == "list":
        for robot in list_robots():
            print(robot)
        return 0
    bundle = get_robot(args.robot)
    errors = validate_robot(args.robot)
    print(json.dumps({"robot": bundle.spec.__dict__, "errors": errors}, indent=2))
    return 0 if not errors else 1


def _handle_sysid(args: argparse.Namespace) -> int:
    from anchor_bench.core.schemas import SysIdRunConfig
    from anchor_bench.sysid.runner import resolve_sysid_run_config, run_resolved_sysid

    config = SysIdRunConfig(
        robot=args.robot,
        data=args.data,
        output=args.output,
        dry_run=args.dry_run,
        headless=args.headless,
        num_envs=args.num_envs,
        max_iter=args.max_iter,
        max_trajectory_len=args.max_trajectory_len,
        seed=args.seed,
    )
    resolved = resolve_sysid_run_config(config)
    if not args.dry_run:
        _require_existing_path("--data", resolved.data, kind="directory")
    return _print_plan(run_resolved_sysid(resolved))


def _handle_train(args: argparse.Namespace) -> int:
    from anchor_bench.core.run_schema import TrainingGroup
    from anchor_bench.core.schemas import TrainActuatorRunConfig
    from anchor_bench.training.runner import resolve_train_actuator_run_config, run_resolved_train_actuator

    groups = None
    if args.groups is not None:
        try:
            raw_groups = json.loads(args.groups)
        except json.JSONDecodeError as exc:
            raise ValueError(f"--groups must be valid JSON: {exc}") from exc
        if not isinstance(raw_groups, list) or not raw_groups:
            raise ValueError("--groups must be a non-empty JSON list of group objects")
        groups = [
            TrainingGroup.from_dict(item, f"--groups[{index}]") for index, item in enumerate(raw_groups)
        ]

    config = TrainActuatorRunConfig(
        robot=args.robot,
        model=args.model,
        data=args.data,
        output=args.output,
        groups=groups,
        input_joints=args.input_joints,
        output_joints=args.output_joints,
        pd_kp=args.pd_kp,
        pd_kd=args.pd_kd,
        friction_dry=args.friction_dry,
        friction_viscous=args.friction_viscous,
        input_columns=args.input_columns,
        dry_run=args.dry_run,
        skip_optuna=args.skip_optuna,
        trials=args.trials,
        trial_epochs=args.trial_epochs,
        final_epochs=args.final_epochs,
        n_streams=args.n_streams,
        noise_std=args.noise_std,
        seed=args.seed,
        sample_dt_s=args.sample_dt_s,
    )
    resolved = resolve_train_actuator_run_config(config)
    if not args.dry_run:
        _require_existing_path("--data", resolved.data, kind="directory")
    return _print_plan(run_resolved_train_actuator(resolved))


def _handle_benchmark(args: argparse.Namespace) -> int:
    from anchor_bench.benchmark.newton_runner import resolve_benchmark_run_config, run_resolved_benchmark
    from anchor_bench.core.schemas import BenchmarkRunConfig

    config = BenchmarkRunConfig(
        robot=args.robot,
        motion=args.motion,
        output=args.output,
        dry_run=args.dry_run,
        headless=args.headless,
        motion_name=args.motion_name,
        actuator_params=args.actuator_params,
        dataset=args.dataset,
        actuator_model=args.actuator_model,
        seed=args.seed,
    )
    resolved = resolve_benchmark_run_config(config)
    if not args.dry_run:
        _require_existing_path("--motion", resolved.motion, kind="directory")
        _require_existing_path("--actuator-params", resolved.actuator_params)
    return _print_plan(run_resolved_benchmark(resolved))


def _handle_analyze(args: argparse.Namespace) -> int:
    from anchor_bench.analysis.runner import resolve_analysis_run_config, run_resolved_analysis

    resolved = resolve_analysis_run_config(
        args.robot,
        args.result_folder,
        args.output,
        dry_run=args.dry_run,
        require_end_effector=args.require_end_effector,
        motions=args.motions,
    )
    if not args.dry_run:
        _require_existing_path("--result-folder", resolved.result_folder, kind="directory")
    return _print_plan(run_resolved_analysis(resolved))


def _handle_experiment(args: argparse.Namespace) -> int:
    from anchor_bench.experiments.runner import run_experiment_analysis, run_experiment_benchmark

    _require_existing_path("config", args.config)
    if args.experiment_command == "benchmark":
        return _print_plan(run_experiment_benchmark(args.config, dry_run=args.dry_run))
    return _print_plan(run_experiment_analysis(args.config, dry_run=args.dry_run))


_HANDLERS: dict[str, Callable[[argparse.Namespace], int]] = {
    "robots": _handle_robots,
    "sysid": _handle_sysid,
    "train-actuator": _handle_train,
    "benchmark": _handle_benchmark,
    "analyze": _handle_analyze,
    "experiment": _handle_experiment,
}


def dispatch(args: argparse.Namespace) -> int:
    return _HANDLERS[args.command](args)
