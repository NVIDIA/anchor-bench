from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from anchor_bench.analysis.runner import resolve_analysis_run_config, run_analysis
from anchor_bench.benchmark.newton_runner import resolve_benchmark_run_config, run_benchmark
from anchor_bench.core.schemas import BenchmarkRunConfig, SysIdRunConfig, TrainActuatorRunConfig
from anchor_bench.data.sage import validate_sage_dir
from anchor_bench.experiments.runner import (
    ExperimentConfigError,
    run_experiment_analysis,
    run_experiment_benchmark,
)
from anchor_bench.runtime.environment import RuntimeUnavailableError
from anchor_bench.robots.registry import get_robot, list_robots, validate_robot
from anchor_bench.sysid.runner import resolve_sysid_run_config, run_sysid
from anchor_bench.training.runner import resolve_train_actuator_run_config, run_train_actuator


def _print_plan(plan) -> int:
    print(plan.to_yaml().rstrip())
    return 0


def _require_existing_path(label: str, value: str | None, kind: str = "path") -> None:
    if value is None:
        return
    if not Path(value).exists():
        raise FileNotFoundError(f"{label} {kind} does not exist: {value}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="anchor-bench")
    sub = parser.add_subparsers(dest="command", required=True)

    robots = sub.add_parser("robots")
    robot_sub = robots.add_subparsers(dest="robot_command", required=True)
    robot_sub.add_parser("list")
    describe = robot_sub.add_parser("describe")
    describe.add_argument("robot")

    sysid = sub.add_parser("sysid")
    sysid.add_argument("--robot", required=True)
    sysid.add_argument("--data")
    sysid.add_argument("--output")
    sysid.add_argument("--headless", action="store_true")
    sysid.add_argument("--num-envs", type=int)
    sysid.add_argument("--max-iter", type=int)
    sysid.add_argument("--max-trajectory-len", type=int)
    sysid.add_argument("--seed", type=int)
    sysid.add_argument("--dry-run", action="store_true")

    train_actuator = sub.add_parser("train-actuator")
    train_actuator.add_argument("--robot", required=True)
    train_actuator.add_argument("--model", choices=("gru", "hybrid_residual"))
    train_actuator.add_argument("--data")
    train_actuator.add_argument("--joint-type")
    train_actuator.add_argument("--output")
    train_actuator.add_argument("--input-columns")
    train_actuator.add_argument("--target-column")
    train_actuator.add_argument("--skip-optuna", action="store_true", default=None)
    train_actuator.add_argument("--trials", type=int)
    train_actuator.add_argument("--trial-epochs", type=int)
    train_actuator.add_argument("--final-epochs", type=int)
    train_actuator.add_argument("--n-streams", type=int)
    train_actuator.add_argument("--noise-std", type=float)
    train_actuator.add_argument("--seed", type=int)
    train_actuator.add_argument("--dry-run", action="store_true")

    benchmark = sub.add_parser("benchmark")
    benchmark.add_argument("--robot", required=True)
    benchmark.add_argument("--motion")
    benchmark.add_argument("--output")
    benchmark.add_argument("--headless", action="store_true")
    benchmark.add_argument("--dataset")
    benchmark.add_argument("--motion-name")
    benchmark.add_argument("--actuator-model")
    benchmark.add_argument("--seed", type=int)
    benchmark.add_argument("--actuator-params")
    benchmark.add_argument("--dry-run", action="store_true")

    analyze = sub.add_parser("analyze")
    analyze.add_argument("--robot", required=True)
    analyze.add_argument("--result-folder")
    analyze.add_argument("--output")
    analyze.add_argument("--motion", action="append", dest="motions")
    analyze.add_argument("--require-end-effector", action="store_true")
    analyze.add_argument("--dry-run", action="store_true")

    experiment = sub.add_parser("experiment")
    experiment_sub = experiment.add_subparsers(dest="experiment_command", required=True)
    experiment_benchmark = experiment_sub.add_parser("benchmark")
    experiment_benchmark.add_argument("config")
    experiment_benchmark.add_argument("--dry-run", action="store_true")
    experiment_analyze = experiment_sub.add_parser("analyze")
    experiment_analyze.add_argument("config")
    experiment_analyze.add_argument("--dry-run", action="store_true")

    convert = sub.add_parser("convert")
    convert.add_argument("--robot", required=True)
    convert.add_argument("--format", required=True)
    convert.add_argument("--input")
    convert.add_argument("--output")
    convert.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)
    try:
        if args.command == "robots":
            if args.robot_command == "list":
                for robot in list_robots():
                    print(robot)
                return 0
            bundle = get_robot(args.robot)
            errors = validate_robot(args.robot)
            print(json.dumps({"robot": bundle.spec.__dict__, "errors": errors}, indent=2))
            return 0 if not errors else 1
        if args.command == "sysid":
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
            return _print_plan(run_sysid(config))

        if args.command == "train-actuator":
            config = TrainActuatorRunConfig(
                robot=args.robot,
                model=args.model,
                data=args.data,
                output=args.output,
                joint_type=args.joint_type,
                input_columns=args.input_columns,
                target_column=args.target_column,
                dry_run=args.dry_run,
                skip_optuna=args.skip_optuna,
                trials=args.trials,
                trial_epochs=args.trial_epochs,
                final_epochs=args.final_epochs,
                n_streams=args.n_streams,
                noise_std=args.noise_std,
                seed=args.seed,
            )
            resolved = resolve_train_actuator_run_config(config)
            if not args.dry_run:
                _require_existing_path("--data", resolved.data, kind="directory")
            return _print_plan(run_train_actuator(config))
        if args.command == "benchmark":
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
            return _print_plan(run_benchmark(config))
        if args.command == "analyze":
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
            return _print_plan(
                run_analysis(
                    args.robot,
                    args.result_folder,
                    args.output,
                    dry_run=args.dry_run,
                    require_end_effector=args.require_end_effector,
                    motions=args.motions,
                )
            )
        if args.command == "experiment":
            _require_existing_path("config", args.config)
            if args.experiment_command == "benchmark":
                return _print_plan(run_experiment_benchmark(args.config, dry_run=args.dry_run))
            if args.experiment_command == "analyze":
                return _print_plan(run_experiment_analysis(args.config, dry_run=args.dry_run))
        if args.command == "convert":
            get_robot(args.robot)
            if args.input:
                try:
                    validate_sage_dir(args.input)
                except Exception:
                    if not args.dry_run:
                        raise
            print(json.dumps({"workflow": "convert", "robot": args.robot, "format": args.format, "dry_run": args.dry_run}, indent=2))
            return 0
    except (RuntimeUnavailableError, ExperimentConfigError, ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
