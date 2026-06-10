from __future__ import annotations

from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Any

import yaml

from anchor_bench.benchmark.newton_runner import build_benchmark_command
from anchor_bench.core.paths import repo_root
from anchor_bench.core.schemas import BenchmarkRunConfig
from anchor_bench.core.schemas import ExperimentFile
from anchor_bench.data.sage import validate_sage_dir
from anchor_bench.robots.registry import get_robot
from anchor_bench.core.plan import CommandPlan


class ExperimentConfigError(ValueError):
    """Raised when an experiment config cannot be expanded safely."""


@dataclass(frozen=True)
class ExperimentConfig:
    path: Path
    robot: str
    outputs: dict[str, str]
    seeds: list[int | None]
    benchmark: dict[str, Any]
    datasets: dict[str, dict[str, Any]]
    actuator_models: dict[str, dict[str, Any]]
    analysis: dict[str, Any]


@dataclass(frozen=True)
class ExperimentBenchmarkRun:
    robot: str
    dataset: str
    motion: str
    motion_name: str
    actuator_model: str
    seed: int | None
    actuator_params: str
    output: str
    headless: bool


@dataclass(frozen=True)
class ExperimentBenchmarkBatch:
    robot: str
    dataset: str
    actuator_model: str
    seed: int | None
    actuator_params: str
    output: str
    headless: bool
    configured_num_envs: int
    batch_size: int
    batch_index: int
    runs: list[ExperimentBenchmarkRun]


@dataclass(frozen=True)
class ExperimentAnalysisRun:
    robot: str
    result_folder: str
    output: str
    seeds: list[int | None]
    datasets: list[str]
    motions: list[str] | None
    actuator_models: list[str] | None
    plots: dict[str, bool]
    sample_dt: float | None


def _as_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return repo_root() / path


def _path_text(value: str | Path) -> str:
    return str(value)


def _format_path_template(value: str, seed: int | None) -> str:
    if seed is None:
        return value.replace("{seed}", "")
    return value.format(seed=seed)


def _normalize_seeds(value: Any) -> list[int | None]:
    if value is None:
        return [None]
    if not isinstance(value, list):
        raise ExperimentConfigError("seeds must be a list")
    return [int(seed) for seed in value]


def _normalize_benchmark(value: Any) -> dict[str, Any]:
    if value is None:
        return {"num_envs": 1}
    if not isinstance(value, dict):
        raise ExperimentConfigError("benchmark must be a mapping")
    num_envs = int(value.get("num_envs", 1))
    if num_envs < 1:
        raise ExperimentConfigError("benchmark.num_envs must be >= 1")
    headless = bool(value.get("headless", False))
    return {"num_envs": num_envs, "headless": headless}


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).expanduser()
    if not config_path.is_absolute():
        config_path = repo_root() / config_path
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    try:
        ExperimentFile.from_dict(data, config_path)
    except ValueError as exc:
        raise ExperimentConfigError(str(exc)) from exc
    bundle = get_robot(str(data["robot"]))
    run_cfg = bundle.run_config
    benchmark_defaults = run_cfg.get("benchmark", {})
    analysis_defaults = run_cfg.get("analysis", {})
    outputs = {
        "benchmark": benchmark_defaults.get("output_folder", "outputs/sim2real_benchmark"),
        "analysis": analysis_defaults.get("output_dir", "outputs/sim2real_analysis"),
        **(data.get("outputs") or {}),
    }
    analysis = {
        "motions": analysis_defaults.get("motion_names", "*"),
        "sample_dt": analysis_defaults.get("sample_dt"),
        **(data.get("analysis") or {}),
    }
    return ExperimentConfig(
        path=config_path,
        robot=str(data["robot"]),
        outputs={key: str(value) for key, value in outputs.items()},
        seeds=_normalize_seeds(data.get("seeds")),
        benchmark=_normalize_benchmark(data.get("benchmark")),
        datasets={str(key): value or {} for key, value in (data.get("datasets") or {}).items()},
        actuator_models={str(key): value or {} for key, value in (data.get("actuator_models") or {}).items()},
        analysis=analysis,
    )


def _is_valid_sage_folder(path: Path) -> bool:
    try:
        validate_sage_dir(path)
    except Exception:
        return False
    return True


def _discover_motions(dataset_name: str, dataset_cfg: dict[str, Any]) -> list[tuple[str, Path]]:
    real_root_value = dataset_cfg.get("real_root")
    if real_root_value is None:
        raise ExperimentConfigError(f"dataset {dataset_name} must set real_root")
    real_root = _as_repo_path(str(real_root_value))
    motions = dataset_cfg.get("motions")
    if motions is None or motions == "*":
        if not real_root.exists():
            raise ExperimentConfigError(f"dataset {dataset_name} real_root does not exist: {real_root}")
        discovered = [(path.name, path) for path in sorted(real_root.iterdir()) if path.is_dir() and _is_valid_sage_folder(path)]
        if not discovered:
            raise ExperimentConfigError(f"dataset {dataset_name} has no valid SAGE child folders under {real_root}")
        return discovered
    if not isinstance(motions, list):
        raise ExperimentConfigError(f"dataset {dataset_name} motions must be a list or omitted")
    selected: list[tuple[str, Path]] = []
    for item in motions:
        if isinstance(item, dict):
            motion_name = str(item.get("name") or item.get("motion") or "")
            motion_path = _as_repo_path(str(item.get("path") or real_root / motion_name))
        else:
            motion_name = str(item)
            motion_path = real_root / motion_name
        if not motion_name:
            raise ExperimentConfigError(f"dataset {dataset_name} has a motion without a name")
        if not _is_valid_sage_folder(motion_path):
            raise ExperimentConfigError(f"motion {motion_name} is not a valid SAGE folder: {motion_path}")
        selected.append((motion_name, motion_path))
    return selected


def _record_dry_run_warning(message: str, warnings: list[str] | None) -> None:
    if warnings is not None and message not in warnings:
        warnings.append(message)
    print(f"warning: {message}", file=sys.stderr)


def _resolve_actuator_params(
    actuator_name: str,
    actuator_cfg: dict[str, Any],
    seed: int | None,
    dry_run: bool = False,
    warnings: list[str] | None = None,
) -> str:
    mode = str(actuator_cfg.get("benchmark_params", "fitted"))
    if mode not in {"fitted", "initial"}:
        raise ExperimentConfigError(f"actuator_model {actuator_name} benchmark_params must be fitted or initial")
    key = "fitted_params" if mode == "fitted" else "initial_params"
    value = actuator_cfg.get(key)
    if value is None:
        raise ExperimentConfigError(f"actuator_model {actuator_name} must set {key}")
    resolved = _as_repo_path(_format_path_template(str(value), seed))
    if mode == "fitted" and not resolved.exists():
        message = (
            f"best_params.yaml does not exist for actuator_model {actuator_name}, seed {seed}: {resolved}. "
            "Run sysid first or set benchmark_params: initial"
        )
        if dry_run:
            _record_dry_run_warning(message, warnings)
        else:
            raise ExperimentConfigError(message)
    if mode == "initial" and not resolved.exists():
        message = f"initial_params does not exist for actuator_model {actuator_name}: {resolved}"
        if dry_run:
            _record_dry_run_warning(message, warnings)
        else:
            raise ExperimentConfigError(message)
    return str(resolved)


def _seeded_output_root(base: str, seed: int | None, seeds: list[int | None]) -> str:
    output = _as_repo_path(base)
    if seed is not None and len(seeds) > 1:
        output = output / f"seed_{seed}"
    return str(output)


def expand_benchmark_runs(
    config: ExperimentConfig,
    dry_run: bool = False,
    warnings: list[str] | None = None,
) -> list[ExperimentBenchmarkRun]:
    runs: list[ExperimentBenchmarkRun] = []
    for seed in config.seeds:
        output = _seeded_output_root(config.outputs["benchmark"], seed, config.seeds)
        for dataset_name, dataset_cfg in config.datasets.items():
            motions = _discover_motions(dataset_name, dataset_cfg)
            for actuator_name, actuator_cfg in config.actuator_models.items():
                actuator_params = _resolve_actuator_params(actuator_name, actuator_cfg, seed, dry_run=dry_run, warnings=warnings)
                for motion_name, motion_path in motions:
                    runs.append(
                        ExperimentBenchmarkRun(
                            robot=config.robot,
                            dataset=dataset_name,
                            motion=str(motion_path),
                            motion_name=motion_name,
                            actuator_model=actuator_name,
                            seed=seed,
                            actuator_params=actuator_params,
                            output=output,
                            headless=bool(config.benchmark.get("headless", False)),
                        )
                    )
    return runs


def _batch_benchmark_runs(config: ExperimentConfig, runs: list[ExperimentBenchmarkRun]) -> list[ExperimentBenchmarkBatch]:
    num_envs = int(config.benchmark["num_envs"])
    groups: dict[tuple[str, str, str, int | None, str, str, bool], list[ExperimentBenchmarkRun]] = {}
    for run in runs:
        key = (run.robot, run.dataset, run.actuator_model, run.seed, run.actuator_params, run.output, run.headless)
        groups.setdefault(key, []).append(run)

    batches: list[ExperimentBenchmarkBatch] = []
    for key, grouped_runs in groups.items():
        robot, dataset, actuator_model, seed, actuator_params, output, headless = key
        for offset in range(0, len(grouped_runs), num_envs):
            chunk = grouped_runs[offset : offset + num_envs]
            batches.append(
                ExperimentBenchmarkBatch(
                    robot=robot,
                    dataset=dataset,
                    actuator_model=actuator_model,
                    seed=seed,
                    actuator_params=actuator_params,
                    output=output,
                    headless=headless,
                    configured_num_envs=num_envs,
                    batch_size=len(chunk),
                    batch_index=len(batches),
                    runs=chunk,
                )
            )
    return batches


def expand_benchmark_batches(
    config: ExperimentConfig,
    dry_run: bool = False,
    warnings: list[str] | None = None,
) -> list[ExperimentBenchmarkBatch]:
    return _batch_benchmark_runs(config, expand_benchmark_runs(config, dry_run=dry_run, warnings=warnings))


def _benchmark_command_for_run(run: ExperimentBenchmarkRun, dry_run: bool) -> list[str]:
    return build_benchmark_command(BenchmarkRunConfig(
        robot=run.robot,
        motion=run.motion,
        output=run.output,
        dry_run=dry_run,
        headless=run.headless,
        motion_name=run.motion_name,
        actuator_params=run.actuator_params,
        dataset=run.dataset,
        actuator_model=run.actuator_model,
        seed=run.seed,
    )).command


def build_experiment_benchmark_plan(config_path: str | Path, dry_run: bool = True) -> CommandPlan:
    config = load_experiment_config(config_path)
    warnings: list[str] = []
    runs = expand_benchmark_runs(config, dry_run=dry_run, warnings=warnings)
    batches = _batch_benchmark_runs(config, runs)
    command = ["anchor-bench", "experiment", "benchmark", str(Path(config_path))]
    if dry_run:
        command.append("--dry-run")
    return CommandPlan(
        workflow="experiment benchmark",
        robot=config.robot,
        command=command,
        dry_run=dry_run,
        notes=[f"expanded {len(runs)} benchmark run(s) into {len(batches)} batch(es) from {config.path}"],
        runs=[_benchmark_command_for_run(run, dry_run=dry_run) for run in runs],
        metadata={
            "datasets": sorted(config.datasets),
            "actuator_models": sorted(config.actuator_models),
            "seeds": [seed for seed in config.seeds if seed is not None],
            "benchmark_num_envs": config.benchmark["num_envs"],
            "benchmark_headless": config.benchmark.get("headless", False),
            "total_runs": len(runs),
            "batch_count": len(batches),
            "batch_sizes": [batch.batch_size for batch in batches],
        },
        warnings=warnings or None,
    )


def run_experiment_benchmark(config_path: str | Path, dry_run: bool = True) -> CommandPlan:
    config = load_experiment_config(config_path)
    plan = build_experiment_benchmark_plan(config_path, dry_run=dry_run)
    if dry_run:
        return plan
    batches = expand_benchmark_batches(config, dry_run=False)
    from anchor_bench.benchmark.newton_runner import BenchmarkMotion, run_benchmark_batch

    for batch in batches:
        run_benchmark_batch(
            batch.robot,
            [BenchmarkMotion(motion=run.motion, motion_name=run.motion_name) for run in batch.runs],
            batch.output,
            dry_run=False,
            actuator_params=batch.actuator_params,
            dataset=batch.dataset,
            actuator_model=batch.actuator_model,
            seed=batch.seed,
            num_envs=batch.batch_size,
            headless=batch.headless,
            batch_index=batch.batch_index,
        )
    return plan


def _selected_list(value: Any, names: list[str]) -> list[str] | None:
    if value is None or value == "*":
        return None
    if not isinstance(value, list):
        raise ExperimentConfigError("analysis selectors must be lists or '*' values")
    unknown = sorted(set(str(item) for item in value) - set(names))
    if unknown:
        raise ExperimentConfigError(f"unknown analysis selection(s): {', '.join(unknown)}")
    return [str(item) for item in value]


def _analysis_seeds(config: ExperimentConfig) -> list[int | None]:
    analysis_seeds = config.analysis.get("seeds")
    return _normalize_seeds(analysis_seeds) if analysis_seeds is not None else config.seeds


def expand_analysis_runs(config: ExperimentConfig) -> list[ExperimentAnalysisRun]:
    dataset_names = sorted(config.datasets)
    actuator_names = sorted(config.actuator_models)
    selected_datasets = _selected_list(config.analysis.get("datasets"), dataset_names) or dataset_names
    selected_actuators = _selected_list(config.analysis.get("actuator_models"), actuator_names)
    plots = {
        "joints": True,
        "end_effector": True,
        "require_end_effector": False,
        **(config.analysis.get("plots") or {}),
    }
    sample_dt_value = config.analysis.get("sample_dt")
    sample_dt = float(sample_dt_value) if sample_dt_value is not None else None

    seeds = _analysis_seeds(config)
    for dataset_name in selected_datasets:
        _discover_motions(dataset_name, config.datasets[dataset_name])
    motion_selector = config.analysis.get("motions", "*")
    if motion_selector in (None, "*"):
        motions = None
    elif isinstance(motion_selector, list):
        motions = [str(item) for item in motion_selector]
    elif isinstance(motion_selector, str):
        motions = [motion_selector]
    else:
        raise ExperimentConfigError(
            f"analysis.motions must be '*', a glob string, or a list, got {type(motion_selector).__name__}"
        )
    return [
        ExperimentAnalysisRun(
            robot=config.robot,
            result_folder=str(_as_repo_path(config.outputs["benchmark"])),
            output=str(_as_repo_path(config.outputs["analysis"])),
            seeds=seeds,
            datasets=selected_datasets,
            motions=motions,
            actuator_models=selected_actuators,
            plots={key: bool(value) for key, value in plots.items()},
            sample_dt=sample_dt,
        )
    ]


def _analysis_command_for_run(run: ExperimentAnalysisRun, dry_run: bool) -> list[str]:
    command = [
        "anchor-bench",
        "analyze",
        "--robot",
        run.robot,
        "--result-folder",
        run.result_folder,
        "--output",
        run.output,
    ]
    if dry_run:
        command.append("--dry-run")
    for motion in run.motions or []:
        command.extend(["--motion", motion])
    if run.plots.get("require_end_effector", False):
        command.append("--require-end-effector")
    return command


def build_experiment_analysis_plan(config_path: str | Path, dry_run: bool = True) -> CommandPlan:
    config = load_experiment_config(config_path)
    runs = expand_analysis_runs(config)
    command = ["anchor-bench", "experiment", "analyze", str(Path(config_path))]
    if dry_run:
        command.append("--dry-run")
    return CommandPlan(
        workflow="experiment analyze",
        robot=config.robot,
        command=command,
        dry_run=dry_run,
        notes=[f"expanded {len(runs)} analysis run(s) from {config.path}"],
        runs=[_analysis_command_for_run(run, dry_run=dry_run) for run in runs],
        metadata={
            "datasets": sorted({dataset for run in runs for dataset in run.datasets}),
            "actuator_models": sorted(config.actuator_models),
            "seeds": sorted({seed for run in runs for seed in run.seeds if seed is not None}),
            "sample_dt": runs[0].sample_dt if runs else None,
        },
    )


def run_experiment_analysis(config_path: str | Path, dry_run: bool = True) -> CommandPlan:
    config = load_experiment_config(config_path)
    runs = expand_analysis_runs(config)
    plan = build_experiment_analysis_plan(config_path, dry_run=dry_run)
    if dry_run:
        return plan
    if config.robot != "so101":
        raise ValueError("Full analysis currently supports only robot=so101")
    from anchor_bench.analysis.so101_analysis import run_so101_analysis

    for run in runs:
        run_so101_analysis(
            result_folder=run.result_folder,
            output=run.output,
            datasets=run.datasets,
            motions=run.motions,
            actuator_models=run.actuator_models,
            make_joint_plots=run.plots.get("joints", True),
            make_end_effector_plots=run.plots.get("end_effector", True),
            require_end_effector=run.plots.get("require_end_effector", False),
            seeds=[seed for seed in run.seeds if seed is not None],
            sample_dt=run.sample_dt,
        )
    return plan
