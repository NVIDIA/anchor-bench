from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path

import yaml

from anchor_bench.benchmark.newton_runner import (
    ResolvedBenchmarkRunConfig,
    build_resolved_benchmark_command,
)
from anchor_bench.core.experiment_schema import (
    AnalysisSelector,
    ExperimentActuatorModel,
    ExperimentDataset,
    ExperimentFile,
)
from anchor_bench.core.paths import repo_root
from anchor_bench.core.plan import CommandPlan
from anchor_bench.data.sage import is_sage_motion, validate_sage_dir
from anchor_bench.robots.registry import get_robot


class ExperimentConfigError(ValueError):
    """Raised when an experiment config cannot be expanded safely."""


@dataclass(frozen=True)
class ResolvedExperimentOutputs:
    benchmark: str
    analysis: str


@dataclass(frozen=True)
class ResolvedExperimentBenchmark:
    num_envs: int
    headless: bool


@dataclass(frozen=True)
class ResolvedAnalysisPlots:
    joints: bool
    end_effector: bool
    require_end_effector: bool


@dataclass(frozen=True)
class ResolvedExperimentAnalysis:
    seeds: list[int] | None
    datasets: AnalysisSelector
    motions: AnalysisSelector
    actuator_models: AnalysisSelector
    plots: ResolvedAnalysisPlots
    sample_dt: float | None


@dataclass(frozen=True)
class ResolvedExperimentConfig:
    path: Path
    robot: str
    outputs: ResolvedExperimentOutputs
    seeds: list[int | None]
    benchmark: ResolvedExperimentBenchmark
    datasets: dict[str, ExperimentDataset]
    actuator_models: dict[str, ExperimentActuatorModel]
    analysis: ResolvedExperimentAnalysis


ExperimentBenchmarkRun = ResolvedBenchmarkRunConfig


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
    plots: ResolvedAnalysisPlots
    sample_dt: float | None


def _as_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return repo_root() / path


def _format_path_template(value: str, seed: int | None) -> str:
    if seed is None:
        return value.replace("{seed}", "")
    return value.format(seed=seed)


def _normalize_seeds(value: list[int] | None) -> list[int | None]:
    return [None] if value is None else list(value)


def _resolved_plots(experiment: ExperimentFile) -> ResolvedAnalysisPlots:
    plots = experiment.analysis.plots
    return ResolvedAnalysisPlots(
        joints=True if plots is None or plots.joints is None else plots.joints,
        end_effector=True if plots is None or plots.end_effector is None else plots.end_effector,
        require_end_effector=(
            False
            if plots is None or plots.require_end_effector is None
            else plots.require_end_effector
        ),
    )


def _analysis_motion_default(value: object) -> AnalysisSelector:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, list):
        return [str(item) for item in value]
    raise ExperimentConfigError("analysis.motion_names default must be a string or list")


def load_experiment_config(path: str | Path) -> ResolvedExperimentConfig:
    config_path = Path(path).expanduser()
    if not config_path.is_absolute():
        config_path = repo_root() / config_path
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    try:
        experiment = ExperimentFile.from_dict(data, config_path)
    except ValueError as exc:
        raise ExperimentConfigError(str(exc)) from exc

    run_config = get_robot(experiment.robot).run_config
    benchmark_defaults = run_config.benchmark
    analysis_defaults = run_config.analysis
    benchmark_output = experiment.outputs.benchmark or str(
        benchmark_defaults.output_folder or "outputs/sim2real_benchmark"
    )
    analysis_output = experiment.outputs.analysis or str(
        analysis_defaults.output_dir or "outputs/sim2real_analysis"
    )
    sample_dt_value = analysis_defaults.sample_dt
    sample_dt = float(sample_dt_value) if sample_dt_value is not None else None
    motions = experiment.analysis.motions
    if motions is None:
        motions = _analysis_motion_default(
            analysis_defaults.motion_names
            if analysis_defaults.motion_names is not None
            else "*"
        )

    return ResolvedExperimentConfig(
        path=config_path,
        robot=experiment.robot,
        outputs=ResolvedExperimentOutputs(
            benchmark=benchmark_output,
            analysis=analysis_output,
        ),
        seeds=_normalize_seeds(experiment.seeds),
        benchmark=ResolvedExperimentBenchmark(
            num_envs=(
                experiment.benchmark.num_envs
                if experiment.benchmark.num_envs is not None
                else 1
            ),
            headless=(
                experiment.benchmark.headless
                if experiment.benchmark.headless is not None
                else False
            ),
        ),
        datasets=experiment.datasets,
        actuator_models=experiment.actuator_models,
        analysis=ResolvedExperimentAnalysis(
            seeds=experiment.analysis.seeds,
            datasets=experiment.analysis.datasets,
            motions=motions,
            actuator_models=experiment.analysis.actuator_models,
            plots=_resolved_plots(experiment),
            sample_dt=sample_dt,
        ),
    )


def _discover_motions(dataset_name: str, dataset_cfg: ExperimentDataset) -> list[tuple[str, Path]]:
    real_root = _as_repo_path(dataset_cfg.real_root)
    motions = dataset_cfg.motions
    if motions is None or motions == "*":
        if not real_root.exists():
            raise ExperimentConfigError(f"dataset {dataset_name} real_root does not exist: {real_root}")
        discovered = [
            (path.name, path)
            for path in sorted(real_root.iterdir())
            if _is_valid_sage_motion(path)
        ]
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
        try:
            validate_sage_dir(motion_path)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            raise ExperimentConfigError(f"motion {motion_name} is not a valid SAGE folder: {exc}") from exc
        selected.append((motion_name, motion_path))
    return selected


def _is_valid_sage_motion(path: Path) -> bool:
    if not is_sage_motion(path):
        return False
    try:
        validate_sage_dir(path)
    except (OSError, ValueError, yaml.YAMLError):
        return False
    return True


def _record_dry_run_warning(message: str, warnings: list[str] | None) -> None:
    if warnings is not None and message not in warnings:
        warnings.append(message)
    print(f"warning: {message}", file=sys.stderr)


def _resolve_actuator_params(
    actuator_name: str,
    actuator_cfg: ExperimentActuatorModel,
    seed: int | None,
    dry_run: bool = False,
    warnings: list[str] | None = None,
) -> str:
    mode = actuator_cfg.benchmark_params
    if mode not in {"fitted", "initial"}:
        raise ExperimentConfigError(f"actuator_model {actuator_name} benchmark_params must be fitted or initial")
    key = "fitted_params" if mode == "fitted" else "initial_params"
    value = getattr(actuator_cfg, key)
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
    config: ResolvedExperimentConfig,
    dry_run: bool = False,
    warnings: list[str] | None = None,
) -> list[ExperimentBenchmarkRun]:
    runs: list[ExperimentBenchmarkRun] = []
    for seed in config.seeds:
        output = _seeded_output_root(config.outputs.benchmark, seed, config.seeds)
        for dataset_name, dataset_cfg in config.datasets.items():
            motions = _discover_motions(dataset_name, dataset_cfg)
            for actuator_name, actuator_cfg in config.actuator_models.items():
                actuator_params = _resolve_actuator_params(actuator_name, actuator_cfg, seed, dry_run=dry_run, warnings=warnings)
                for motion_name, motion_path in motions:
                    runs.append(
                        ResolvedBenchmarkRunConfig(
                            robot=config.robot,
                            dataset=dataset_name,
                            motion=str(motion_path),
                            motion_name=motion_name,
                            actuator_model=actuator_name,
                            seed=seed,
                            actuator_params=actuator_params,
                            output=output,
                            headless=bool(config.benchmark.headless),
                            dry_run=dry_run,
                        )
                    )
    return runs


def _batch_benchmark_runs(config: ResolvedExperimentConfig, runs: list[ExperimentBenchmarkRun]) -> list[ExperimentBenchmarkBatch]:
    num_envs = int(config.benchmark.num_envs)
    groups: dict[tuple[str, str, str, int | None, str, str, bool], list[ExperimentBenchmarkRun]] = {}
    for run in runs:
        if run.dataset is None or run.actuator_model is None or run.actuator_params is None:
            raise ExperimentConfigError("expanded benchmark runs require dataset, actuator model, and parameters")
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
    config: ResolvedExperimentConfig,
    dry_run: bool = False,
    warnings: list[str] | None = None,
) -> list[ExperimentBenchmarkBatch]:
    return _batch_benchmark_runs(config, expand_benchmark_runs(config, dry_run=dry_run, warnings=warnings))


def _benchmark_command_for_run(run: ExperimentBenchmarkRun, dry_run: bool) -> list[str]:
    return build_resolved_benchmark_command(replace(run, dry_run=dry_run)).command


def _build_experiment_benchmark_plan(
    config_path: str | Path,
    config: ResolvedExperimentConfig,
    runs: list[ExperimentBenchmarkRun],
    warnings: list[str],
    dry_run: bool,
) -> CommandPlan:
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
            "benchmark_num_envs": config.benchmark.num_envs,
            "benchmark_headless": config.benchmark.headless,
            "total_runs": len(runs),
            "batch_count": len(batches),
            "batch_sizes": [batch.batch_size for batch in batches],
        },
        warnings=warnings or None,
    )


def build_experiment_benchmark_plan(config_path: str | Path, dry_run: bool = True) -> CommandPlan:
    config = load_experiment_config(config_path)
    warnings: list[str] = []
    runs = expand_benchmark_runs(config, dry_run=dry_run, warnings=warnings)
    return _build_experiment_benchmark_plan(config_path, config, runs, warnings, dry_run)


def run_experiment_benchmark(config_path: str | Path, dry_run: bool = True) -> CommandPlan:
    config = load_experiment_config(config_path)
    warnings: list[str] = []
    runs = expand_benchmark_runs(config, dry_run=dry_run, warnings=warnings)
    plan = _build_experiment_benchmark_plan(config_path, config, runs, warnings, dry_run)
    if dry_run:
        return plan
    batches = _batch_benchmark_runs(config, runs)
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


def _selected_list(value: AnalysisSelector, names: list[str]) -> list[str] | None:
    if value is None or value == "*":
        return None
    if not isinstance(value, list):
        raise ExperimentConfigError("analysis selectors must be lists or '*' values")
    unknown = sorted(set(str(item) for item in value) - set(names))
    if unknown:
        raise ExperimentConfigError(f"unknown analysis selection(s): {', '.join(unknown)}")
    return [str(item) for item in value]


def _analysis_seeds(config: ResolvedExperimentConfig) -> list[int | None]:
    analysis_seeds = config.analysis.seeds
    return _normalize_seeds(analysis_seeds) if analysis_seeds is not None else config.seeds


def expand_analysis_runs(config: ResolvedExperimentConfig) -> list[ExperimentAnalysisRun]:
    dataset_names = sorted(config.datasets)
    actuator_names = sorted(config.actuator_models)
    selected_datasets = _selected_list(config.analysis.datasets, dataset_names) or dataset_names
    selected_actuators = _selected_list(config.analysis.actuator_models, actuator_names)
    plots = config.analysis.plots
    sample_dt = config.analysis.sample_dt

    seeds = _analysis_seeds(config)
    for dataset_name in selected_datasets:
        _discover_motions(dataset_name, config.datasets[dataset_name])
    motion_selector = config.analysis.motions
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
            result_folder=str(_as_repo_path(config.outputs.benchmark)),
            output=str(_as_repo_path(config.outputs.analysis)),
            seeds=seeds,
            datasets=selected_datasets,
            motions=motions,
            actuator_models=selected_actuators,
            plots=plots,
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
    if run.plots.require_end_effector:
        command.append("--require-end-effector")
    return command


def _build_experiment_analysis_plan(
    config_path: str | Path,
    config: ResolvedExperimentConfig,
    runs: list[ExperimentAnalysisRun],
    dry_run: bool,
) -> CommandPlan:
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


def build_experiment_analysis_plan(config_path: str | Path, dry_run: bool = True) -> CommandPlan:
    config = load_experiment_config(config_path)
    runs = expand_analysis_runs(config)
    return _build_experiment_analysis_plan(config_path, config, runs, dry_run)


def run_experiment_analysis(config_path: str | Path, dry_run: bool = True) -> CommandPlan:
    config = load_experiment_config(config_path)
    runs = expand_analysis_runs(config)
    plan = _build_experiment_analysis_plan(config_path, config, runs, dry_run)
    if dry_run:
        return plan
    from anchor_bench.analysis.runtime import run_robot_analysis

    for run in runs:
        run_robot_analysis(
            config.robot,
            result_folder=run.result_folder,
            output=run.output,
            datasets=run.datasets,
            motions=run.motions,
            actuator_models=run.actuator_models,
            make_joint_plots=run.plots.joints,
            make_end_effector_plots=run.plots.end_effector,
            require_end_effector=run.plots.require_end_effector,
            seeds=[seed for seed in run.seeds if seed is not None],
            sample_dt=run.sample_dt,
        )
    return plan
