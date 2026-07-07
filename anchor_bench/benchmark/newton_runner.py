from __future__ import annotations

from dataclasses import dataclass

from anchor_bench.core.plan import CommandPlan
from anchor_bench.core.run_config import resolve_repo_path_text, resolve_required_text
from anchor_bench.core.schemas import BenchmarkRunConfig
from anchor_bench.robots.registry import get_robot


@dataclass(frozen=True)
class BenchmarkMotion:
    motion: str
    motion_name: str | None = None


@dataclass(frozen=True)
class ResolvedBenchmarkRunConfig:
    robot: str
    motion: str
    output: str
    dry_run: bool
    headless: bool
    motion_name: str | None
    actuator_params: str | None
    dataset: str | None
    actuator_model: str | None
    seed: int | None


def resolve_benchmark_run_config(config: BenchmarkRunConfig) -> ResolvedBenchmarkRunConfig:
    bundle = get_robot(config.robot)
    benchmark_cfg = bundle.run_config.get("benchmark", {})
    motion = resolve_required_text(config.motion, benchmark_cfg.get("motion_files"), None, "--motion")
    output = resolve_required_text(config.output, benchmark_cfg.get("output_folder"), "outputs/sim2real_benchmark", "--output")
    return ResolvedBenchmarkRunConfig(
        robot=config.robot,
        motion=resolve_repo_path_text(motion),
        output=resolve_repo_path_text(output),
        dry_run=config.dry_run,
        headless=config.headless,
        motion_name=config.motion_name or benchmark_cfg.get("motion_name"),
        actuator_params=resolve_repo_path_text(config.actuator_params) if config.actuator_params is not None else None,
        dataset=config.dataset or benchmark_cfg.get("dataset"),
        actuator_model=config.actuator_model,
        seed=config.seed,
    )


def build_benchmark_command(config: BenchmarkRunConfig) -> CommandPlan:
    resolved = resolve_benchmark_run_config(config)
    command = [
        "anchor-bench", "benchmark",
        "--robot", resolved.robot,
        "--motion", resolved.motion,
        "--output", resolved.output,
    ]
    if resolved.headless:
        command.append("--headless")
    if resolved.dataset is not None:
        command.extend(["--dataset", resolved.dataset])
    if resolved.motion_name is not None:
        command.extend(["--motion-name", resolved.motion_name])
    if resolved.actuator_model is not None:
        command.extend(["--actuator-model", resolved.actuator_model])
    if resolved.seed is not None:
        command.extend(["--seed", str(resolved.seed)])
    if resolved.actuator_params is not None:
        command.extend(["--actuator-params", resolved.actuator_params])
    if resolved.dry_run:
        command.append("--dry-run")
    return CommandPlan(
        workflow="benchmark",
        robot=resolved.robot,
        command=command,
        dry_run=resolved.dry_run,
        notes=["uses Newton composable actuator descriptions from configs/robots"],
    )


def run_benchmark(config: BenchmarkRunConfig) -> CommandPlan:
    plan = build_benchmark_command(config)
    resolved = resolve_benchmark_run_config(config)
    if resolved.dry_run:
        return plan
    run_benchmark_batch(
        resolved.robot,
        [BenchmarkMotion(motion=resolved.motion, motion_name=resolved.motion_name)],
        output=resolved.output,
        dry_run=False,
        headless=resolved.headless,
        actuator_params=resolved.actuator_params,
        dataset=resolved.dataset,
        actuator_model=resolved.actuator_model,
        seed=resolved.seed,
        num_envs=1,
        batch_index=0,
    )
    return plan


def run_benchmark_batch(
    robot: str,
    motions: list[BenchmarkMotion],
    output: str = "outputs/sim2real_benchmark",
    dry_run: bool = True,
    headless: bool = False,
    actuator_params: str | None = None,
    dataset: str | None = None,
    actuator_model: str | None = None,
    seed: int | None = None,
    num_envs: int | None = None,
    batch_index: int = 0,
) -> None:
    if dry_run:
        return
    from anchor_bench.runtime.environment import require_full_runtime

    require_full_runtime(("isaaclab", "isaaclab_newton", "newton"), workflow="benchmark")
    if robot != "so101":
        raise ValueError("Full runtime execution currently supports only robot=so101")
    from anchor_bench.runtime.assets import resolve_runtime_asset

    resolve_runtime_asset(get_robot(robot))
    from anchor_bench.benchmark.so101_runtime import So101BenchmarkMotion, run_so101_benchmark_batch

    run_so101_benchmark_batch(
        motions=[
            So101BenchmarkMotion(motion=motion.motion, motion_name=motion.motion_name)
            for motion in motions
        ],
        output=output,
        headless=headless,
        actuator_params=actuator_params,
        dataset=dataset,
        actuator_model=actuator_model,
        seed=seed,
        num_envs=num_envs,
        batch_index=batch_index,
    )
