from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from anchor_bench.core.plan import CommandPlan
from anchor_bench.core.run_config import defaulted_value, resolve_repo_path_text, resolve_required_text
from anchor_bench.robots.registry import get_robot


@dataclass(frozen=True)
class ResolvedAnalysisRunConfig:
    robot: str
    result_folder: str
    output: str
    dry_run: bool
    require_end_effector: bool
    sample_dt: float | None
    motion_names: str | list[str] | None


def resolve_analysis_run_config(
    robot: str,
    result_folder: str | None = None,
    output: str | None = None,
    dry_run: bool = True,
    require_end_effector: bool = False,
    motions: str | list[str] | None = None,
) -> ResolvedAnalysisRunConfig:
    bundle = get_robot(robot)
    run_cfg = bundle.run_config
    benchmark_cfg = run_cfg.get("benchmark", {})
    analysis_cfg = run_cfg.get("analysis", {})
    result = resolve_required_text(result_folder, benchmark_cfg.get("output_folder"), "outputs/sim2real_benchmark", "--result-folder")
    output_value = resolve_required_text(output, analysis_cfg.get("output_dir"), "outputs/sim2real_analysis", "--output")
    sample_dt_value = analysis_cfg.get("sample_dt")
    sample_dt = float(sample_dt_value) if sample_dt_value is not None else None
    motion_names = defaulted_value(motions, analysis_cfg.get("motion_names"), None)
    return ResolvedAnalysisRunConfig(
        robot=robot,
        result_folder=resolve_repo_path_text(result),
        output=resolve_repo_path_text(output_value),
        dry_run=dry_run,
        require_end_effector=require_end_effector,
        sample_dt=sample_dt,
        motion_names=motion_names,
    )


def _selector_to_cli_values(value: str | list[str] | None) -> list[str]:
    if value is None or value == "*":
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def build_analysis_command(
    robot: str,
    result_folder: str | None = None,
    output: str | None = None,
    dry_run: bool = True,
    require_end_effector: bool = False,
    motions: str | list[str] | None = None,
) -> CommandPlan:
    resolved = resolve_analysis_run_config(
        robot,
        result_folder=result_folder,
        output=output,
        dry_run=dry_run,
        require_end_effector=require_end_effector,
        motions=motions,
    )
    command = ["anchor-bench", "analyze", "--robot", resolved.robot, "--result-folder", resolved.result_folder, "--output", resolved.output]
    for motion in _selector_to_cli_values(resolved.motion_names):
        command.extend(["--motion", motion])
    if require_end_effector:
        command.append("--require-end-effector")
    if dry_run:
        command.append("--dry-run")
    metadata = {
        "sample_dt": resolved.sample_dt,
        "motion_names": resolved.motion_names,
    }
    return CommandPlan(
        workflow="analysis",
        robot=resolved.robot,
        command=command,
        dry_run=resolved.dry_run,
        notes=["analysis adapters stay host-safe until plotting dependencies are installed"],
        metadata=metadata,
    )


def run_analysis(
    robot: str,
    result_folder: str | None = None,
    output: str | None = None,
    dry_run: bool = True,
    require_end_effector: bool = False,
    motions: str | list[str] | None = None,
) -> CommandPlan:
    plan = build_analysis_command(
        robot,
        result_folder,
        output,
        dry_run=dry_run,
        require_end_effector=require_end_effector,
        motions=motions,
    )
    resolved = resolve_analysis_run_config(
        robot,
        result_folder=result_folder,
        output=output,
        dry_run=dry_run,
        require_end_effector=require_end_effector,
        motions=motions,
    )
    if resolved.dry_run:
        return plan
    if resolved.robot != "so101":
        raise ValueError("Full runtime execution currently supports only robot=so101")
    from anchor_bench.analysis.so101_analysis import run_so101_analysis

    motions: Iterable[str] | None
    if resolved.motion_names is None or resolved.motion_names == "*":
        motions = None
    elif isinstance(resolved.motion_names, list):
        motions = [str(item) for item in resolved.motion_names]
    else:
        motions = [str(resolved.motion_names)]
    run_so101_analysis(
        result_folder=resolved.result_folder,
        output=resolved.output,
        motions=motions,
        require_end_effector=resolved.require_end_effector,
        sample_dt=resolved.sample_dt,
    )
    return plan
