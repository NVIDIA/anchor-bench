
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anchor_bench.actuators.loader import load_actuator_file
from anchor_bench.core.plan import CommandPlan
from anchor_bench.core.run_config import resolve_repo_path_text, resolve_required_text
from anchor_bench.core.schemas import TrainActuatorRunConfig
from anchor_bench.robots.registry import get_robot
from anchor_bench.training.data import (
    DEFAULT_INPUT_COLUMNS,
    DEFAULT_TARGET_COLUMN,
    MODEL_CHOICES,
    SUPPORTED_INPUT_COLUMNS,
    discover_csv_files,
    read_csv_header,
    resolve_training_columns,
    split_csv_files,
)
from anchor_bench.training.trainer import ResidualPhysics, TrainingJob, train_stateful_gru


@dataclass(frozen=True)
class ResolvedTrainActuatorRunConfig:
    robot: str
    model: str
    data: str
    output: str
    joint_type: str
    dry_run: bool
    skip_optuna: bool
    trials: int
    trial_epochs: int
    final_epochs: int
    n_streams: int
    noise_std: float
    seed: int
    input_columns: list[str]
    target_column: str
    config_sources: dict[str, str]


def _defaulted(explicit: Any, default: Any, fallback: Any) -> Any:
    if explicit is not None:
        return explicit
    if default is not None:
        return default
    return fallback


def _positive_int(name: str, value: Any) -> int:
    resolved = int(value)
    if resolved < 1:
        raise ValueError(f"{name} must be >= 1")
    return resolved


def _nonnegative_float(name: str, value: Any) -> float:
    resolved = float(value)
    if resolved < 0.0:
        raise ValueError(f"{name} must be >= 0")
    return resolved


def _column_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        columns = [item.strip() for item in value.split(",")]
    else:
        columns = [str(item).strip() for item in value]
    columns = [column for column in columns if column]
    if not columns:
        raise ValueError("input_columns must contain at least one column")
    return columns


def resolve_train_actuator_run_config(config: TrainActuatorRunConfig) -> ResolvedTrainActuatorRunConfig:
    bundle = get_robot(config.robot)
    training_cfg = bundle.run_config.get("training", {})
    model = str(_defaulted(config.model, training_cfg.get("model"), "gru"))
    if model not in MODEL_CHOICES:
        raise ValueError(f"--model {model!r} must be one of {sorted(MODEL_CHOICES)}")
    data = resolve_required_text(config.data, training_cfg.get("real_data_dir"), None, "--data")
    output = resolve_required_text(config.output, training_cfg.get("output_dir"), f"outputs/training/{config.robot}", "--output")
    joint_type = resolve_required_text(config.joint_type, training_cfg.get("joint_type"), "full", "--joint-type")
    skip_optuna = bool(_defaulted(config.skip_optuna, training_cfg.get("skip_optuna"), True))
    trials = _positive_int("trials", _defaulted(config.trials, training_cfg.get("trials"), 20))
    trial_epochs = _positive_int("trial_epochs", _defaulted(config.trial_epochs, training_cfg.get("trial_epochs"), 5))
    final_epochs = _positive_int("final_epochs", _defaulted(config.final_epochs, training_cfg.get("final_epochs"), 50))
    n_streams = _positive_int("n_streams", _defaulted(config.n_streams, training_cfg.get("n_streams"), 4))
    noise_std = _nonnegative_float("noise_std", _defaulted(config.noise_std, training_cfg.get("noise_std"), 0.0))
    seed = int(_defaulted(config.seed, training_cfg.get("seed"), 42))
    input_columns = _column_list(_defaulted(config.input_columns, training_cfg.get("input_columns"), DEFAULT_INPUT_COLUMNS))
    assert input_columns is not None
    unsupported_columns = sorted(set(input_columns) - SUPPORTED_INPUT_COLUMNS)
    if unsupported_columns:
        raise ValueError(f"unsupported GRU input columns: {', '.join(unsupported_columns)}")
    target_column = str(_defaulted(config.target_column, training_cfg.get("target_column"), DEFAULT_TARGET_COLUMN))
    return ResolvedTrainActuatorRunConfig(
        robot=config.robot,
        model=model,
        data=resolve_repo_path_text(data),
        output=resolve_repo_path_text(output),
        joint_type=str(joint_type),
        dry_run=config.dry_run,
        skip_optuna=skip_optuna,
        trials=trials,
        trial_epochs=trial_epochs,
        final_epochs=final_epochs,
        n_streams=n_streams,
        noise_std=noise_std,
        seed=seed,
        input_columns=input_columns,
        target_column=target_column,
        config_sources={
            "model": "cli" if config.model is not None else ("run.yaml" if training_cfg.get("model") is not None else "fallback"),
            "real_data_dir": "cli" if config.data is not None else "run.yaml",
            "output_dir": "cli" if config.output is not None else ("run.yaml" if training_cfg.get("output_dir") is not None else "fallback"),
            "joint_type": "cli" if config.joint_type is not None else ("run.yaml" if training_cfg.get("joint_type") is not None else "fallback"),
            "skip_optuna": "cli" if config.skip_optuna is not None else ("run.yaml" if training_cfg.get("skip_optuna") is not None else "fallback"),
            "trials": "cli" if config.trials is not None else ("run.yaml" if training_cfg.get("trials") is not None else "fallback"),
            "trial_epochs": "cli" if config.trial_epochs is not None else ("run.yaml" if training_cfg.get("trial_epochs") is not None else "fallback"),
            "final_epochs": "cli" if config.final_epochs is not None else ("run.yaml" if training_cfg.get("final_epochs") is not None else "fallback"),
            "n_streams": "cli" if config.n_streams is not None else ("run.yaml" if training_cfg.get("n_streams") is not None else "fallback"),
            "noise_std": "cli" if config.noise_std is not None else ("run.yaml" if training_cfg.get("noise_std") is not None else "fallback"),
            "seed": "cli" if config.seed is not None else ("run.yaml" if training_cfg.get("seed") is not None else "fallback"),
            "input_columns": "cli" if config.input_columns is not None else ("run.yaml" if training_cfg.get("input_columns") is not None else "fallback"),
            "target_column": "cli" if config.target_column is not None else ("run.yaml" if training_cfg.get("target_column") is not None else "fallback"),
        },
    )


def _command_from_resolved(resolved: ResolvedTrainActuatorRunConfig) -> list[str]:
    command = [
        "anchor-bench", "train-actuator",
        "--robot", resolved.robot,
        "--model", resolved.model,
        "--data", resolved.data,
        "--joint-type", resolved.joint_type,
        "--output", resolved.output,
        "--trials", str(resolved.trials),
        "--trial-epochs", str(resolved.trial_epochs),
        "--final-epochs", str(resolved.final_epochs),
        "--n-streams", str(resolved.n_streams),
        "--noise-std", str(resolved.noise_std),
        "--seed", str(resolved.seed),
        "--input-columns", ",".join(resolved.input_columns),
        "--target-column", resolved.target_column,
    ]
    if resolved.skip_optuna:
        command.append("--skip-optuna")
    if resolved.dry_run:
        command.append("--dry-run")
    return command


def build_train_actuator_command(config: TrainActuatorRunConfig) -> CommandPlan:
    resolved = resolve_train_actuator_run_config(config)
    notes = [
        "trains a stateful TBPTT GRU actuator model with torch/pandas imported only during execution",
        f"generated checkpoints and stats should stay under outputs/training/{resolved.robot}/ by convention",
        "promote only non-weight metadata into inputs/actuator_models/<robot>/ when needed",
    ]
    if resolved.model == "hybrid_residual":
        notes.append("hybrid_residual trains residual torque after subtracting native PD and friction terms")
    return CommandPlan(
        workflow="train-actuator",
        robot=resolved.robot,
        command=_command_from_resolved(resolved),
        dry_run=resolved.dry_run,
        notes=notes,
        metadata={
            "model": resolved.model,
            "joint_type": resolved.joint_type,
            "data": resolved.data,
            "output": resolved.output,
            "input_columns": resolved.input_columns,
            "target_column": resolved.target_column,
            "skip_optuna": resolved.skip_optuna,
            "config_sources": resolved.config_sources,
            "artifact_convention": {
                "generated_outputs": f"outputs/training/{resolved.robot}/...",
                "promoted_metadata": f"inputs/actuator_models/{resolved.robot}/...",
                "ignored_weight_patterns": ["*.pt", "*.pth", "*.onnx"],
            },
        },
    )


def _param_for_joint(value: Any, joint_type: str, default: float = 0.0) -> float:
    if isinstance(value, dict):
        if joint_type in value:
            return float(value[joint_type])
        return default
    if value is None:
        return default
    return float(value)


def _residual_physics_for_joint(robot: str, joint_type: str) -> ResidualPhysics:
    bundle = get_robot(robot)
    spec = load_actuator_file(bundle.paths.actuators)
    matching = [actuator for actuator in spec.actuators if joint_type in actuator.joints or actuator.name == joint_type]
    if not matching:
        raise ValueError(f"no actuator entry in {bundle.paths.actuators} covers joint_type {joint_type!r}")
    actuator = matching[0]
    controller_params = actuator.controller.parameters
    physical = actuator.physical_parameters
    dry = physical.get("dry_friction", physical.get("dynamic_friction"))
    return ResidualPhysics(
        stiffness=_param_for_joint(controller_params.get("stiffness"), joint_type),
        damping=_param_for_joint(controller_params.get("damping"), joint_type),
        dry_friction=_param_for_joint(dry, joint_type),
        viscous_friction=_param_for_joint(physical.get("viscous_friction"), joint_type),
    )


def run_train_actuator(config: TrainActuatorRunConfig) -> CommandPlan:
    plan = build_train_actuator_command(config)
    resolved = resolve_train_actuator_run_config(config)
    if resolved.dry_run:
        return plan
    csvs = discover_csv_files(resolved.data)
    train_csvs, val_csvs = split_csv_files(csvs, seed=resolved.seed)
    derived_columns = ["solver_pd"] if resolved.model == "hybrid_residual" else []
    columns = resolve_training_columns(
        read_csv_header(train_csvs[0]),
        model=resolved.model,
        input_columns=resolved.input_columns,
        target_column=resolved.target_column,
        derived_columns=derived_columns,
    )
    residual_physics = _residual_physics_for_joint(resolved.robot, resolved.joint_type) if resolved.model == "hybrid_residual" else None
    metadata: dict[str, Any] = {
        "robot": resolved.robot,
        "config_sources": resolved.config_sources,
        "skip_optuna": resolved.skip_optuna,
        "noise_std": resolved.noise_std,
    }
    if residual_physics is not None:
        metadata["physics_params"] = residual_physics.__dict__
    artifacts = train_stateful_gru(
        TrainingJob(
            model=resolved.model,
            joint_type=resolved.joint_type,
            train_csvs=train_csvs,
            val_csvs=val_csvs,
            output_dir=Path(resolved.output),
            input_columns=columns.input_columns,
            target_columns=columns.target_columns,
            source_target_columns=columns.source_target_columns,
            seed=resolved.seed,
            final_epochs=resolved.final_epochs,
            trial_epochs=resolved.trial_epochs,
            trials=resolved.trials,
            skip_optuna=resolved.skip_optuna,
            n_streams=resolved.n_streams,
            noise_std=resolved.noise_std,
            residual_physics=residual_physics,
            metadata=metadata,
        )
    )
    extra_metadata = dict(plan.metadata or {})
    extra_metadata["artifacts"] = {
        "checkpoint": str(artifacts.checkpoint_path),
        "stats": str(artifacts.stats_path),
        "network": str(artifacts.network_path),
        "metrics": artifacts.metrics,
    }
    return CommandPlan(
        workflow=plan.workflow,
        robot=plan.robot,
        command=plan.command,
        dry_run=False,
        notes=plan.notes,
        metadata=extra_metadata,
    )
