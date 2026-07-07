from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anchor_bench.actuators.loader import load_actuator_file
from anchor_bench.core.plan import CommandPlan
from anchor_bench.core.run_config import (
    resolve_repo_path_text,
    resolve_required_text,
    resolve_simulation_timing,
    resolve_value_with_source,
)
from anchor_bench.core.schemas import TrainActuatorRunConfig
from anchor_bench.robots.registry import get_robot
from anchor_bench.training.data import (
    DEFAULT_INPUT_COLUMNS,
    DEFAULT_TARGET_COLUMN,
    MODEL_CHOICES,
    SUPPORTED_INPUT_COLUMNS,
    discover_training_sources,
    expand_joint_sources,
    read_training_header,
    resolve_training_columns,
    split_training_sources,
)


_DERIVED_INPUT_COLUMNS = {"dynamic_bias", "previous_torque", "solver_pd"}


@dataclass(frozen=True)
class ResolvedTrainActuatorRunConfig:
    robot: str
    model: str
    data: str
    output: str
    joint_type: str | None
    input_joints: list[str] | None
    output_joints: list[str] | None
    sample_dt_s: float
    pd_kp: float | None
    pd_kd: float | None
    friction_dry: float | None
    friction_viscous: float | None
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


def _positive_int(name: str, value: Any) -> int:
    resolved = int(value)
    if resolved < 1:
        raise ValueError(f"{name} must be >= 1")
    return resolved


def _nonnegative_float(name: str, value: Any) -> float:
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < 0.0:
        raise ValueError(f"{name} must be a finite value >= 0")
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
    duplicates = sorted({column for column in columns if columns.count(column) > 1})
    if duplicates:
        raise ValueError(f"duplicate GRU input columns: {', '.join(duplicates)}")
    return columns


def _canonical_joint(bundle: Any, value: Any, *, label: str) -> str:
    name = str(value).strip()
    if not name:
        raise ValueError(f"{label} contains an empty joint name")
    canonical = bundle.spec.joint_aliases.get(name, name)
    if canonical not in bundle.spec.joint_names:
        raise ValueError(f"{label} contains unknown joint {name!r}")
    return canonical


def _canonical_joint_list(bundle: Any, values: Sequence[Any], *, label: str) -> list[str]:
    joints = [_canonical_joint(bundle, value, label=label) for value in values]
    if not joints:
        raise ValueError(f"{label} must contain at least one joint")
    duplicates = sorted({joint for joint in joints if joints.count(joint) > 1})
    if duplicates:
        raise ValueError(
            f"{label} contains duplicate joints after alias resolution: "
            + ", ".join(duplicates)
        )
    return joints


def _resolve_joint_selection(
    config: TrainActuatorRunConfig,
    training_cfg: Any,
    bundle: Any,
) -> tuple[str | None, list[str] | None, list[str] | None, dict[str, str]]:
    sources: dict[str, str] = {}
    if config.joint_type is not None:
        joint_type = str(config.joint_type).strip()
        if joint_type != "full":
            joint_type = _canonical_joint(bundle, joint_type, label="joint_type")
        sources["joint_type"] = "cli"
        sources["input_joints"] = "overridden_by_cli_joint_type"
        sources["output_joints"] = "overridden_by_cli_joint_type"
        return joint_type, None, None, sources

    cli_lists = config.input_joints is not None or config.output_joints is not None
    configured_lists = (
        training_cfg.input_joints is not None
        or training_cfg.output_joints is not None
    )
    if cli_lists or configured_lists:
        raw_inputs = (
            config.input_joints
            if config.input_joints is not None
            else training_cfg.input_joints
        )
        if raw_inputs is None:
            raise ValueError(
                "input_joints is required when selecting output_joints; "
                "pass --input-joints or configure training.input_joints"
            )
        input_joints = _canonical_joint_list(
            bundle,
            raw_inputs,
            label="input_joints",
        )
        if config.output_joints is not None:
            raw_outputs = config.output_joints
        elif config.input_joints is not None:
            raw_outputs = None
        else:
            raw_outputs = training_cfg.output_joints
        output_joints = (
            list(input_joints)
            if raw_outputs is None
            else _canonical_joint_list(bundle, raw_outputs, label="output_joints")
        )
        missing = [joint for joint in output_joints if joint not in input_joints]
        if missing:
            raise ValueError(
                "output_joints must be a subset of input_joints; missing: "
                + ", ".join(missing)
            )
        sources["joint_type"] = (
            "overridden_by_cli_joint_lists" if cli_lists else "not_configured"
        )
        sources["input_joints"] = (
            "cli" if config.input_joints is not None else "run.yaml"
        )
        if config.output_joints is not None:
            sources["output_joints"] = "cli"
        elif config.input_joints is not None:
            sources["output_joints"] = "input_joints"
        elif training_cfg.output_joints is not None:
            sources["output_joints"] = "run.yaml"
        else:
            sources["output_joints"] = "input_joints"
        return None, input_joints, output_joints, sources

    joint_type = str(training_cfg.joint_type or "full").strip()
    if joint_type != "full":
        joint_type = _canonical_joint(bundle, joint_type, label="joint_type")
    sources["joint_type"] = (
        "run.yaml" if training_cfg.joint_type is not None else "fallback"
    )
    sources["input_joints"] = "legacy_joint_type"
    sources["output_joints"] = "legacy_joint_type"
    return joint_type, None, None, sources


def _selected_joints(
    bundle: Any,
    joint_type: str | None,
    input_joints: list[str] | None,
    output_joints: list[str] | None,
) -> tuple[list[str], list[str]]:
    if input_joints is not None:
        assert output_joints is not None
        return list(input_joints), list(output_joints)
    if joint_type == "full":
        joints = list(bundle.spec.joint_names)
        return joints, joints
    if joint_type is None:
        raise ValueError("training joint selection is incomplete")
    return [joint_type], [joint_type]


def _actuator_for_joint(spec: Any, joint: str) -> Any:
    matches = [actuator for actuator in spec.actuators if joint in actuator.joints]
    if not matches:
        raise ValueError(f"actuator YAML has no actuator covering joint {joint!r}")
    if len(matches) > 1:
        names = ", ".join(actuator.name for actuator in matches)
        raise ValueError(
            f"actuator YAML has multiple actuators covering joint {joint!r}: {names}"
        )
    return matches[0]


def _configured_parameter(
    values: Any,
    *,
    joint: str,
    label: str,
) -> float:
    if isinstance(values, Mapping):
        if joint not in values:
            raise ValueError(f"actuator YAML is missing required {label} for joint {joint!r}")
        value = values[joint]
    elif values is None:
        raise ValueError(f"actuator YAML is missing required {label} for joint {joint!r}")
    else:
        value = values
    return _nonnegative_float(label, value)


def _infer_uniform_baseline(
    spec: Any,
    joints: Sequence[str],
    *,
    kind: str,
) -> tuple[float, float]:
    if not joints:
        raise ValueError(f"{kind} baseline inference requires at least one joint")
    first_values: list[tuple[str, float]] = []
    second_values: list[tuple[str, float]] = []
    if kind == "pd":
        first_key, second_key = "stiffness", "damping"
        first_label, second_label = "pd kp", "pd kd"
    elif kind == "friction":
        first_key, second_key = "dry_friction", "viscous_friction"
        first_label, second_label = "friction dry", "friction viscous"
    else:
        raise ValueError(f"unsupported baseline kind {kind!r}")

    for joint in joints:
        actuator = _actuator_for_joint(spec, joint)
        values = (
            actuator.controller.parameters
            if kind == "pd"
            else actuator.physical_parameters
        )
        first_values.append(
            (
                joint,
                _configured_parameter(
                    values.get(first_key),
                    joint=joint,
                    label=first_label,
                ),
            )
        )
        second_values.append(
            (
                joint,
                _configured_parameter(
                    values.get(second_key),
                    joint=joint,
                    label=second_label,
                ),
            )
        )

    def uniform(label: str, values: list[tuple[str, float]]) -> float:
        first = values[0][1]
        if any(
            not math.isclose(value, first, rel_tol=1.0e-9, abs_tol=1.0e-12)
            for _, value in values[1:]
        ):
            details = ", ".join(f"{joint}={value}" for joint, value in values)
            raise ValueError(
                f"actuator YAML {label} is nonuniform across selected joints: {details}"
            )
        return first

    return uniform(first_label, first_values), uniform(second_label, second_values)


def _resolve_baseline_pair(
    *,
    label: str,
    explicit_first: Any,
    explicit_second: Any,
    explicit_flags: tuple[str, str],
    configured: Any,
    configured_fields: tuple[str, str],
    required: bool,
    infer: Any,
) -> tuple[float | None, float | None, str]:
    has_first = explicit_first is not None
    has_second = explicit_second is not None
    if has_first != has_second:
        raise ValueError(
            f"{explicit_flags[0]} and {explicit_flags[1]} must be provided together"
        )
    if has_first:
        return (
            _nonnegative_float(explicit_flags[0], explicit_first),
            _nonnegative_float(explicit_flags[1], explicit_second),
            "cli",
        )
    if configured is not None:
        return (
            _nonnegative_float(
                f"training.{label}.{configured_fields[0]}",
                getattr(configured, configured_fields[0]),
            ),
            _nonnegative_float(
                f"training.{label}.{configured_fields[1]}",
                getattr(configured, configured_fields[1]),
            ),
            "run.yaml",
        )
    if required:
        first, second = infer()
        return first, second, "actuators.yaml"
    return None, None, "not_required"


def resolve_train_actuator_run_config(
    config: TrainActuatorRunConfig,
) -> ResolvedTrainActuatorRunConfig:
    bundle = get_robot(config.robot)
    training_cfg = bundle.run_config.training
    values: dict[str, Any] = {}
    sources: dict[str, str] = {}

    def select(name: str, explicit: Any, fallback: Any) -> Any:
        values[name], sources[name] = resolve_value_with_source(
            explicit,
            getattr(training_cfg, name),
            fallback,
        )
        return values[name]

    model = str(select("model", config.model, "gru"))
    if model not in MODEL_CHOICES:
        raise ValueError(f"--model {model!r} must be one of {sorted(MODEL_CHOICES)}")

    data = resolve_required_text(
        select("real_data_dir", config.data, None),
        None,
        None,
        "--data",
    )
    output = resolve_required_text(
        select("output_dir", config.output, f"outputs/training/{config.robot}"),
        None,
        None,
        "--output",
    )
    joint_type, input_joints, output_joints, joint_sources = _resolve_joint_selection(
        config,
        training_cfg,
        bundle,
    )
    sources.update(joint_sources)

    input_columns = _column_list(
        select("input_columns", config.input_columns, DEFAULT_INPUT_COLUMNS)
    )
    if input_columns is None:
        raise ValueError("input_columns must contain at least one column")
    unsupported_columns = sorted(set(input_columns) - SUPPORTED_INPUT_COLUMNS)
    if unsupported_columns:
        raise ValueError(
            f"unsupported GRU input columns: {', '.join(unsupported_columns)}"
        )

    target_column = str(
        select("target_column", config.target_column, DEFAULT_TARGET_COLUMN)
    )
    if target_column != "torque":
        raise ValueError("SAGE actuator training requires target_column='torque'")

    timing = resolve_simulation_timing(
        bundle.run_config.simulation,
        f"{bundle.paths.run}.simulation",
    )
    sample_dt_s = timing.control_dt
    sources["sample_dt_s"] = (
        "run.yaml"
        if bundle.run_config.simulation.control_freq is not None
        else "fallback"
    )

    pd_required = model == "hybrid_residual" or "solver_pd" in input_columns
    friction_required = model == "hybrid_residual"
    selected_inputs, selected_outputs = _selected_joints(
        bundle,
        joint_type,
        input_joints,
        output_joints,
    )
    pd_joints = selected_inputs if "solver_pd" in input_columns else selected_outputs
    friction_joints = selected_outputs
    actuator_spec: Any | None = None

    def infer(kind: str, joints: Sequence[str]) -> tuple[float, float]:
        nonlocal actuator_spec
        if actuator_spec is None:
            actuator_spec = load_actuator_file(bundle.paths.actuators)
        return _infer_uniform_baseline(actuator_spec, joints, kind=kind)

    pd_kp, pd_kd, pd_source = _resolve_baseline_pair(
        label="pd_baseline",
        explicit_first=config.pd_kp,
        explicit_second=config.pd_kd,
        explicit_flags=("--pd-kp", "--pd-kd"),
        configured=training_cfg.pd_baseline,
        configured_fields=("kp", "kd"),
        required=pd_required,
        infer=lambda: infer("pd", pd_joints),
    )
    friction_dry, friction_viscous, friction_source = _resolve_baseline_pair(
        label="friction_baseline",
        explicit_first=config.friction_dry,
        explicit_second=config.friction_viscous,
        explicit_flags=("--friction-dry", "--friction-viscous"),
        configured=training_cfg.friction_baseline,
        configured_fields=("dry", "viscous"),
        required=friction_required,
        infer=lambda: infer("friction", friction_joints),
    )
    sources["pd_baseline"] = pd_source
    sources["friction_baseline"] = friction_source

    return ResolvedTrainActuatorRunConfig(
        robot=config.robot,
        model=model,
        data=resolve_repo_path_text(data),
        output=resolve_repo_path_text(output),
        joint_type=joint_type,
        input_joints=input_joints,
        output_joints=output_joints,
        sample_dt_s=sample_dt_s,
        pd_kp=pd_kp,
        pd_kd=pd_kd,
        friction_dry=friction_dry,
        friction_viscous=friction_viscous,
        dry_run=config.dry_run,
        skip_optuna=bool(select("skip_optuna", config.skip_optuna, True)),
        trials=_positive_int("trials", select("trials", config.trials, 20)),
        trial_epochs=_positive_int(
            "trial_epochs",
            select("trial_epochs", config.trial_epochs, 5),
        ),
        final_epochs=_positive_int(
            "final_epochs",
            select("final_epochs", config.final_epochs, 50),
        ),
        n_streams=_positive_int("n_streams", select("n_streams", config.n_streams, 4)),
        noise_std=_nonnegative_float(
            "noise_std",
            select("noise_std", config.noise_std, 0.0),
        ),
        seed=int(select("seed", config.seed, 42)),
        input_columns=input_columns,
        target_column=target_column,
        config_sources=sources,
    )


def _command_from_resolved(resolved: ResolvedTrainActuatorRunConfig) -> list[str]:
    command = [
        "anchor-bench",
        "train-actuator",
        "--robot",
        resolved.robot,
        "--model",
        resolved.model,
        "--data",
        resolved.data,
    ]
    if resolved.joint_type is not None:
        command.extend(("--joint-type", resolved.joint_type))
    else:
        assert resolved.input_joints is not None
        assert resolved.output_joints is not None
        command.extend(("--input-joints", ",".join(resolved.input_joints)))
        command.extend(("--output-joints", ",".join(resolved.output_joints)))
    command.extend(
        (
            "--output",
            resolved.output,
            "--trials",
            str(resolved.trials),
            "--trial-epochs",
            str(resolved.trial_epochs),
            "--final-epochs",
            str(resolved.final_epochs),
            "--n-streams",
            str(resolved.n_streams),
            "--noise-std",
            str(resolved.noise_std),
            "--seed",
            str(resolved.seed),
            "--input-columns",
            ",".join(resolved.input_columns),
            "--target-column",
            resolved.target_column,
        )
    )
    if resolved.pd_kp is not None:
        assert resolved.pd_kd is not None
        command.extend(("--pd-kp", str(resolved.pd_kp), "--pd-kd", str(resolved.pd_kd)))
    if resolved.friction_dry is not None:
        assert resolved.friction_viscous is not None
        command.extend(
            (
                "--friction-dry",
                str(resolved.friction_dry),
                "--friction-viscous",
                str(resolved.friction_viscous),
            )
        )
    if resolved.skip_optuna:
        command.append("--skip-optuna")
    if resolved.dry_run:
        command.append("--dry-run")
    return command


def build_resolved_train_actuator_command(
    resolved: ResolvedTrainActuatorRunConfig,
) -> CommandPlan:
    notes = [
        "trains a stateful TBPTT GRU actuator model with torch imported only during execution",
        f"generated checkpoints and stats should stay under outputs/training/{resolved.robot}/ by convention",
        "promote only non-weight metadata into inputs/actuator_models/<robot>/ when needed",
    ]
    if resolved.model == "hybrid_residual":
        notes.append(
            "hybrid_residual trains residual torque after subtracting native PD and friction terms"
        )
    if "dynamic_bias" in resolved.input_columns:
        notes.append(
            "dynamic_bias is evaluated lazily with Newton SolverMuJoCo only during training execution"
        )

    metadata: dict[str, Any] = {
        "model": resolved.model,
        "joint_type": resolved.joint_type,
        "input_joints": resolved.input_joints,
        "output_joints": resolved.output_joints,
        "sample_dt_s": resolved.sample_dt_s,
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
    }
    if resolved.pd_kp is not None:
        metadata["pd_baseline"] = {
            "kp": resolved.pd_kp,
            "kd": resolved.pd_kd,
        }
    if resolved.friction_dry is not None:
        metadata["friction_baseline"] = {
            "dry": resolved.friction_dry,
            "viscous": resolved.friction_viscous,
        }
    return CommandPlan(
        workflow="train-actuator",
        robot=resolved.robot,
        command=_command_from_resolved(resolved),
        dry_run=resolved.dry_run,
        notes=notes,
        metadata=metadata,
    )


def _residual_physics(resolved: ResolvedTrainActuatorRunConfig) -> Any | None:
    needs_physics = (
        resolved.model == "hybrid_residual"
        or "solver_pd" in resolved.input_columns
    )
    if not needs_physics:
        return None
    if resolved.pd_kp is None or resolved.pd_kd is None:
        raise ValueError("residual and solver_pd training require a PD baseline")
    if resolved.model == "hybrid_residual" and (
        resolved.friction_dry is None or resolved.friction_viscous is None
    ):
        raise ValueError("hybrid residual training requires a friction baseline")

    from anchor_bench.training.trainer import ResidualPhysics

    return ResidualPhysics(
        stiffness=resolved.pd_kp,
        damping=resolved.pd_kd,
        dry_friction=resolved.friction_dry or 0.0,
        viscous_friction=resolved.friction_viscous or 0.0,
    )


def _job_joint_type(resolved: ResolvedTrainActuatorRunConfig) -> str:
    if resolved.joint_type is not None:
        return resolved.joint_type
    assert resolved.output_joints
    return "__".join(resolved.output_joints)


def run_resolved_train_actuator(
    resolved: ResolvedTrainActuatorRunConfig,
) -> CommandPlan:
    plan = build_resolved_train_actuator_command(resolved)
    if resolved.dry_run:
        return plan

    from anchor_bench.training.trainer import TrainingJob, train_stateful_gru

    bundle = get_robot(resolved.robot)
    sources = discover_training_sources(resolved.data)
    if resolved.joint_type == "full":
        sources = expand_joint_sources(sources, bundle.spec.joint_names)
    train_sources, val_sources = split_training_sources(sources, seed=resolved.seed)

    if resolved.input_joints is not None:
        header_joint = resolved.input_joints[0]
    elif train_sources[0].joint is not None:
        header_joint = train_sources[0].joint
    else:
        assert resolved.joint_type is not None
        header_joint = resolved.joint_type
    derived_columns = [
        column
        for column in resolved.input_columns
        if column in _DERIVED_INPUT_COLUMNS
    ]
    columns = resolve_training_columns(
        read_training_header(train_sources[0], header_joint),
        model=resolved.model,
        input_columns=resolved.input_columns,
        target_column=resolved.target_column,
        derived_columns=derived_columns,
    )
    residual_physics = _residual_physics(resolved)

    dynamic_bias_provider: Any | None = None
    if "dynamic_bias" in resolved.input_columns:
        from anchor_bench.training.dynamic_bias import (
            NewtonMuJoCoDynamicBiasProvider,
        )

        dynamic_bias_provider = NewtonMuJoCoDynamicBiasProvider(bundle)

    metadata: dict[str, Any] = {
        "robot": resolved.robot,
        "config_sources": resolved.config_sources,
        "skip_optuna": resolved.skip_optuna,
        "noise_std": resolved.noise_std,
        "sample_dt_s": resolved.sample_dt_s,
    }
    if residual_physics is not None:
        metadata["physics_params"] = residual_physics.__dict__

    artifacts = train_stateful_gru(
        TrainingJob(
            model=resolved.model,
            joint_type=_job_joint_type(resolved),
            train_sources=train_sources,
            val_sources=val_sources,
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
            input_joints=resolved.input_joints,
            output_joints=resolved.output_joints,
            sample_dt_s=resolved.sample_dt_s,
            dynamic_bias_provider=dynamic_bias_provider,
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
