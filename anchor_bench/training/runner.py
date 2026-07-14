from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from anchor_bench.actuators.loader import load_actuator_file
from anchor_bench.core.config_model import parse_gru_columns
from anchor_bench.core.plan import CommandPlan
from anchor_bench.core.run_config import (
    resolve_repo_path_text,
    resolve_required_text,
    resolve_simulation_timing,
    resolve_value_with_source,
)
from anchor_bench.core.schemas import TrainActuatorRunConfig, TrainingJointMapping
from anchor_bench.robots.registry import get_robot
from anchor_bench.training.data import (
    DEFAULT_INPUT_COLUMNS,
    DEFAULT_TARGET_COLUMN,
    MODEL_CHOICES,
    TrainingMapping,
    discover_training_sources,
    read_training_header,
    resolve_training_columns,
    split_training_sources,
)

if TYPE_CHECKING:
    from anchor_bench.training.trainer import PhysicsBaselines


@dataclass(frozen=True)
class ResolvedTrainActuatorRunConfig:
    robot: str
    model: str
    data: str
    output: str
    artifact_name: str
    mappings: tuple[TrainingMapping, ...]
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
    hidden_size: int | None = None
    num_layers: int | None = None
    dropout: float | None = None
    learning_rate: float | None = None
    weight_decay: float | None = None
    huber_delta: float | None = None
    chunk_length: int | None = None
    output_activation: str = "linear"
    output_scale: float = 1.0
    target_normalization: str = "standard"
    pd_kp: float | tuple[float, ...] | None = None
    pd_kd: float | tuple[float, ...] | None = None


def _positive_int(name: str, value: Any) -> int:
    resolved = int(value)
    if resolved < 1:
        raise ValueError(f"{name} must be >= 1")
    return resolved


def _nonnegative_float(name: str, value: Any) -> float:
    """Validate one finite non-negative float."""

    resolved = float(value)
    if not math.isfinite(resolved) or resolved < 0.0:
        raise ValueError(f"{name} must be finite and >= 0")
    return resolved


def _positive_float(name: str, value: Any) -> float:
    """Validate one finite positive float."""

    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return resolved


def _optional_positive_int(name: str, value: Any | None) -> int | None:
    """Validate an optional positive integer CLI override."""

    return None if value is None else _positive_int(name, value)


def _optional_dropout(value: Any | None) -> float | None:
    """Validate an optional GRU dropout probability."""

    if value is None:
        return None
    resolved = float(value)
    if not 0.0 <= resolved < 1.0:
        raise ValueError("dropout must be >= 0 and < 1")
    return resolved


def _optional_pd_gain(name: str, value: Any | None) -> float | tuple[float, ...] | None:
    """Validate one optional scalar gain or ordered gain vector."""

    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError(f"{name} must contain at least one value")
        result: float | tuple[float, ...] = tuple(float(item) for item in value)
        items = result
    else:
        result = float(value)
        items = (result,)
    if any(not math.isfinite(item) or item < 0.0 for item in items):
        raise ValueError(f"{name} must contain only finite values >= 0")
    return result


def _canonical_joints(bundle, values: Sequence[str], label: str) -> tuple[str, ...]:
    """Resolve aliases and the explicit coupled all shorthand."""

    names = tuple(str(value).strip() for value in values)
    if names == ("all",):
        return tuple(bundle.spec.joint_names)
    if "all" in names:
        raise ValueError(f"{label} cannot combine 'all' with named joints")
    aliases = bundle.spec.joint_aliases
    resolved = tuple(aliases.get(name, name) for name in names)
    unknown = [name for name in resolved if name not in bundle.spec.joint_names]
    if unknown:
        raise ValueError(f"unknown training joints in {label}: {', '.join(unknown)}")
    duplicates = sorted({name for name in resolved if resolved.count(name) > 1})
    if duplicates:
        raise ValueError(f"duplicate training joints in {label}: {', '.join(duplicates)}")
    return resolved


def _mapping_from_value(bundle, value: Any) -> TrainingMapping:
    """Convert one YAML/Pydantic mapping into canonical physical joints."""

    if isinstance(value, dict):
        value = TrainingJointMapping.from_dict(value, "training.mapping")
    inputs = value.input_joints
    outputs = value.output_joints
    canonical_inputs = _canonical_joints(bundle, inputs, "input_joints")
    canonical_outputs = (
        canonical_inputs if outputs is None else _canonical_joints(bundle, outputs, "output_joints")
    )
    return TrainingMapping(canonical_inputs, canonical_outputs)


def _validated_mappings(mappings: Sequence[TrainingMapping]) -> tuple[TrainingMapping, ...]:
    """Enforce the shared-weight shape contract across mappings."""

    result = tuple(mappings)
    if not result:
        raise ValueError("training requires at least one joint mapping")
    shape = (len(result[0].input_joints), len(result[0].output_joints))
    if any((len(item.input_joints), len(item.output_joints)) != shape for item in result):
        raise ValueError("all training mappings must have identical input and output widths")
    identities = [(item.input_joints, item.output_joints) for item in result]
    if len(set(identities)) != len(identities):
        raise ValueError("training mappings must not contain duplicates")
    return result


def _mapping_artifact_name(mappings: Sequence[TrainingMapping]) -> str:
    """Create a stable collision-resistant label from canonical mappings."""

    payload = [
        {"input_joints": item.input_joints, "output_joints": item.output_joints}
        for item in mappings
    ]
    digest = hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()[:10]
    return f"mapped_{digest}"


def _resolve_training_mappings(
    bundle, config: TrainActuatorRunConfig
) -> tuple[str, tuple[TrainingMapping, ...], str]:
    """Resolve mappings from a complete CLI override or robot training YAML."""

    defaults = bundle.run_config.training
    cli_modes = sum(
        (
            config.mappings is not None,
            config.input_joints is not None or config.output_joints is not None,
        )
    )
    if cli_modes > 1:
        raise ValueError("choose only one of mappings or input/output joints")

    if cli_modes:
        source = "cli"
        mappings_value = config.mappings
        input_joints = config.input_joints
        output_joints = config.output_joints
    else:
        source = "run.yaml"
        mappings_value = defaults.mappings
        input_joints = defaults.input_joints
        output_joints = defaults.output_joints
        configured_modes = sum(
            (
                mappings_value is not None,
                input_joints is not None or output_joints is not None,
            )
        )
        if configured_modes > 1:
            raise ValueError("run.yaml training must choose mappings or input/output joints")
        if configured_modes == 0:
            raise ValueError("training requires mappings or input_joints")

    if mappings_value is not None:
        mappings = _validated_mappings([_mapping_from_value(bundle, value) for value in mappings_value])
        return _mapping_artifact_name(mappings), mappings, source
    if input_joints is None:
        raise ValueError("input_joints is required when output_joints is configured")
    mapping = _mapping_from_value(
        bundle,
        {"input_joints": input_joints, "output_joints": output_joints},
    )
    return _mapping_artifact_name((mapping,)), (mapping,), source


def resolve_train_actuator_run_config(config: TrainActuatorRunConfig) -> ResolvedTrainActuatorRunConfig:
    bundle = get_robot(config.robot)
    training_cfg = bundle.run_config.training
    values: dict[str, Any] = {}
    sources: dict[str, str] = {}

    def select(name: str, explicit: Any, fallback: Any) -> Any:
        values[name], sources[name] = resolve_value_with_source(explicit, getattr(training_cfg, name), fallback)
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
    artifact_name, mappings, mapping_source = _resolve_training_mappings(bundle, config)
    sources["joint_selection"] = mapping_source
    input_columns = parse_gru_columns(select("input_columns", config.input_columns, DEFAULT_INPUT_COLUMNS))
    hidden_size = _optional_positive_int(
        "hidden_size", select("hidden_size", config.hidden_size, None)
    )
    num_layers = _optional_positive_int("num_layers", select("num_layers", config.num_layers, None))
    dropout = _optional_dropout(select("dropout", config.dropout, None))
    if num_layers == 1 and dropout not in (None, 0.0):
        raise ValueError("dropout must be zero when num_layers is 1")
    learning_rate = select("learning_rate", config.learning_rate, None)
    if learning_rate is not None:
        learning_rate = _positive_float("learning_rate", learning_rate)
    weight_decay = select("weight_decay", config.weight_decay, None)
    if weight_decay is not None:
        weight_decay = _nonnegative_float("weight_decay", weight_decay)
    huber_delta = select("huber_delta", config.huber_delta, None)
    if huber_delta is not None:
        huber_delta = _positive_float("huber_delta", huber_delta)
    chunk_length = select("chunk_length", config.chunk_length, None)
    if chunk_length is not None:
        chunk_length = _positive_int("chunk_length", chunk_length)
    output_activation = str(select("output_activation", config.output_activation, "linear"))
    if output_activation not in {"linear", "tanh"}:
        raise ValueError("output_activation must be 'linear' or 'tanh'")
    output_scale = _positive_float("output_scale", select("output_scale", config.output_scale, 1.0))
    target_normalization = str(
        select("target_normalization", config.target_normalization, "standard")
    )
    if target_normalization not in {"standard", "identity"}:
        raise ValueError("target_normalization must be 'standard' or 'identity'")
    pd_kp = _optional_pd_gain("pd_kp", select("pd_kp", config.pd_kp, None))
    pd_kd = _optional_pd_gain("pd_kd", select("pd_kd", config.pd_kd, None))

    return ResolvedTrainActuatorRunConfig(
        robot=config.robot,
        model=model,
        data=resolve_repo_path_text(data),
        output=resolve_repo_path_text(output),
        artifact_name=artifact_name,
        mappings=mappings,
        dry_run=config.dry_run,
        skip_optuna=bool(select("skip_optuna", config.skip_optuna, True)),
        trials=_positive_int("trials", select("trials", config.trials, 20)),
        trial_epochs=_positive_int("trial_epochs", select("trial_epochs", config.trial_epochs, 5)),
        final_epochs=_positive_int("final_epochs", select("final_epochs", config.final_epochs, 50)),
        n_streams=_positive_int("n_streams", select("n_streams", config.n_streams, 4)),
        noise_std=_nonnegative_float("noise_std", select("noise_std", config.noise_std, 0.0)),
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        huber_delta=huber_delta,
        chunk_length=chunk_length,
        output_activation=output_activation,
        output_scale=output_scale,
        target_normalization=target_normalization,
        pd_kp=pd_kp,
        pd_kd=pd_kd,
        seed=int(select("seed", config.seed, 42)),
        input_columns=input_columns,
        target_column=str(select("target_column", config.target_column, DEFAULT_TARGET_COLUMN)),
        config_sources=sources,
    )


def _gain_text(value: float | tuple[float, ...]) -> str:
    """Render a scalar or ordered gain vector for the CLI."""

    if isinstance(value, tuple):
        return ",".join(str(item) for item in value)
    return str(value)


def _command_from_resolved(resolved: ResolvedTrainActuatorRunConfig) -> list[str]:
    """Render a reproducible command for the resolved topology."""

    command = [
        "anchor-bench",
        "train-actuator",
        "--robot",
        resolved.robot,
        "--model",
        resolved.model,
        "--data",
        resolved.data,
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
    ]
    if resolved.hidden_size is not None:
        command.extend(("--hidden-size", str(resolved.hidden_size)))
    if resolved.num_layers is not None:
        command.extend(("--num-layers", str(resolved.num_layers)))
    if resolved.dropout is not None:
        command.extend(("--dropout", str(resolved.dropout)))
    if resolved.learning_rate is not None:
        command.extend(("--learning-rate", str(resolved.learning_rate)))
    if resolved.weight_decay is not None:
        command.extend(("--weight-decay", str(resolved.weight_decay)))
    if resolved.huber_delta is not None:
        command.extend(("--huber-delta", str(resolved.huber_delta)))
    if resolved.chunk_length is not None:
        command.extend(("--chunk-length", str(resolved.chunk_length)))
    command.extend(("--output-activation", resolved.output_activation))
    command.extend(("--output-scale", str(resolved.output_scale)))
    command.extend(("--target-normalization", resolved.target_normalization))
    if resolved.pd_kp is not None:
        command.extend(("--pd-kp", _gain_text(resolved.pd_kp)))
    if resolved.pd_kd is not None:
        command.extend(("--pd-kd", _gain_text(resolved.pd_kd)))
    if len(resolved.mappings) == 1:
        mapping = resolved.mappings[0]
        command.extend(("--input-joints", ",".join(mapping.input_joints)))
        command.extend(("--output-joints", ",".join(mapping.output_joints)))
    else:
        command.extend(
            (
                "--mappings",
                json.dumps(
                    [
                        {"input_joints": item.input_joints, "output_joints": item.output_joints}
                        for item in resolved.mappings
                    ],
                    separators=(",", ":"),
                ),
            )
        )
    if resolved.skip_optuna:
        command.append("--skip-optuna")
    if resolved.dry_run:
        command.append("--dry-run")
    return command


def build_resolved_train_actuator_command(resolved: ResolvedTrainActuatorRunConfig) -> CommandPlan:
    notes = [
        "trains a stateful TBPTT GRU actuator model with torch imported only during execution",
        f"generated checkpoints and stats should stay under outputs/training/{resolved.robot}/ by convention",
        "promote only non-weight metadata into inputs/actuator_models/<robot>/ when needed",
    ]
    if resolved.model == "hybrid_residual":
        notes.append("hybrid_residual trains residual torque after subtracting native PD and friction terms")
    if "dynamic_bias" in resolved.input_columns:
        notes.append("dynamic_bias is derived from recorded state with Anchor's batched MuJoCo-Warp evaluator")
    if "previous_torque" in resolved.input_columns:
        notes.append("previous_torque is the prior raw network output and resets at every episode and mapping")
    return CommandPlan(
        workflow="train-actuator",
        robot=resolved.robot,
        command=_command_from_resolved(resolved),
        dry_run=resolved.dry_run,
        notes=notes,
        metadata={
            "model": resolved.model,
            "joint_mappings": [
                {"input_joints": list(item.input_joints), "output_joints": list(item.output_joints)}
                for item in resolved.mappings
            ],
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


def _param_for_joint(value: Any, joint: str, name: str, default: float | None = None) -> float:
    """Read one scalar-or-per-joint actuator parameter."""

    if isinstance(value, dict):
        if joint in value:
            return float(value[joint])
        raise ValueError(f"{name} is missing a value for joint {joint!r}")
    if value is None:
        if default is None:
            raise ValueError(f"{name} is missing a value for joint {joint!r}")
        return float(default)
    return float(value)


def _actuator_for_joint(spec, joint: str):
    """Find the single declarative actuator covering one physical joint."""

    matches = [actuator for actuator in spec.actuators if joint in actuator.joints]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one actuator entry covering joint {joint!r}")
    return matches[0]


def _gain_for_slot(
    value: float | tuple[float, ...] | None, index: int, width: int, name: str
) -> float | None:
    """Select one configured gain slot, preserving scalar broadcast."""

    if isinstance(value, tuple):
        if len(value) != width:
            raise ValueError(f"{name} must contain {width} value(s)")
        return value[index]
    return value


def _physics_baselines_for_mappings(
    robot: str,
    mappings: Sequence[TrainingMapping],
    *,
    include_solver_pd: bool,
    include_residual: bool,
    pd_kp: float | tuple[float, ...] | None = None,
    pd_kd: float | tuple[float, ...] | None = None,
) -> "PhysicsBaselines":
    """Resolve configured PD overrides or actuator fallbacks plus output friction."""

    from anchor_bench.training.trainer import PhysicsBaselines

    spec = load_actuator_file(get_robot(robot).paths.actuators)
    candidates: list[PhysicsBaselines] = []
    for mapping in mappings:
        pd_joints = mapping.input_joints if include_solver_pd else mapping.output_joints
        kp: list[float] = []
        kd: list[float] = []
        for index, joint in enumerate(pd_joints):
            kp_value = _gain_for_slot(pd_kp, index, len(pd_joints), "pd_kp")
            kd_value = _gain_for_slot(pd_kd, index, len(pd_joints), "pd_kd")
            if kp_value is None or kd_value is None:
                params = _actuator_for_joint(spec, joint).controller.parameters
                kp_value = (
                    kp_value
                    if kp_value is not None
                    else _param_for_joint(params.get("stiffness"), joint, "stiffness")
                )
                kd_value = (
                    kd_value
                    if kd_value is not None
                    else _param_for_joint(params.get("damping"), joint, "damping")
                )
            kp.append(kp_value)
            kd.append(kd_value)

        dry: list[float] = []
        viscous: list[float] = []
        if include_residual:
            for joint in mapping.output_joints:
                physical = _actuator_for_joint(spec, joint).physical_parameters
                dry_value = physical.get("dry_friction", physical.get("dynamic_friction"))
                dry.append(_param_for_joint(dry_value, joint, "dry_friction", default=0.0))
                viscous.append(
                    _param_for_joint(
                        physical.get("viscous_friction"),
                        joint,
                        "viscous_friction",
                        default=0.0,
                    )
                )
        candidates.append(
            PhysicsBaselines(
                kp=tuple(kp),
                kd=tuple(kd),
                dry_friction=tuple(dry) if include_residual else None,
                viscous_friction=tuple(viscous) if include_residual else None,
            )
        )

    first = candidates[0]
    for candidate in candidates[1:]:
        for name in ("kp", "kd", "dry_friction", "viscous_friction"):
            if getattr(first, name) != getattr(candidate, name):
                raise ValueError(
                    f"shared mappings require identical slot-wise {name}; "
                    "train separate artifacts instead"
                )
    return first


def run_resolved_train_actuator(resolved: ResolvedTrainActuatorRunConfig) -> CommandPlan:
    """Return a dry-run plan or execute one resolved shared-weight training job."""

    if resolved.target_column != "torque":
        raise ValueError("SAGE actuator training currently supports target_column='torque'")
    plan = build_resolved_train_actuator_command(resolved)
    if resolved.dry_run:
        return plan

    from anchor_bench.training.trainer import TrainingJob, train_stateful_gru

    bundle = get_robot(resolved.robot)
    sources = discover_training_sources(resolved.data)
    train_sources, val_sources = split_training_sources(sources, seed=resolved.seed)
    derived_columns = ["solver_pd"] if "solver_pd" in resolved.input_columns else []
    columns = resolve_training_columns(
        read_training_header(train_sources[0], resolved.mappings[0].input_joints[0]),
        model=resolved.model,
        input_columns=resolved.input_columns,
        target_column=resolved.target_column,
        derived_columns=derived_columns,
    )
    include_solver_pd = "solver_pd" in columns.input_columns
    include_residual = resolved.model == "hybrid_residual"
    physics_baselines = (
        _physics_baselines_for_mappings(
            resolved.robot,
            resolved.mappings,
            include_solver_pd=include_solver_pd,
            include_residual=include_residual,
            pd_kp=resolved.pd_kp,
            pd_kd=resolved.pd_kd,
        )
        if include_solver_pd or include_residual
        else None
    )
    metadata: dict[str, Any] = {
        "robot": resolved.robot,
        "config_sources": resolved.config_sources,
        "skip_optuna": resolved.skip_optuna,
        "noise_std": resolved.noise_std,
        "sample_dt_s": resolve_simulation_timing(
            bundle.run_config.simulation,
            f"{bundle.spec.display_name} actuator training simulation",
        ).control_dt,
    }
    if physics_baselines is not None:
        metadata["physics_params"] = physics_baselines.__dict__
    dynamic_bias_provider = None
    if "dynamic_bias" in columns.input_columns:
        from anchor_bench.training.dynamic_bias import NewtonMuJoCoDynamicBiasProvider

        dynamic_bias_provider = NewtonMuJoCoDynamicBiasProvider(bundle)
    artifacts = train_stateful_gru(
        TrainingJob(
            model=resolved.model,
            artifact_name=resolved.artifact_name,
            mappings=resolved.mappings,
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
            hidden_size=resolved.hidden_size,
            num_layers=resolved.num_layers,
            dropout=resolved.dropout,
            learning_rate=resolved.learning_rate,
            weight_decay=resolved.weight_decay,
            huber_delta=resolved.huber_delta,
            chunk_length=resolved.chunk_length,
            output_activation=resolved.output_activation,
            output_scale=resolved.output_scale,
            target_normalization=resolved.target_normalization,
            physics_baselines=physics_baselines,
            dynamic_bias_provider=dynamic_bias_provider,
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
