from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anchor_bench.actuators.loader import load_actuator_file
from anchor_bench.core.config_model import find_duplicates, parse_gru_columns, positive_int
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
    TrainingMapping,
    discover_training_sources,
    split_training_sources,
    training_topology,
)


_MODEL_CHOICES = {"gru", "hybrid_residual"}


@dataclass(frozen=True)
class ResolvedTrainingGroup:
    name: str
    mappings: tuple[TrainingMapping, ...]
    pd_kp: float | None
    pd_kd: float | None
    friction_dry: float | None
    friction_viscous: float | None
    config_sources: dict[str, str]


@dataclass(frozen=True)
class ResolvedTrainActuatorRunConfig:
    robot: str
    model: str
    data: str
    output: str
    groups: tuple[ResolvedTrainingGroup, ...]
    joint_selection_mode: str
    sample_dt_s: float
    dry_run: bool
    skip_optuna: bool
    trials: int
    trial_epochs: int
    final_epochs: int
    n_streams: int
    noise_std: float
    seed: int
    input_columns: list[str]
    config_sources: dict[str, str]


def _nonnegative_float(name: str, value: Any) -> float:
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < 0.0:
        raise ValueError(f"{name} must be a finite value >= 0")
    return resolved


def _resolve_sample_dt(
    config: TrainActuatorRunConfig,
    training_cfg: Any,
    bundle: Any,
) -> tuple[float, str]:
    explicit = config.sample_dt_s
    configured = getattr(training_cfg, "sample_dt_s", None)
    if explicit is not None:
        value, source = float(explicit), "cli"
    elif configured is not None:
        value, source = float(configured), "run.yaml"
    else:
        timing = resolve_simulation_timing(
            bundle.run_config.simulation,
            f"{bundle.paths.run}.simulation",
        )
        source = "simulation.control_freq" if bundle.run_config.simulation.control_freq is not None else "fallback"
        return timing.control_dt, source
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("sample_dt_s must be finite and greater than zero")
    return value, source


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
    duplicates = find_duplicates(joints)
    if duplicates:
        raise ValueError(f"{label} contains duplicate joints after alias resolution: " + ", ".join(duplicates))
    return joints


def _resolved_joint_list(
    bundle: Any,
    value: Any,
    *,
    label: str,
    all_joints: Sequence[str] | None = None,
) -> list[str]:
    if value == "all":
        joints = list(all_joints if all_joints is not None else bundle.spec.joint_names)
        if not joints:
            raise ValueError(f"{label}='all' resolved to an empty joint list")
        return joints
    if isinstance(value, str):
        raise ValueError(f"{label} must be 'all' or an ordered joint list")
    return _canonical_joint_list(bundle, value, label=label)


def _resolve_mapping_values(
    bundle: Any,
    raw_inputs: Any,
    raw_outputs: Any,
    *,
    label: str,
) -> TrainingMapping:
    input_joints = _resolved_joint_list(
        bundle,
        raw_inputs,
        label=f"{label}.input_joints",
    )
    output_joints = (
        list(input_joints)
        if raw_outputs is None
        else _resolved_joint_list(
            bundle,
            raw_outputs,
            label=f"{label}.output_joints",
            all_joints=input_joints,
        )
    )
    missing = [joint for joint in output_joints if joint not in input_joints]
    if missing:
        raise ValueError(f"{label}.output_joints must be a subset of input_joints; missing: " + ", ".join(missing))
    return TrainingMapping(
        input_joints=tuple(input_joints),
        output_joints=tuple(output_joints),
    )


def _validate_resolved_group(
    name: str,
    mappings: Sequence[TrainingMapping],
) -> tuple[TrainingMapping, ...]:
    if not mappings:
        raise ValueError(f"training group {name!r} requires at least one mapping")
    input_width = len(mappings[0].input_joints)
    output_width = len(mappings[0].output_joints)
    for index, mapping in enumerate(mappings[1:], start=1):
        if len(mapping.input_joints) != input_width or len(mapping.output_joints) != output_width:
            raise ValueError(
                f"training group {name!r} mapping {index} has shape "
                f"{len(mapping.input_joints)}->{len(mapping.output_joints)}; "
                f"expected {input_width}->{output_width}"
            )
    identities = [(mapping.input_joints, mapping.output_joints) for mapping in mappings]
    if find_duplicates(identities):
        raise ValueError(f"training group {name!r} contains duplicate mappings")
    return tuple(mappings)


def _resolve_joint_groups(
    config: TrainActuatorRunConfig,
    training_cfg: Any,
    bundle: Any,
) -> tuple[list[tuple[str, tuple[TrainingMapping, ...]]], str, dict[str, str]]:
    sources: dict[str, str] = {}
    cli_selection = config.input_joints is not None or config.output_joints is not None
    programmatic_groups = config.groups

    if cli_selection:
        mapping = _resolve_mapping_values(
            bundle,
            config.input_joints,
            config.output_joints,
            label="training.groups[0].mappings[0]",
        )
        sources.update(
            {
                "groups": "cli",
                "input_joints": "cli",
                "output_joints": ("cli" if config.output_joints is not None else "input_joints"),
            }
        )
        return [("default", (mapping,))], "shorthand", sources

    if programmatic_groups:
        raw_groups = programmatic_groups
        source = "api"
        mode = "groups"
    else:
        raw_groups = training_cfg.resolved_groups
        source = "run.yaml"
        mode = "groups" if training_cfg.groups is not None else "shorthand"

    if not raw_groups:
        raise ValueError("training requires groups or input_joints")

    resolved: list[tuple[str, tuple[TrainingMapping, ...]]] = []
    for group_index, raw_group in enumerate(raw_groups):
        name = str(raw_group.name)
        mappings = [
            _resolve_mapping_values(
                bundle,
                raw_mapping.input_joints,
                raw_mapping.output_joints,
                label=f"training.groups[{group_index}].mappings[{mapping_index}]",
            )
            for mapping_index, raw_mapping in enumerate(raw_group.mappings)
        ]
        resolved.append((name, _validate_resolved_group(name, mappings)))

    sources["groups"] = source
    if mode == "shorthand":
        sources["input_joints"] = source
        raw_output = raw_groups[0].mappings[0].output_joints
        sources["output_joints"] = source if raw_output is not None else "input_joints"
    return resolved, mode, sources


def _ordered_mapping_joints(
    mappings: Sequence[TrainingMapping],
    *,
    domain: str,
) -> list[str]:
    joints: list[str] = []
    for mapping in mappings:
        values = mapping.input_joints if domain == "input_joints" else mapping.output_joints
        joints.extend(joint for joint in values if joint not in joints)
    return joints


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

    coverage: dict[str, list[Any]] = {}
    for actuator in spec.actuators:
        for joint in actuator.joints:
            coverage.setdefault(joint, []).append(actuator)

    for joint in joints:
        matches = coverage.get(joint, [])
        if not matches:
            raise ValueError(f"actuator YAML has no actuator covering joint {joint!r}")
        if len(matches) > 1:
            names = ", ".join(actuator.name for actuator in matches)
            raise ValueError(f"actuator YAML has multiple actuators covering joint {joint!r}: {names}")
        actuator = matches[0]
        values = actuator.controller.parameters if kind == "pd" else actuator.physical_parameters
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
        if any(not math.isclose(value, first, rel_tol=1.0e-9, abs_tol=1.0e-12) for _, value in values[1:]):
            details = ", ".join(f"{joint}={value}" for joint, value in values)
            raise ValueError(f"actuator YAML {label} is nonuniform across selected joints: {details}")
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
        raise ValueError(f"{explicit_flags[0]} and {explicit_flags[1]} must be provided together")
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
    sources: dict[str, str] = {}

    def select(name: str, explicit: Any, fallback: Any) -> Any:
        value, sources[name] = resolve_value_with_source(
            explicit,
            getattr(training_cfg, name),
            fallback,
        )
        return value

    model = str(select("model", config.model, "gru"))
    if model not in _MODEL_CHOICES:
        raise ValueError(f"--model {model!r} must be one of {sorted(_MODEL_CHOICES)}")

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
    raw_groups, joint_selection_mode, joint_sources = _resolve_joint_groups(
        config,
        training_cfg,
        bundle,
    )
    sources.update(joint_sources)

    input_columns = parse_gru_columns(select("input_columns", config.input_columns, DEFAULT_INPUT_COLUMNS))

    sample_dt_s, sources["sample_dt_s"] = _resolve_sample_dt(config, training_cfg, bundle)

    pd_required = model == "hybrid_residual" or "solver_pd" in input_columns
    friction_required = model == "hybrid_residual"
    actuator_spec: Any | None = None

    def infer(kind: str, joints: Sequence[str]) -> tuple[float, float]:
        nonlocal actuator_spec
        if actuator_spec is None:
            actuator_spec = load_actuator_file(bundle.paths.actuators)
        return _infer_uniform_baseline(actuator_spec, joints, kind=kind)

    groups: list[ResolvedTrainingGroup] = []
    for group_name, mappings in raw_groups:
        pd_domain = "input_joints" if "solver_pd" in input_columns else "output_joints"
        pd_joints = _ordered_mapping_joints(mappings, domain=pd_domain)
        friction_joints = _ordered_mapping_joints(
            mappings,
            domain="output_joints",
        )
        pd_kp, pd_kd, pd_source = _resolve_baseline_pair(
            label="pd_baseline",
            explicit_first=config.pd_kp,
            explicit_second=config.pd_kd,
            explicit_flags=("--pd-kp", "--pd-kd"),
            configured=training_cfg.pd_baseline,
            configured_fields=("kp", "kd"),
            required=pd_required,
            infer=lambda joints=pd_joints: infer("pd", joints),
        )
        friction_dry, friction_viscous, friction_source = _resolve_baseline_pair(
            label="friction_baseline",
            explicit_first=config.friction_dry,
            explicit_second=config.friction_viscous,
            explicit_flags=("--friction-dry", "--friction-viscous"),
            configured=training_cfg.friction_baseline,
            configured_fields=("dry", "viscous"),
            required=friction_required,
            infer=lambda joints=friction_joints: infer("friction", joints),
        )
        groups.append(
            ResolvedTrainingGroup(
                name=group_name,
                mappings=mappings,
                pd_kp=pd_kp,
                pd_kd=pd_kd,
                friction_dry=friction_dry,
                friction_viscous=friction_viscous,
                config_sources={
                    "pd_baseline": pd_source,
                    "friction_baseline": friction_source,
                },
            )
        )

    return ResolvedTrainActuatorRunConfig(
        robot=config.robot,
        model=model,
        data=resolve_repo_path_text(data),
        output=resolve_repo_path_text(output),
        groups=tuple(groups),
        joint_selection_mode=joint_selection_mode,
        sample_dt_s=sample_dt_s,
        dry_run=config.dry_run,
        skip_optuna=bool(select("skip_optuna", config.skip_optuna, True)),
        trials=positive_int(select("trials", config.trials, 20), name="trials"),
        trial_epochs=positive_int(
            select("trial_epochs", config.trial_epochs, 5),
            name="trial_epochs",
        ),
        final_epochs=positive_int(
            select("final_epochs", config.final_epochs, 50),
            name="final_epochs",
        ),
        n_streams=positive_int(select("n_streams", config.n_streams, 4), name="n_streams"),
        noise_std=_nonnegative_float(
            "noise_std",
            select("noise_std", config.noise_std, 0.0),
        ),
        seed=int(select("seed", config.seed, 42)),
        input_columns=input_columns,
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
    if resolved.joint_selection_mode == "shorthand":
        mapping = resolved.groups[0].mappings[0]
        command.extend(("--input-joints", ",".join(mapping.input_joints)))
        command.extend(("--output-joints", ",".join(mapping.output_joints)))
    else:
        groups_payload = [
            {
                "name": group.name,
                "mappings": [
                    {
                        "input_joints": list(mapping.input_joints),
                        "output_joints": list(mapping.output_joints),
                    }
                    for mapping in group.mappings
                ],
            }
            for group in resolved.groups
        ]
        command.extend(("--groups", json.dumps(groups_payload, separators=(",", ":"))))
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
            "--sample-dt",
            str(resolved.sample_dt_s),
            "--input-columns",
            ",".join(resolved.input_columns),
        )
    )
    emit_pd = resolved.joint_selection_mode == "shorthand" or all(
        group.config_sources["pd_baseline"] == "cli" for group in resolved.groups
    )
    if emit_pd:
        pd_pairs = {(group.pd_kp, group.pd_kd) for group in resolved.groups}
        if len(pd_pairs) == 1:
            pd_kp, pd_kd = pd_pairs.pop()
            if pd_kp is not None:
                assert pd_kd is not None
                command.extend(("--pd-kp", str(pd_kp), "--pd-kd", str(pd_kd)))

    emit_friction = resolved.joint_selection_mode == "shorthand" or all(
        group.config_sources["friction_baseline"] == "cli" for group in resolved.groups
    )
    if emit_friction:
        friction_pairs = {(group.friction_dry, group.friction_viscous) for group in resolved.groups}
        if len(friction_pairs) == 1:
            friction_dry, friction_viscous = friction_pairs.pop()
            if friction_dry is not None:
                assert friction_viscous is not None
                command.extend(
                    (
                        "--friction-dry",
                        str(friction_dry),
                        "--friction-viscous",
                        str(friction_viscous),
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
        notes.append("hybrid_residual trains residual torque after subtracting native PD and friction terms")
    if "dynamic_bias" in resolved.input_columns:
        notes.append("dynamic_bias is evaluated lazily with Newton SolverMuJoCo only during training execution")

    group_metadata: list[dict[str, Any]] = []
    for group in resolved.groups:
        payload: dict[str, Any] = {
            "name": group.name,
            "training_topology": training_topology(
                len(group.mappings[0].input_joints),
                len(group.mappings[0].output_joints),
            ),
            "mapping_count": len(group.mappings),
            "mappings": [
                {
                    "input_joints": list(mapping.input_joints),
                    "output_joints": list(mapping.output_joints),
                }
                for mapping in group.mappings
            ],
            "config_sources": group.config_sources,
        }
        if group.pd_kp is not None:
            payload["pd_baseline"] = {
                "kp": group.pd_kp,
                "kd": group.pd_kd,
            }
        if group.friction_dry is not None:
            payload["friction_baseline"] = {
                "dry": group.friction_dry,
                "viscous": group.friction_viscous,
            }
        group_metadata.append(payload)

    metadata: dict[str, Any] = {
        "model": resolved.model,
        "groups": group_metadata,
        "sample_dt_s": resolved.sample_dt_s,
        "data": resolved.data,
        "output": resolved.output,
        "input_columns": resolved.input_columns,
        "skip_optuna": resolved.skip_optuna,
        "config_sources": resolved.config_sources,
        "artifact_convention": {
            "generated_outputs": f"outputs/training/{resolved.robot}/...",
            "promoted_metadata": f"inputs/actuator_models/{resolved.robot}/...",
            "ignored_weight_patterns": ["*.pt", "*.pth", "*.onnx"],
        },
    }
    return CommandPlan(
        workflow="train-actuator",
        robot=resolved.robot,
        command=_command_from_resolved(resolved),
        dry_run=resolved.dry_run,
        notes=notes,
        metadata=metadata,
    )


def _residual_physics(
    resolved: ResolvedTrainActuatorRunConfig,
    group: ResolvedTrainingGroup,
) -> Any | None:
    needs_physics = resolved.model == "hybrid_residual" or "solver_pd" in resolved.input_columns
    if not needs_physics:
        return None
    if group.pd_kp is None or group.pd_kd is None:
        raise ValueError("residual and solver_pd training require a PD baseline")
    if resolved.model == "hybrid_residual" and (group.friction_dry is None or group.friction_viscous is None):
        raise ValueError("hybrid residual training requires a friction baseline")

    from anchor_bench.training.trainer import ResidualPhysics

    return ResidualPhysics(
        stiffness=group.pd_kp,
        damping=group.pd_kd,
        dry_friction=group.friction_dry or 0.0,
        viscous_friction=group.friction_viscous or 0.0,
    )


def run_resolved_train_actuator(
    resolved: ResolvedTrainActuatorRunConfig,
) -> CommandPlan:
    plan = build_resolved_train_actuator_command(resolved)
    if resolved.dry_run:
        return plan

    from anchor_bench.training.trainer import TrainingJob, train_stateful_gru

    bundle = get_robot(resolved.robot)
    sources = discover_training_sources(resolved.data)
    train_sources, val_sources = split_training_sources(sources, seed=resolved.seed)

    dynamic_bias_provider: Any | None = None
    if "dynamic_bias" in resolved.input_columns:
        from anchor_bench.training.dynamic_bias import (
            NewtonMuJoCoDynamicBiasProvider,
        )

        dynamic_bias_provider = NewtonMuJoCoDynamicBiasProvider(bundle)

    artifact_metadata: dict[str, Any] = {}
    for group in resolved.groups:
        residual_physics = _residual_physics(resolved, group)
        metadata: dict[str, Any] = {
            "robot": resolved.robot,
            "config_sources": {
                **resolved.config_sources,
                **group.config_sources,
            },
            "skip_optuna": resolved.skip_optuna,
            "noise_std": resolved.noise_std,
            "sample_dt_s": resolved.sample_dt_s,
        }
        if residual_physics is not None:
            metadata["physics_params"] = residual_physics.__dict__

        artifacts = train_stateful_gru(
            TrainingJob(
                model=resolved.model,
                group_name=group.name,
                mappings=group.mappings,
                train_sources=train_sources,
                val_sources=val_sources,
                output_dir=Path(resolved.output),
                input_columns=resolved.input_columns,
                seed=resolved.seed,
                final_epochs=resolved.final_epochs,
                trial_epochs=resolved.trial_epochs,
                trials=resolved.trials,
                skip_optuna=resolved.skip_optuna,
                n_streams=resolved.n_streams,
                noise_std=resolved.noise_std,
                residual_physics=residual_physics,
                metadata=metadata,
                sample_dt_s=resolved.sample_dt_s,
                dynamic_bias_provider=dynamic_bias_provider,
            )
        )
        artifact_metadata[group.name] = {
            "checkpoint": str(artifacts.checkpoint_path),
            "stats": str(artifacts.stats_path),
            "network": str(artifacts.network_path),
            "metrics": artifacts.metrics,
        }

    extra_metadata = dict(plan.metadata or {})
    extra_metadata["artifacts"] = artifact_metadata
    return CommandPlan(
        workflow=plan.workflow,
        robot=plan.robot,
        command=plan.command,
        dry_run=False,
        notes=plan.notes,
        metadata=extra_metadata,
    )
