from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from anchor_bench.training.data import (
    EPSILON_STD,
    TrainingEpisode,
    FeatureBlock,
    TrainingSource,
    write_stats_json,
)
from anchor_bench.training.export import build_runtime_gru, export_gru_checkpoint


@dataclass(frozen=True)
class ResidualPhysics:
    stiffness: float = 0.0
    damping: float = 0.0
    dry_friction: float = 0.0
    viscous_friction: float = 0.0


@dataclass(frozen=True)
class TrainingHyperParameters:
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.0
    learning_rate: float = 1.0e-3
    weight_decay: float = 0.0
    huber_delta: float = 1.0
    chunk_length: int = 16


@dataclass(frozen=True)
class TrainingJob:
    model: str
    train_sources: Sequence[TrainingSource]
    val_sources: Sequence[TrainingSource]
    output_dir: Path
    input_columns: list[str]
    input_joints: Sequence[str]
    output_joints: Sequence[str]
    sample_dt_s: float
    seed: int
    final_epochs: int
    trial_epochs: int
    trials: int
    skip_optuna: bool
    n_streams: int
    noise_std: float
    residual_physics: ResidualPhysics | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    dynamic_bias_provider: Any | None = None


@dataclass(frozen=True)
class TrainingArtifacts:
    checkpoint_path: Path
    stats_path: Path
    network_path: Path
    metrics: dict[str, Any]


def _import_torch():
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "PyTorch is required for train-actuator execution. Run in the full container "
            "or install torch in the active environment."
        ) from exc
    return torch


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "joint"


def _validated_axes(
    input_joints: Sequence[str],
    output_joints: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    inputs = tuple(input_joints)
    outputs = tuple(output_joints)
    if not inputs:
        raise ValueError("GRU training requires at least one input joint")
    if not outputs:
        raise ValueError("GRU training requires at least one output joint")
    if any(not isinstance(name, str) or not name for name in (*inputs, *outputs)):
        raise ValueError("GRU joint names must be non-empty strings")
    if len(set(inputs)) != len(inputs):
        raise ValueError("input_joints must not contain duplicates")
    if len(set(outputs)) != len(outputs):
        raise ValueError("output_joints must not contain duplicates")
    missing = [name for name in outputs if name not in inputs]
    if missing:
        raise ValueError("output_joints must be a subset of input_joints; missing: " + ", ".join(missing))
    return inputs, outputs


def _runtime_axes(job: TrainingJob) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return _validated_axes(job.input_joints, job.output_joints)


def _training_topology(job: TrainingJob) -> str:
    inputs, outputs = _runtime_axes(job)
    if len(inputs) == 1:
        return "siso"
    if len(outputs) == 1:
        return "miso"
    return "mimo"


def _artifact_label(job: TrainingJob) -> str:
    topology = _training_topology(job)
    input_joints, output_joints = _runtime_axes(job)
    identity = {
        "topology": topology,
        "input_joints": list(input_joints),
        "output_joints": list(output_joints),
    }
    readable = f"{topology}__in__{'__'.join(input_joints)}__out__{'__'.join(output_joints)}"

    digest = hashlib.sha256(json.dumps(identity, separators=(",", ":"), sort_keys=True).encode("utf-8")).hexdigest()[
        :10
    ]
    suffix = f"__{digest}"
    prefix = _safe_name(readable)[: 96 - len(suffix)].rstrip("_.-")
    return f"{prefix}{suffix}"


def _sample_dt_s(job: TrainingJob) -> float:
    try:
        result = float(job.sample_dt_s)
    except (TypeError, ValueError) as exc:
        raise ValueError("GRU training requires sample_dt_s") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError("sample_dt_s must be finite and greater than zero")
    return result


def _target_column(job: TrainingJob) -> str:
    return "torque_residual" if job.model == "hybrid_residual" else "torque"


def _input_feature_specs(job: TrainingJob) -> list[dict[str, Any]]:
    specs = job.metadata.get("input_feature_specs")
    if not isinstance(specs, list):
        raise ValueError(
            "training metadata is missing input_feature_specs; assemble at least one training episode first"
        )
    return specs


def _record_input_feature_specs(
    job: TrainingJob,
    specs: list[dict[str, Any]],
    *,
    source: str,
) -> None:
    if "input_feature_specs" not in job.metadata:
        job.metadata["input_feature_specs"] = specs
        return
    if job.metadata["input_feature_specs"] != specs:
        raise ValueError(f"{source} input feature layout differs from other training episodes")


def _feature_widths(job: TrainingJob) -> tuple[int, ...]:
    inputs, outputs = _runtime_axes(job)
    axis_widths = {
        "input_joints": len(inputs),
        "output_joints": len(outputs),
    }
    return tuple(axis_widths[spec["domain"]] * len(spec["channels"]) for spec in _input_feature_specs(job))


def _physics_scalar_input(
    blocks: Mapping[str, FeatureBlock],
    name: str,
    *,
    source: str,
) -> np.ndarray:
    block = blocks[name]
    if block.domain != "input_joints" or len(block.channels) != 1:
        raise ValueError(f"{source} physics feature {name!r} must use domain 'input_joints' with one scalar channel")
    return block.values[:, :, 0]


def _apply_physics_baseline(
    job: TrainingJob,
    blocks: dict[str, FeatureBlock],
    target: np.ndarray,
    *,
    input_joints: Sequence[str],
    output_joints: Sequence[str],
    source: str,
) -> np.ndarray:
    needs_physics = job.model == "hybrid_residual" or "solver_pd" in job.input_columns
    if not needs_physics:
        return target
    physics = job.residual_physics
    if physics is None:
        raise ValueError("residual and solver_pd training require residual physics parameters")
    missing = [name for name in ("position_error", "velocity") if name not in blocks]
    if missing:
        raise ValueError(f"{source} missing features required by the physics baseline: " + ", ".join(missing))
    position_error = _physics_scalar_input(
        blocks,
        "position_error",
        source=source,
    )
    velocity = _physics_scalar_input(blocks, "velocity", source=source)
    solver_pd = (physics.stiffness * position_error - physics.damping * velocity).astype(np.float32)
    blocks["solver_pd"] = FeatureBlock(
        solver_pd[:, :, np.newaxis],
        "input_joints",
        ("value",),
    )
    if job.model != "hybrid_residual":
        return target
    output_indices = [input_joints.index(name) for name in output_joints]
    output_velocity = velocity[:, output_indices]
    output_solver_pd = solver_pd[:, output_indices]
    friction = physics.dry_friction * np.sign(output_velocity) + physics.viscous_friction * output_velocity
    return (target - output_solver_pd - friction).astype(np.float32)


def _flatten_feature_blocks(
    job: TrainingJob,
    blocks: dict[str, FeatureBlock],
    target: np.ndarray,
    *,
    rows: int,
    source: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if "previous_torque" in job.input_columns:
        previous_target = np.zeros_like(target, dtype=np.float32)
        previous_target[1:] = target[:-1]
        blocks["previous_torque"] = FeatureBlock(
            previous_target[:, :, np.newaxis],
            "output_joints",
            ("value",),
        )
    ordered: list[np.ndarray] = []
    specs: list[dict[str, Any]] = []
    for column in job.input_columns:
        if column not in blocks:
            raise ValueError(f"{source} missing required feature {column!r}")
        block = blocks[column]
        ordered.append(block.values.reshape(rows, -1))
        specs.append(
            {
                "name": column,
                "domain": block.domain,
                "channels": list(block.channels),
            }
        )
    if not ordered:
        raise ValueError("GRU training requires at least one input feature")
    features = np.concatenate(ordered, axis=1).astype(np.float32, copy=False)
    if not np.isfinite(features).all():
        raise ValueError(f"{source} contains non-finite training values")
    return features, specs


def _prepare_episode(
    job: TrainingJob,
    episode: TrainingEpisode,
    *,
    source: TrainingSource | None = None,
    source_name: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    input_joints, output_joints = _runtime_axes(job)
    label = source_name or (str(source.path) if source is not None else "training episode")
    episode_inputs = tuple(episode.input_joints)
    episode_outputs = tuple(episode.output_joints)
    if episode_inputs != input_joints:
        raise ValueError(f"{label} input joint order {episode_inputs} does not match {input_joints}")
    if episode_outputs != output_joints:
        raise ValueError(f"{label} output joint order {episode_outputs} does not match {output_joints}")

    rows = len(episode.timestamps)
    blocks = dict(episode.feature_blocks)
    target = episode.target_torque
    target = _apply_physics_baseline(
        job,
        blocks,
        target,
        input_joints=input_joints,
        output_joints=output_joints,
        source=label,
    )
    features, specs = _flatten_feature_blocks(
        job,
        blocks,
        target,
        rows=rows,
        source=label,
    )
    _record_input_feature_specs(job, specs, source=label)
    return features, target


_DYNAMIC_BIAS_ORDERED_JOINT_LIST_FIELDS = {"requested_joint_names"}

_DYNAMIC_BIAS_JOINT_LIST_FIELDS = {
    "trajectory_joint_names",
    "requested_joint_names",
    "mapped_joint_names",
    "unmapped_trajectory_joint_names",
    "defaulted_joint_names",
}


def _merge_dynamic_bias_provenance(
    accumulated: dict[str, Any] | None,
    episode: Mapping[str, Any],
) -> dict[str, Any]:
    stable = {key: value for key, value in episode.items() if key not in _DYNAMIC_BIAS_JOINT_LIST_FIELDS}
    if accumulated is None:
        result = dict(stable)
        for key in _DYNAMIC_BIAS_JOINT_LIST_FIELDS:
            values = episode.get(key, [])
            if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
                raise ValueError(f"dynamic_bias provenance {key} must be a list of strings")
            result[key] = (
                list(dict.fromkeys(values)) if key in _DYNAMIC_BIAS_ORDERED_JOINT_LIST_FIELDS else sorted(set(values))
            )
        return result
    existing_stable = {key: value for key, value in accumulated.items() if key not in _DYNAMIC_BIAS_JOINT_LIST_FIELDS}
    if existing_stable != stable:
        raise ValueError("stable dynamic_bias provenance differs between training episodes")
    result = dict(accumulated)
    for key in _DYNAMIC_BIAS_JOINT_LIST_FIELDS:
        values = episode.get(key, [])
        if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
            raise ValueError(f"dynamic_bias provenance {key} must be a list of strings")
        if key in _DYNAMIC_BIAS_ORDERED_JOINT_LIST_FIELDS:
            combined = list(result.get(key, ()))
            combined.extend(value for value in values if value not in combined)
            result[key] = combined
        else:
            result[key] = sorted(set(result.get(key, ())).union(values))
    return result


def _load_sequences(
    job: TrainingJob,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[tuple[np.ndarray, np.ndarray]]]:
    from anchor_bench.training.data import load_training_episode

    if not job.train_sources:
        raise ValueError("TrainingJob requires at least one training sequence")

    loader_columns = [column for column in job.input_columns if column not in {"solver_pd", "previous_torque"}]
    if job.model == "hybrid_residual" or "solver_pd" in job.input_columns:
        for required in ("position_error", "velocity"):
            if required not in loader_columns:
                loader_columns.append(required)

    provenance: dict[str, Any] | None = None

    def merge_provenance(source: TrainingSource, episode: TrainingEpisode) -> None:
        nonlocal provenance
        if "dynamic_bias" not in job.input_columns:
            return
        episode_provenance = episode.dynamic_bias_provenance
        if not isinstance(episode_provenance, Mapping):
            raise ValueError(f"{source.path} dynamic_bias episode is missing derivation provenance")
        provenance = _merge_dynamic_bias_provenance(
            provenance,
            dict(episode_provenance),
        )

    def load_one(source: TrainingSource) -> tuple[np.ndarray, np.ndarray]:
        input_joints, output_joints = _runtime_axes(job)
        episode = load_training_episode(
            source,
            input_joints=input_joints,
            output_joints=output_joints,
            input_columns=loader_columns,
            sample_dt_s=_sample_dt_s(job),
            dynamic_bias_provider=job.dynamic_bias_provider,
        )
        merge_provenance(source, episode)
        return _prepare_episode(job, episode, source=source)

    train = [load_one(source) for source in job.train_sources]
    val = [load_one(source) for source in job.val_sources]
    if provenance is not None:
        existing = job.metadata.get("dynamic_bias_derivation")
        if existing is not None and existing != provenance:
            raise ValueError("configured dynamic_bias provenance does not match episode provenance")
        job.metadata["dynamic_bias_derivation"] = provenance
    return train, val


def _scalar_or_list(values: np.ndarray) -> float | list[float]:
    result = [float(value) for value in values]
    return result[0] if len(result) == 1 else result


def _normalization_stats(
    sequences: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    input_columns: Sequence[str],
    target_column: str,
    metadata: dict[str, Any],
    input_widths: Sequence[int],
    target_width: int,
) -> dict[str, Any]:
    reserved = {"schema_version", "input_columns", "target_columns", "normalization"}
    collisions = sorted(reserved.intersection(metadata))
    if collisions:
        raise ValueError("training metadata cannot override reserved fields: " + ", ".join(collisions))
    if not sequences:
        raise ValueError("normalization requires at least one training sequence")
    inputs = np.concatenate([item[0] for item in sequences], axis=0)
    targets = np.concatenate([item[1] for item in sequences], axis=0)
    widths = tuple(input_widths)
    if len(widths) != len(input_columns) or any(width <= 0 for width in widths):
        raise ValueError("input normalization widths must match input_columns")
    if sum(widths) != inputs.shape[1]:
        raise ValueError(f"input normalization widths sum to {sum(widths)}, expected {inputs.shape[1]}")
    if target_width != targets.shape[1] or target_width <= 0:
        raise ValueError(f"target normalization width is {target_width}, expected {targets.shape[1]}")
    input_mean = inputs.mean(axis=0)
    input_std = np.maximum(inputs.std(axis=0), EPSILON_STD)
    target_mean = targets.mean(axis=0)
    target_std = np.maximum(targets.std(axis=0), EPSILON_STD)
    offsets = np.cumsum((0, *widths))
    input_means = {
        column: _scalar_or_list(input_mean[offsets[index] : offsets[index + 1]])
        for index, column in enumerate(input_columns)
    }
    input_stds = {
        column: _scalar_or_list(input_std[offsets[index] : offsets[index + 1]])
        for index, column in enumerate(input_columns)
    }
    stats = {
        "schema_version": 1,
        "input_columns": list(input_columns),
        "target_columns": [target_column],
        "normalization": {
            "inputs": {
                "mean": input_means,
                "std": input_stds,
            },
            "targets": {
                "mean": {target_column: _scalar_or_list(target_mean)},
                "std": {target_column: _scalar_or_list(target_std)},
            },
        },
    }
    stats.update(metadata)
    return stats


def _stat_vector(
    values: Mapping[str, Any],
    *,
    columns: Sequence[str],
    widths: Sequence[int],
    name: str,
) -> np.ndarray:
    flattened: list[float] = []
    for column, width in zip(columns, widths, strict=True):
        value = values.get(column)
        if width == 1 and not isinstance(value, list):
            raw = [value]
        elif isinstance(value, list) and len(value) == width:
            raw = value
        else:
            raise ValueError(f"{name}.{column} must contain {width} value(s)")
        try:
            numbers = [float(item) for item in raw]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}.{column} must contain numeric values") from exc
        if not all(math.isfinite(item) for item in numbers):
            raise ValueError(f"{name}.{column} must contain finite values")
        flattened.extend(numbers)
    return np.asarray(flattened, dtype=np.float32)


def _normalize_sequences(
    sequences: Sequence[tuple[np.ndarray, np.ndarray]],
    stats: dict[str, Any],
    *,
    input_columns: Sequence[str],
    target_column: str,
    input_widths: Sequence[int],
    target_width: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    input_norm = stats["normalization"]["inputs"]
    target_norm = stats["normalization"]["targets"]
    widths = tuple(input_widths)
    input_mean = _stat_vector(
        input_norm["mean"],
        columns=input_columns,
        widths=widths,
        name="normalization.inputs.mean",
    )
    input_std = _stat_vector(
        input_norm["std"],
        columns=input_columns,
        widths=widths,
        name="normalization.inputs.std",
    )
    target_mean = _stat_vector(
        target_norm["mean"],
        columns=(target_column,),
        widths=(target_width,),
        name="normalization.targets.mean",
    )
    target_std = _stat_vector(
        target_norm["std"],
        columns=(target_column,),
        widths=(target_width,),
        name="normalization.targets.std",
    )
    if np.any(input_std <= 0.0) or np.any(target_std <= 0.0):
        raise ValueError("normalization standard deviations must be greater than zero")
    return [((inputs - input_mean) / input_std, (targets - target_mean) / target_std) for inputs, targets in sequences]


def _run_epoch(torch, model, optimizer, loss_fn, sequences, hparams, *, train: bool, device, noise_std: float) -> float:
    model.train(train)
    losses: list[float] = []
    grad_context = torch.enable_grad() if train else torch.no_grad()
    with grad_context:
        for features, targets in sequences:
            hidden = None
            x = torch.as_tensor(features, dtype=torch.float32, device=device)
            y = torch.as_tensor(targets, dtype=torch.float32, device=device)
            for start in range(0, x.shape[0], hparams.chunk_length):
                xb = x[start : start + hparams.chunk_length].unsqueeze(0)
                yb = y[start : start + hparams.chunk_length].unsqueeze(0)
                if train and noise_std > 0.0:
                    xb = xb + torch.randn_like(xb) * noise_std
                pred, hidden = model(xb, hidden)
                loss = loss_fn(pred, yb)
                if train:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    hidden = hidden.detach()
                losses.append(float(loss.detach().cpu()))
    return float(sum(losses) / max(1, len(losses)))


def _runtime_metadata(job: TrainingJob, stats: dict[str, Any]) -> dict[str, Any]:
    input_joints, output_joints = _runtime_axes(job)
    semantic_target = _target_column(job)
    normalization = stats["normalization"]
    input_feature_specs = _input_feature_specs(job)

    runtime: dict[str, Any] = {
        "schema_version": 1,
        "model_type": "gru",
        "training_topology": _training_topology(job),
        "joint_names": list(input_joints),
        "output_joint_names": list(output_joints),
        "input_columns": list(job.input_columns),
        "input_feature_specs": input_feature_specs,
        "target_columns": [semantic_target],
        "sample_dt_s": _sample_dt_s(job),
        "normalization": normalization,
        "delay": {
            "handling": "learned",
            "training_delay_s": 0.0,
            "external_delay_s": 0.0,
        },
    }
    if "dynamic_bias" in job.input_columns:
        derivation = job.metadata.get("dynamic_bias_derivation")
        if not isinstance(derivation, Mapping):
            raise ValueError("dynamic_bias training requires derivation provenance")
        runtime["dynamic_bias_derivation"] = dict(derivation)
    if "previous_torque" in job.input_columns:
        runtime["previous_torque_derivation"] = {
            "source": "previous_target",
            "runtime_source": "previous_raw_network_output_pre_clamp",
            "initialization": "physical_zero",
            "reset": "episode",
            "value_space": "physical_torque",
        }

    needs_pd = semantic_target == "torque_residual" or "solver_pd" in job.input_columns
    physics = job.residual_physics
    if needs_pd and physics is None:
        raise ValueError("residual and solver_pd models require residual physics parameters")
    if needs_pd:
        assert physics is not None
        runtime["pd_baseline"] = {
            "kp": float(physics.stiffness),
            "kd": float(physics.damping),
            "velocity_target": "zero",
        }
    if semantic_target == "torque_residual":
        assert physics is not None
        runtime["friction_baseline"] = {
            "dry": float(physics.dry_friction),
            "viscous": float(physics.viscous_friction),
        }
    return runtime


def _fit_once(
    job: TrainingJob,
    hparams: TrainingHyperParameters,
    *,
    train_sequences: Sequence[tuple[np.ndarray, np.ndarray]],
    val_sequences: Sequence[tuple[np.ndarray, np.ndarray]],
    epochs: int,
    save_artifacts: bool,
    stats: dict[str, Any],
) -> tuple[float, TrainingArtifacts | None]:
    torch = _import_torch()
    torch.manual_seed(job.seed)
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not train_sequences:
        raise ValueError("TrainingJob requires at least one training sequence")
    input_size = train_sequences[0][0].shape[1]
    output_size = train_sequences[0][1].shape[1]
    for features, targets in (*train_sequences, *val_sequences):
        if features.ndim != 2 or features.shape[1] != input_size:
            raise ValueError("all GRU training sequences must have the same input width")
        if targets.ndim != 2 or targets.shape[1] != output_size:
            raise ValueError("all GRU training sequences must have the same output width")
        if len(features) != len(targets):
            raise ValueError("GRU input and target sequences must have the same row count")
    expected_input_size = sum(_feature_widths(job))
    _, output_joints = _runtime_axes(job)
    if input_size != expected_input_size or output_size != len(output_joints):
        raise ValueError("assembled GRU tensor dimensions do not match the configured joint axes")
    model = build_runtime_gru(
        input_size=input_size,
        output_size=output_size,
        hidden_size=hparams.hidden_size,
        num_layers=hparams.num_layers,
        dropout=hparams.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hparams.learning_rate, weight_decay=hparams.weight_decay)
    loss_fn = torch.nn.HuberLoss(delta=hparams.huber_delta)
    best_val = math.inf
    best_state = None
    history: list[dict[str, float]] = []
    eval_sequences = val_sequences or train_sequences
    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(
            torch,
            model,
            optimizer,
            loss_fn,
            train_sequences,
            hparams,
            train=True,
            device=device,
            noise_std=job.noise_std,
        )
        val_loss = _run_epoch(
            torch,
            model,
            optimizer,
            loss_fn,
            eval_sequences,
            hparams,
            train=False,
            device=device,
            noise_std=0.0,
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    if not save_artifacts:
        return best_val, None
    job.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "hybrid_residual" if job.model == "hybrid_residual" else "gru"
    joint = _safe_name(_artifact_label(job))
    checkpoint_path = job.output_dir / f"{prefix}_{joint}_best.pt"
    stats_path = job.output_dir / f"{prefix}_{joint}_stats.json"
    network_path = job.output_dir / f"{prefix}_{joint}_network.pt"
    metrics = {
        "best_val_loss": best_val,
        "history": history,
        "device": str(device),
        "hyperparameters": hparams.__dict__,
    }
    if best_state is not None:
        model.load_state_dict(best_state)
    checkpoint = {
        "model": job.model,
        "model_state_dict": model.state_dict(),
        "input_size": input_size,
        "output_size": output_size,
        "runtime_metadata": _runtime_metadata(job, stats),
        "hyperparameters": hparams.__dict__,
        "metrics": metrics,
    }
    torch.save(checkpoint, checkpoint_path)
    write_stats_json(stats_path, {**stats, "metrics": metrics})
    export_gru_checkpoint(checkpoint_path, network_path)
    return best_val, TrainingArtifacts(
        checkpoint_path=checkpoint_path,
        stats_path=stats_path,
        network_path=network_path,
        metrics=metrics,
    )


def _default_hparams() -> TrainingHyperParameters:
    return TrainingHyperParameters()


def _select_hparams(
    job: TrainingJob,
    train_sequences: Sequence[tuple[np.ndarray, np.ndarray]],
    val_sequences: Sequence[tuple[np.ndarray, np.ndarray]],
    stats: dict[str, Any],
) -> TrainingHyperParameters:
    if job.skip_optuna:
        return _default_hparams()
    try:
        import optuna
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("optuna is required unless --skip-optuna is passed") from exc

    def objective(trial):
        hparams = TrainingHyperParameters(
            hidden_size=trial.suggest_categorical("hidden_size", [32, 64, 128, 256]),
            num_layers=trial.suggest_int("num_layers", 1, 3),
            dropout=trial.suggest_float("dropout", 0.0, 0.3),
            learning_rate=trial.suggest_float("learning_rate", 1.0e-4, 5.0e-3, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1.0e-7, 1.0e-3, log=True),
            huber_delta=trial.suggest_float("huber_delta", 0.25, 2.0),
            chunk_length=trial.suggest_categorical("chunk_length", [16, 32, 64, 100]),
        )
        score, _ = _fit_once(
            job,
            hparams,
            train_sequences=train_sequences,
            val_sequences=val_sequences,
            epochs=job.trial_epochs,
            save_artifacts=False,
            stats=stats,
        )
        return score

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=job.trials)
    return TrainingHyperParameters(**study.best_params)


def _training_stats_metadata(job: TrainingJob) -> dict[str, Any]:
    input_joints, output_joints = _runtime_axes(job)
    return {
        **job.metadata,
        "model": job.model,
        "input_feature_specs": _input_feature_specs(job),
        "training_topology": _training_topology(job),
        "input_joints": list(input_joints),
        "output_joints": list(output_joints),
        "data": {
            "train_source_count": len(job.train_sources),
            "val_source_count": len(job.val_sources),
            "n_streams": job.n_streams,
        },
    }


def train_stateful_gru(job: TrainingJob) -> TrainingArtifacts:
    train_raw, val_raw = _load_sequences(job)
    input_widths = _feature_widths(job)
    output_width = len(_runtime_axes(job)[1])
    target_column = _target_column(job)
    stats = _normalization_stats(
        train_raw,
        input_columns=job.input_columns,
        target_column=target_column,
        metadata=_training_stats_metadata(job),
        input_widths=input_widths,
        target_width=output_width,
    )
    train_sequences = _normalize_sequences(
        train_raw,
        stats,
        input_columns=job.input_columns,
        target_column=target_column,
        input_widths=input_widths,
        target_width=output_width,
    )
    val_sequences = (
        _normalize_sequences(
            val_raw,
            stats,
            input_columns=job.input_columns,
            target_column=target_column,
            input_widths=input_widths,
            target_width=output_width,
        )
        if val_raw
        else []
    )
    hparams = _select_hparams(job, train_sequences, val_sequences, stats)
    _, artifacts = _fit_once(
        job,
        hparams,
        train_sequences=train_sequences,
        val_sequences=val_sequences,
        epochs=job.final_epochs,
        save_artifacts=True,
        stats=stats,
    )
    assert artifacts is not None
    return artifacts
