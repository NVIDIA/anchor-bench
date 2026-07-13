
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from anchor_bench.training.data import (
    EPSILON_STD,
    TrainingMapping,
    TrainingSource,
    load_training_episode,
    load_training_motion,
    write_stats_json,
)
from anchor_bench.training.export import export_gru_checkpoint


@dataclass(frozen=True)
class PhysicsBaselines:
    """One PD vector plus optional output-domain friction."""

    kp: float | tuple[float, ...]
    kd: float | tuple[float, ...]
    dry_friction: float | tuple[float, ...] | None = None
    viscous_friction: float | tuple[float, ...] | None = None


@dataclass(frozen=True)
class TrainingHyperParameters:
    hidden_size: int = 192
    num_layers: int = 2
    dropout: float = 0.225
    learning_rate: float = 1.0e-3
    weight_decay: float = 0.0
    huber_delta: float = 1.0
    chunk_length: int = 16


@dataclass(frozen=True)
class _PreviousTorqueFeedback:
    """Normalization values needed for autoregressive torque feedback."""

    input_slice: slice
    input_mean: np.ndarray
    input_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray


@dataclass(frozen=True)
class TrainingJob:
    """One shared-weight GRU training job over source/mapping sequences."""

    model: str
    artifact_name: str
    mappings: tuple[TrainingMapping, ...]
    train_sources: Sequence[TrainingSource]
    val_sources: Sequence[TrainingSource]
    output_dir: Path
    input_columns: list[str]
    target_columns: list[str]
    source_target_columns: list[str]
    seed: int
    final_epochs: int
    trial_epochs: int
    trials: int
    skip_optuna: bool
    n_streams: int
    noise_std: float
    physics_baselines: PhysicsBaselines | None = None
    dynamic_bias_provider: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    hidden_size: int | None = None
    num_layers: int | None = None
    dropout: float | None = None


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
    """Return a filesystem-safe artifact label."""

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "joint"


def _physics_vector(
    value: float | tuple[float, ...] | None, width: int, name: str
) -> np.ndarray:
    """Broadcast a scalar baseline or validate one value per model slot."""

    if value is None:
        raise ValueError(f"{name} is required")
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0:
        array = np.full(width, float(array), dtype=np.float32)
    if array.shape != (width,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain {width} finite value(s)")
    return array


def _mapping_widths(job: TrainingJob) -> tuple[int, int]:
    """Validate shared mapping shapes and return one mapping axis size."""

    if not job.mappings:
        raise ValueError("TrainingJob requires at least one joint mapping")
    input_width = len(job.mappings[0].input_joints)
    output_width = len(job.mappings[0].output_joints)
    if any(
        len(mapping.input_joints) != input_width or len(mapping.output_joints) != output_width
        for mapping in job.mappings
    ):
        raise ValueError("all training mappings must have identical input and output widths")
    return input_width, output_width


def _feature_widths(job: TrainingJob) -> list[int]:
    """Return each feature block width in configured order."""

    input_width, output_width = _mapping_widths(job)
    return [output_width if column == "previous_torque" else input_width for column in job.input_columns]


def _load_sequences(
    job: TrainingJob,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[tuple[np.ndarray, np.ndarray]]]:
    """Assemble one independent recurrent sequence per source and mapping."""

    input_width, output_width = _mapping_widths(job)

    union_joints = tuple(dict.fromkeys(name for mapping in job.mappings for name in mapping.input_joints))
    bias_cache: dict[Path, np.ndarray] = {}
    bias_indices = {name: index for index, name in enumerate(union_joints)}

    def dynamic_bias(source: TrainingSource, mapping: TrainingMapping) -> np.ndarray:
        """Evaluate one full episode once and select this mapping's input slots."""

        cached = bias_cache.get(source.path)
        if cached is None:
            if job.dynamic_bias_provider is None:
                raise ValueError("dynamic_bias input requires a dynamic-bias provider")
            motion = load_training_motion(source.path)
            cached = np.asarray(
                job.dynamic_bias_provider.compute(motion, union_joints), dtype=np.float32
            )
            if cached.shape != (len(motion.position.time), len(union_joints)):
                raise ValueError("dynamic-bias provider returned an inconsistent episode shape")
            bias_cache[source.path] = cached
        return cached[:, [bias_indices[name] for name in mapping.input_joints]]

    def load_one(source: TrainingSource, mapping: TrainingMapping) -> tuple[np.ndarray, np.ndarray]:
        """Build feature-major inputs and final targets for one sequence."""

        episode = load_training_episode(source, mapping)
        if episode.torque is None:
            raise ValueError(f"{source.path} is missing measured torque")
        needs_velocity = (
            job.model == "hybrid_residual"
            or "velocity" in job.input_columns
            or "solver_pd" in job.input_columns
            or "dynamic_bias" in job.input_columns
        )
        if needs_velocity and episode.velocity is None:
            raise ValueError(f"{source.path} is missing measured velocity")

        baselines = job.physics_baselines
        if baselines is None and (
            job.model == "hybrid_residual" or "solver_pd" in job.input_columns
        ):
            raise ValueError("PD features and residual targets require physics baselines")
        solver_pd: np.ndarray | None = None
        if "solver_pd" in job.input_columns:
            assert baselines is not None and episode.velocity is not None
            kp = _physics_vector(baselines.kp, input_width, "kp")
            kd = _physics_vector(baselines.kd, input_width, "kd")
            solver_pd = episode.position_error * kp - episode.velocity * kd

        target = np.asarray(episode.torque, dtype=np.float32)
        if job.model == "hybrid_residual":
            assert baselines is not None and episode.velocity is not None
            output_indices = np.asarray(
                [mapping.input_joints.index(name) for name in mapping.output_joints],
                dtype=int,
            )
            velocity = episode.velocity[:, output_indices]
            if solver_pd is None:
                kp = _physics_vector(baselines.kp, output_width, "kp")
                kd = _physics_vector(baselines.kd, output_width, "kd")
                output_pd = episode.position_error[:, output_indices] * kp - velocity * kd
            else:
                output_pd = solver_pd[:, output_indices]
            dry = _physics_vector(baselines.dry_friction, output_width, "dry_friction")
            viscous = _physics_vector(
                baselines.viscous_friction, output_width, "viscous_friction"
            )
            friction = dry * np.sign(velocity) + viscous * velocity
            target = target - output_pd - friction

        # Shifted targets define stable input normalization only. The epoch
        # runner replaces every row with the prior raw network prediction.
        previous = np.zeros_like(target)
        previous[1:] = target[:-1]
        blocks: dict[str, np.ndarray | None] = {
            "dynamic_bias": dynamic_bias(source, mapping) if "dynamic_bias" in job.input_columns else None,
            "position": episode.position,
            "position_error": episode.position_error,
            "previous_torque": previous,
            "solver_pd": solver_pd,
            "velocity": episode.velocity,
        }
        missing = [column for column in job.input_columns if blocks.get(column) is None]
        if missing:
            raise ValueError(f"{source.path} cannot derive input features: {', '.join(missing)}")
        features = np.concatenate([blocks[column] for column in job.input_columns], axis=1).astype(np.float32)
        target = target.astype(np.float32)
        if not np.all(np.isfinite(features)) or not np.all(np.isfinite(target)):
            raise ValueError(f"{source.path} contains non-finite training values")
        return features, target

    train = [load_one(source, mapping) for source in job.train_sources for mapping in job.mappings]
    val = [load_one(source, mapping) for source in job.val_sources for mapping in job.mappings]
    return train, val


def _scalar_or_list(values: np.ndarray) -> float | list[float]:
    """Serialize a one-wide statistic as a scalar and wider axes as lists."""

    result = [float(value) for value in values]
    return result[0] if len(result) == 1 else result


def _normalization_stats(
    sequences: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    input_columns: Sequence[str],
    target_columns: Sequence[str],
    input_widths: Sequence[int],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Pool per-slot normalization across every training sequence."""

    if len(target_columns) != 1:
        raise ValueError("GRU training requires one target semantic")
    inputs = np.concatenate([item[0] for item in sequences], axis=0)
    targets = np.concatenate([item[1] for item in sequences], axis=0)
    if len(input_widths) != len(input_columns) or sum(input_widths) != inputs.shape[1]:
        raise ValueError("input feature widths do not match assembled tensors")
    input_mean = inputs.mean(axis=0)
    input_std = np.maximum(inputs.std(axis=0), EPSILON_STD)
    target_mean = targets.mean(axis=0)
    target_std = np.maximum(targets.std(axis=0), EPSILON_STD)
    offsets = np.cumsum((0, *input_widths))
    target = target_columns[0]
    stats = {
        "schema_version": 3,
        "input_columns": list(input_columns),
        "target_columns": list(target_columns),
        "normalization": {
            "inputs": {
                "mean": {
                    column: _scalar_or_list(input_mean[offsets[index] : offsets[index + 1]])
                    for index, column in enumerate(input_columns)
                },
                "std": {
                    column: _scalar_or_list(input_std[offsets[index] : offsets[index + 1]])
                    for index, column in enumerate(input_columns)
                },
            },
            "targets": {
                "mean": {target: _scalar_or_list(target_mean)},
                "std": {target: _scalar_or_list(target_std)},
            },
        },
    }
    stats.update(metadata)
    return stats


def _normalization_vector(
    values: dict[str, Any],
    columns: Sequence[str],
    widths: Sequence[int],
) -> np.ndarray:
    """Flatten scalar-or-list statistics in feature-major order."""

    result: list[float] = []
    for column, width in zip(columns, widths, strict=True):
        value = values[column]
        items = value if isinstance(value, list) else [value]
        if len(items) != width:
            raise ValueError(f"normalization for {column} must contain {width} value(s)")
        result.extend(float(item) for item in items)
    return np.asarray(result, dtype=np.float32)


def _normalize_sequences(
    sequences: Sequence[tuple[np.ndarray, np.ndarray]],
    stats: dict[str, Any],
    *,
    input_columns: Sequence[str],
    target_columns: Sequence[str],
    input_widths: Sequence[int],
    target_width: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Normalize flattened tensors using domain-sized statistics."""

    input_norm = stats["normalization"]["inputs"]
    target_norm = stats["normalization"]["targets"]
    if len(target_columns) != 1:
        raise ValueError("GRU training requires one target semantic")

    input_mean = _normalization_vector(input_norm["mean"], input_columns, input_widths)
    input_std = _normalization_vector(input_norm["std"], input_columns, input_widths)
    target_mean = _normalization_vector(target_norm["mean"], target_columns, (target_width,))
    target_std = _normalization_vector(target_norm["std"], target_columns, (target_width,))
    return [((inputs - input_mean) / input_std, (targets - target_mean) / target_std) for inputs, targets in sequences]


def _previous_torque_feedback(
    job: TrainingJob,
    stats: dict[str, Any],
) -> _PreviousTorqueFeedback | None:
    """Locate previous torque and return its input/output normalization."""

    if "previous_torque" not in job.input_columns:
        return None
    widths = _feature_widths(job)
    index = job.input_columns.index("previous_torque")
    offsets = np.cumsum((0, *widths))
    input_slice = slice(int(offsets[index]), int(offsets[index + 1]))
    output_width = len(job.mappings[0].output_joints)
    if input_slice.stop - input_slice.start != output_width:
        raise ValueError("previous_torque width must match model output width")

    normalization = stats["normalization"]
    input_mean = _normalization_vector(
        normalization["inputs"]["mean"], job.input_columns, widths
    )
    input_std = _normalization_vector(
        normalization["inputs"]["std"], job.input_columns, widths
    )
    target_mean = _normalization_vector(
        normalization["targets"]["mean"], job.target_columns, (output_width,)
    )
    target_std = _normalization_vector(
        normalization["targets"]["std"], job.target_columns, (output_width,)
    )
    return _PreviousTorqueFeedback(
        input_slice=input_slice,
        input_mean=input_mean[input_slice],
        input_std=input_std[input_slice],
        target_mean=target_mean,
        target_std=target_std,
    )


def _build_model(torch, input_size: int, output_size: int, hparams: TrainingHyperParameters):
    class TorqueGRU(torch.nn.Module):
        def __init__(self):
            super().__init__()
            dropout = hparams.dropout if hparams.num_layers > 1 else 0.0
            self.gru = torch.nn.GRU(
                input_size=input_size,
                hidden_size=hparams.hidden_size,
                num_layers=hparams.num_layers,
                dropout=dropout,
                batch_first=True,
            )
            self.head = torch.nn.Linear(hparams.hidden_size, output_size)

        def forward(self, x, hidden=None):
            out, hidden = self.gru(x, hidden)
            return self.head(out), hidden

    return TorqueGRU()


def _run_epoch(
    torch,
    model,
    optimizer,
    loss_fn,
    sequences,
    hparams,
    *,
    train: bool,
    device,
    noise_std: float,
    previous_torque_feedback: _PreviousTorqueFeedback | None = None,
) -> float:
    """Run one epoch with independent sequences and detached TBPTT chunks."""

    model.train(train)
    losses: list[float] = []
    if previous_torque_feedback is not None:
        feedback = previous_torque_feedback
        previous_input_mean = torch.as_tensor(
            feedback.input_mean, dtype=torch.float32, device=device
        ).reshape(1, 1, -1)
        previous_input_std = torch.as_tensor(
            feedback.input_std, dtype=torch.float32, device=device
        ).reshape(1, 1, -1)
        target_mean = torch.as_tensor(
            feedback.target_mean, dtype=torch.float32, device=device
        ).reshape(1, 1, -1)
        target_std = torch.as_tensor(
            feedback.target_std, dtype=torch.float32, device=device
        ).reshape(1, 1, -1)

    grad_context = torch.enable_grad() if train else torch.no_grad()
    with grad_context:
        for features, targets in sequences:
            hidden = None
            previous_physical = (
                torch.zeros_like(target_mean)
                if previous_torque_feedback is not None
                else None
            )
            x = torch.as_tensor(features, dtype=torch.float32, device=device)
            y = torch.as_tensor(targets, dtype=torch.float32, device=device)
            for start in range(0, x.shape[0], hparams.chunk_length):
                end = start + hparams.chunk_length
                xb = x[start:end].unsqueeze(0)
                yb = y[start:end].unsqueeze(0)
                if previous_torque_feedback is None:
                    if train and noise_std > 0.0:
                        xb = xb + torch.randn_like(xb) * noise_std
                    pred, hidden = model(xb, hidden)
                else:
                    assert previous_physical is not None
                    step_predictions = []
                    for offset in range(xb.shape[1]):
                        step = xb[:, offset : offset + 1, :]
                        if train and noise_std > 0.0:
                            step = step + torch.randn_like(step) * noise_std
                        normalized_previous = (
                            previous_physical - previous_input_mean
                        ) / previous_input_std
                        step = torch.cat(
                            (
                                step[..., : feedback.input_slice.start],
                                normalized_previous,
                                step[..., feedback.input_slice.stop :],
                            ),
                            dim=-1,
                        )
                        step_prediction, hidden = model(step, hidden)
                        step_predictions.append(step_prediction)
                        previous_physical = step_prediction * target_std + target_mean
                    pred = torch.cat(step_predictions, dim=1)
                loss = loss_fn(pred, yb)
                if train:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    hidden = hidden.detach()
                    if previous_physical is not None:
                        previous_physical = previous_physical.detach()
                losses.append(float(loss.detach().cpu()))
    return float(sum(losses) / max(1, len(losses)))


def _runtime_metadata(job: TrainingJob, stats: dict[str, Any]) -> dict[str, Any]:
    """Build the self-contained schema-v3 contract embedded in TorchScript."""

    input_width = len(job.mappings[0].input_joints)
    output_width = len(job.mappings[0].output_joints)
    input_widths = _feature_widths(job)
    target = job.target_columns[0]
    sample_dt_s = float(job.metadata.get("sample_dt_s", 0.0))
    if not math.isfinite(sample_dt_s) or sample_dt_s <= 0.0:
        raise ValueError("training metadata requires a positive finite sample_dt_s")
    topology = "siso" if input_width == output_width == 1 else ("miso" if output_width == 1 else "mimo")
    metadata: dict[str, Any] = {
        "schema_version": 3,
        "model_type": "gru",
        "training_topology": topology,
        "joint_mappings": [
            {
                "input_joints": list(mapping.input_joints),
                "output_joints": list(mapping.output_joints),
            }
            for mapping in job.mappings
        ],
        "input_columns": list(job.input_columns),
        "input_feature_specs": [
            {
                "name": column,
                "domain": "output_joints" if column == "previous_torque" else "input_joints",
                "channels": ["value"],
            }
            for column in job.input_columns
        ],
        "target_columns": [target],
        "input_size": sum(input_widths),
        "output_size": output_width,
        "sample_dt_s": sample_dt_s,
        "normalization": stats["normalization"],
        "delay": {"handling": "learned", "external_delay_s": 0.0},
    }
    if "previous_torque" in job.input_columns:
        metadata["previous_torque_derivation"] = {
            "source": "previous_raw_network_output_pre_clamp",
            "runtime_source": "previous_raw_network_output_pre_clamp",
            "initialization": "physical_zero",
            "reset_boundaries": ["recorded_episode", "joint_mapping"],
            "value_space": "residual_torque" if target == "torque_residual" else "physical_torque",
        }

    baselines = job.physics_baselines
    if target == "torque_residual" or "solver_pd" in job.input_columns:
        if baselines is None:
            raise ValueError("PD features and residual targets require physics baselines")
        pd_is_input = "solver_pd" in job.input_columns
        pd_width = input_width if pd_is_input else output_width
        metadata["pd_baseline"] = {
            "domain": "input_joints" if pd_is_input else "output_joints",
            "kp": _scalar_or_list(_physics_vector(baselines.kp, pd_width, "kp")),
            "kd": _scalar_or_list(_physics_vector(baselines.kd, pd_width, "kd")),
            "velocity_target": "zero",
        }
    if target == "torque_residual":
        assert baselines is not None
        metadata["friction_baseline"] = {
            "domain": "output_joints",
            "dry": _scalar_or_list(
                _physics_vector(baselines.dry_friction, output_width, "dry_friction")
            ),
            "viscous": _scalar_or_list(
                _physics_vector(baselines.viscous_friction, output_width, "viscous_friction")
            ),
        }
    return metadata


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
    """Fit one hyperparameter choice and optionally write final artifacts."""

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
            raise ValueError("all GRU input sequences must have the same width")
        if targets.ndim != 2 or targets.shape[1] != output_size or len(targets) != len(features):
            raise ValueError("all GRU target sequences must match input rows and output width")
    expected_input_size = sum(_feature_widths(job))
    expected_output_size = len(job.mappings[0].output_joints)
    if (input_size, output_size) != (expected_input_size, expected_output_size):
        raise ValueError("assembled GRU tensors do not match configured mapping dimensions")
    model = _build_model(torch, input_size, output_size, hparams).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hparams.learning_rate, weight_decay=hparams.weight_decay)
    loss_fn = torch.nn.HuberLoss(delta=hparams.huber_delta)
    best_val = math.inf
    best_state = None
    history: list[dict[str, float]] = []
    eval_sequences = val_sequences or train_sequences
    previous_feedback = _previous_torque_feedback(job, stats)
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
            previous_torque_feedback=previous_feedback,
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
            previous_torque_feedback=previous_feedback,
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    if not save_artifacts:
        return best_val, None
    job.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "hybrid_residual" if job.model == "hybrid_residual" else "gru"
    artifact = _safe_name(job.artifact_name)
    checkpoint_path = job.output_dir / f"{prefix}_{artifact}_best.pt"
    stats_path = job.output_dir / f"{prefix}_{artifact}_stats.json"
    network_path = job.output_dir / f"{prefix}_{artifact}_network.pt"
    metrics = {
        "best_val_loss": best_val,
        "history": history,
        "device": str(device),
        "hyperparameters": hparams.__dict__,
    }
    if best_state is not None:
        model.load_state_dict(best_state)
    runtime_metadata = _runtime_metadata(job, stats)
    checkpoint = {
        "model": job.model,
        "artifact_name": job.artifact_name,
        "model_state_dict": model.state_dict(),
        "input_size": input_size,
        "output_size": output_size,
        "input_columns": job.input_columns,
        "target_columns": job.target_columns,
        "normalization": stats["normalization"],
        "runtime_metadata": runtime_metadata,
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


def _default_hparams(job: TrainingJob) -> TrainingHyperParameters:
    """Apply fixed architecture overrides to the benchmark defaults."""

    defaults = TrainingHyperParameters()
    num_layers = job.num_layers if job.num_layers is not None else defaults.num_layers
    dropout = job.dropout if job.dropout is not None else defaults.dropout
    return TrainingHyperParameters(
        hidden_size=job.hidden_size if job.hidden_size is not None else defaults.hidden_size,
        num_layers=num_layers,
        dropout=dropout if num_layers > 1 else 0.0,
    )


def _select_hparams(
    job: TrainingJob,
    train_sequences: Sequence[tuple[np.ndarray, np.ndarray]],
    val_sequences: Sequence[tuple[np.ndarray, np.ndarray]],
    stats: dict[str, Any],
) -> TrainingHyperParameters:
    if job.skip_optuna:
        return _default_hparams(job)
    try:
        import optuna
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("optuna is required unless --skip-optuna is passed") from exc

    def objective(trial):
        hidden_size = (
            job.hidden_size
            if job.hidden_size is not None
            else trial.suggest_categorical("hidden_size", [32, 64, 128, 256])
        )
        num_layers = (
            job.num_layers
            if job.num_layers is not None
            else trial.suggest_int("num_layers", 1, 3)
        )
        dropout = (
            0.0
            if num_layers == 1
            else job.dropout
            if job.dropout is not None
            else trial.suggest_float("dropout", 0.0, 0.3)
        )
        hparams = TrainingHyperParameters(
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            learning_rate=trial.suggest_float("learning_rate", 1.0e-4, 5.0e-3, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1.0e-7, 1.0e-3, log=True),
            huber_delta=trial.suggest_float("huber_delta", 0.25, 2.0),
            chunk_length=trial.suggest_categorical("chunk_length", [16, 32, 64, 100]),
        )
        score, _ = _fit_once(
            job, hparams,
            train_sequences=train_sequences,
            val_sequences=val_sequences,
            epochs=job.trial_epochs,
            save_artifacts=False,
            stats=stats,
        )
        return score

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=job.trials)
    selected = dict(study.best_params)
    for name in ("hidden_size", "num_layers", "dropout"):
        override = getattr(job, name)
        if override is not None:
            selected[name] = override
    if selected["num_layers"] == 1:
        selected["dropout"] = 0.0
    return TrainingHyperParameters(**selected)


def train_stateful_gru(job: TrainingJob) -> TrainingArtifacts:
    """Train one shared stateful GRU and export all three artifacts."""

    train_raw, val_raw = _load_sequences(job)
    input_widths = _feature_widths(job)
    target_width = len(job.mappings[0].output_joints)
    stats = _normalization_stats(
        train_raw,
        input_columns=job.input_columns,
        target_columns=job.target_columns,
        input_widths=input_widths,
        metadata={
            **job.metadata,
            "model": job.model,
            "artifact_name": job.artifact_name,
            "joint_mappings": [
                {"input_joints": list(mapping.input_joints), "output_joints": list(mapping.output_joints)}
                for mapping in job.mappings
            ],
            "data": {
                "train_source_count": len(job.train_sources),
                "val_source_count": len(job.val_sources),
                "sequence_count": len(job.mappings) * (len(job.train_sources) + len(job.val_sources)),
                "n_streams": job.n_streams,
            },
        },
    )
    train_sequences = _normalize_sequences(
        train_raw,
        stats,
        input_columns=job.input_columns,
        target_columns=job.target_columns,
        input_widths=input_widths,
        target_width=target_width,
    )
    val_sequences = _normalize_sequences(
        val_raw,
        stats,
        input_columns=job.input_columns,
        target_columns=job.target_columns,
        input_widths=input_widths,
        target_width=target_width,
    ) if val_raw else []
    hparams = _select_hparams(job, train_sequences, val_sequences, stats)
    _, artifacts = _fit_once(
        job, hparams,
        train_sequences=train_sequences,
        val_sequences=val_sequences,
        epochs=job.final_epochs,
        save_artifacts=True,
        stats=stats,
    )
    assert artifacts is not None
    return artifacts
