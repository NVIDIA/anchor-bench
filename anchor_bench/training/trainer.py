
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from anchor_bench.training.data import EPSILON_STD, write_stats_json
from anchor_bench.training.export import export_gru_checkpoint


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
    joint_type: str
    train_csvs: Sequence[Path]
    val_csvs: Sequence[Path]
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
    residual_physics: ResidualPhysics | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    train_sage_dirs: Sequence[Path] = field(default_factory=tuple)
    val_sage_dirs: Sequence[Path] = field(default_factory=tuple)
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


def _import_pandas():
    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "pandas is required for train-actuator execution. Run in the full container "
            "or install pandas in the active environment."
        ) from exc
    return pd


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "joint"


def _normalized_joint_name(value: str) -> str:
    return value.rsplit("/", 1)[-1].removesuffix("_joint")


def _joint_index(joint_names: Sequence[str], requested_name: str) -> int:
    exact = [index for index, name in enumerate(joint_names) if name == requested_name]
    if len(exact) == 1:
        return exact[0]
    normalized = _normalized_joint_name(requested_name)
    matches = [
        index for index, name in enumerate(joint_names)
        if _normalized_joint_name(name) == normalized
    ]
    if len(matches) != 1:
        if not matches:
            raise ValueError(f"training joint {requested_name!r} was not found in SAGE joint_list.txt")
        raise ValueError(
            f"training joint {requested_name!r} matches multiple SAGE joints: "
            + ", ".join(joint_names[index] for index in matches)
        )
    return matches[0]


def _load_sequences(job: TrainingJob) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[tuple[np.ndarray, np.ndarray]]]:
    sage_joint_set: frozenset[str] | None = None

    def prepare(raw_columns: dict[str, np.ndarray], source: Path) -> tuple[np.ndarray, np.ndarray]:
        columns = {name: np.asarray(values, dtype=np.float32) for name, values in raw_columns.items()}
        for name, values in columns.items():
            if values.ndim != 1:
                raise ValueError(f"{source} column {name!r} must be one-dimensional")
        lengths = {len(values) for values in columns.values()}
        if len(lengths) != 1:
            raise ValueError(f"{source} training columns have mismatched row counts")
        if not lengths or next(iter(lengths)) == 0:
            raise ValueError(f"training source has no rows: {source}")

        if "solver_pd" in job.input_columns and "solver_pd" not in columns:
            physics = job.residual_physics
            if physics is None:
                raise ValueError("solver_pd input requires residual physics parameters")
            try:
                error = columns["position_error"]
                velocity = columns["velocity"]
            except KeyError as exc:
                raise ValueError("solver_pd input requires position_error and velocity columns") from exc
            columns["solver_pd"] = (physics.stiffness * error - physics.damping * velocity).astype(np.float32)

        missing_inputs = [name for name in job.input_columns if name not in columns]
        if missing_inputs:
            raise ValueError(f"{source} missing required training columns: {', '.join(missing_inputs)}")
        features = np.column_stack([columns[name] for name in job.input_columns]).astype(np.float32, copy=False)

        if job.model == "hybrid_residual":
            physics = job.residual_physics or ResidualPhysics()
            try:
                error = columns["position_error"]
                velocity = columns["velocity"]
                torque = columns[job.source_target_columns[0]]
            except KeyError as exc:
                raise ValueError(
                    "hybrid_residual training requires position_error, velocity, and torque"
                ) from exc
            solver_pd = physics.stiffness * error - physics.damping * velocity
            friction = physics.dry_friction * np.sign(velocity) + physics.viscous_friction * velocity
            target = (torque - solver_pd - friction).astype(np.float32)[:, None]
        else:
            missing_targets = [name for name in job.target_columns if name not in columns]
            if missing_targets:
                raise ValueError(f"{source} missing required target columns: {', '.join(missing_targets)}")
            target = np.column_stack([columns[name] for name in job.target_columns]).astype(np.float32, copy=False)

        if not np.isfinite(features).all() or not np.isfinite(target).all():
            raise ValueError(f"{source} contains non-finite training values")
        return features, target

    def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
        pd = _import_pandas()
        frame = pd.read_csv(path)
        required_columns = set(job.input_columns)
        if "solver_pd" in job.input_columns and "solver_pd" not in frame.columns:
            required_columns.update(("position_error", "velocity"))
        if job.model == "hybrid_residual":
            required_columns.update(("position_error", "velocity", *job.source_target_columns))
        else:
            required_columns.update(job.target_columns)
        raw_columns = {
            name: frame[name].to_numpy()
            for name in required_columns
            if name in frame.columns
        }
        return prepare(raw_columns, path)

    def load_sage(path: Path) -> tuple[np.ndarray, np.ndarray]:
        nonlocal sage_joint_set
        from anchor_bench.data.sage import load_real_data

        if job.source_target_columns != ["torque"]:
            raise ValueError("SAGE actuator training supports target_column='torque'")
        provider = job.dynamic_bias_provider
        if provider is None:
            raise ValueError("SAGE dynamic_bias training requires a bias provider")
        try:
            sample_dt_s = float(job.metadata["sample_dt_s"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("SAGE training requires sample_dt_s metadata") from exc

        dataset = load_real_data(
            path,
            target_dt=sample_dt_s,
            require_state_dynamics=True,
            hold_controls=True,
        )
        normalized_names = tuple(
            _normalized_joint_name(name) for name in dataset.joint_names
        )
        if len(set(normalized_names)) != len(normalized_names):
            raise ValueError(f"{path} has duplicate normalized joint names")
        episode_joint_set = frozenset(normalized_names)
        if sage_joint_set is None:
            sage_joint_set = episode_joint_set
            derivation = job.metadata.get("dynamic_bias_derivation")
            if isinstance(derivation, dict):
                derivation["trajectory_joint_names"] = sorted(episode_joint_set)
        elif episode_joint_set != sage_joint_set:
            missing = sorted(sage_joint_set - episode_joint_set)
            extra = sorted(episode_joint_set - sage_joint_set)
            details = []
            if missing:
                details.append(f"missing {', '.join(missing)}")
            if extra:
                details.append(f"extra {', '.join(extra)}")
            raise ValueError(
                f"{path} does not use the common SAGE joint set: {'; '.join(details)}"
            )
        positions = np.asarray(dataset.state_positions, dtype=np.float32)
        velocities = np.asarray(dataset.state_velocities, dtype=np.float32)
        torques = np.asarray(dataset.state_torques, dtype=np.float32)
        controls = np.asarray(dataset.control_positions, dtype=np.float32)
        expected_shape = (len(dataset.state_timestamps), len(dataset.joint_names))
        for name, values in (
            ("state_positions", positions),
            ("state_velocities", velocities),
            ("state_torques", torques),
            ("control_positions", controls),
        ):
            if values.shape != expected_shape:
                raise ValueError(f"{path} {name} has shape {values.shape}, expected {expected_shape}")

        dynamic_bias = np.asarray(
            provider.compute(
                joint_names=dataset.joint_names,
                positions=positions,
                velocities=velocities,
            ),
            dtype=np.float32,
        )
        if dynamic_bias.shape != expected_shape:
            raise ValueError(
                f"dynamic-bias provider returned shape {dynamic_bias.shape}, expected {expected_shape}"
            )

        joint_index = _joint_index(dataset.joint_names, job.joint_type)
        return prepare(
            {
                "position": positions[:, joint_index],
                "position_error": controls[:, joint_index] - positions[:, joint_index],
                "velocity": velocities[:, joint_index],
                "dynamic_bias": dynamic_bias[:, joint_index],
                "torque": torques[:, joint_index],
            },
            path,
        )

    if job.train_sage_dirs or job.val_sage_dirs:
        if job.train_csvs or job.val_csvs:
            raise ValueError("TrainingJob cannot mix scalar CSV and SAGE sources")
        train = [load_sage(Path(path)) for path in job.train_sage_dirs]
        val = [load_sage(Path(path)) for path in job.val_sage_dirs]
    else:
        train = [load_csv(Path(path)) for path in job.train_csvs]
        val = [load_csv(Path(path)) for path in job.val_csvs]
    if not train:
        raise ValueError("TrainingJob requires at least one training sequence")
    return train, val


def _normalization_stats(
    sequences: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    input_columns: Sequence[str],
    target_columns: Sequence[str],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    inputs = np.concatenate([item[0] for item in sequences], axis=0)
    targets = np.concatenate([item[1] for item in sequences], axis=0)
    input_mean = inputs.mean(axis=0)
    input_std = np.maximum(inputs.std(axis=0), EPSILON_STD)
    target_mean = targets.mean(axis=0)
    target_std = np.maximum(targets.std(axis=0), EPSILON_STD)
    stats = {
        "schema_version": 1,
        "input_columns": list(input_columns),
        "target_columns": list(target_columns),
        "normalization": {
            "inputs": {
                "mean": {column: float(input_mean[idx]) for idx, column in enumerate(input_columns)},
                "std": {column: float(input_std[idx]) for idx, column in enumerate(input_columns)},
            },
            "targets": {
                "mean": {column: float(target_mean[idx]) for idx, column in enumerate(target_columns)},
                "std": {column: float(target_std[idx]) for idx, column in enumerate(target_columns)},
            },
        },
    }
    stats.update(metadata)
    return stats


def _normalize_sequences(
    sequences: Sequence[tuple[np.ndarray, np.ndarray]],
    stats: dict[str, Any],
    *,
    input_columns: Sequence[str],
    target_columns: Sequence[str],
) -> list[tuple[np.ndarray, np.ndarray]]:
    input_norm = stats["normalization"]["inputs"]
    target_norm = stats["normalization"]["targets"]
    input_mean = np.array([input_norm["mean"][column] for column in input_columns], dtype=np.float32)
    input_std = np.array([input_norm["std"][column] for column in input_columns], dtype=np.float32)
    target_mean = np.array([target_norm["mean"][column] for column in target_columns], dtype=np.float32)
    target_std = np.array([target_norm["std"][column] for column in target_columns], dtype=np.float32)
    return [((inputs - input_mean) / input_std, (targets - target_mean) / target_std) for inputs, targets in sequences]


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
                xb = x[start:start + hparams.chunk_length].unsqueeze(0)
                yb = y[start:start + hparams.chunk_length].unsqueeze(0)
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
    if len(job.target_columns) != 1:
        raise ValueError("GRU runtime artifacts require exactly one target column")

    try:
        sample_dt_s = float(job.metadata["sample_dt_s"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("training metadata must define sample_dt_s") from exc
    if not math.isfinite(sample_dt_s) or sample_dt_s <= 0.0:
        raise ValueError("sample_dt_s must be finite and greater than zero")

    delay = job.metadata.get("delay")
    if not isinstance(delay, dict):
        raise ValueError("training metadata must define delay handling")

    source_target = job.target_columns[0]
    target = "torque_residual" if job.model == "hybrid_residual" else "torque"
    normalization = stats["normalization"]
    runtime = {
        "schema_version": 1,
        "model_type": "gru",
        "joint_names": [job.joint_type],
        "input_columns": list(job.input_columns),
        "target_columns": [target],
        "sample_dt_s": sample_dt_s,
        "normalization": {
            "inputs": normalization["inputs"],
            "targets": {
                "mean": {target: normalization["targets"]["mean"][source_target]},
                "std": {target: normalization["targets"]["std"][source_target]},
            },
        },
        "delay": dict(delay),
    }
    if "dynamic_bias" in job.input_columns:
        derivation = job.metadata.get("dynamic_bias_derivation")
        if not isinstance(derivation, dict):
            raise ValueError("dynamic_bias training metadata must define dynamic_bias_derivation")
        runtime["dynamic_bias_derivation"] = dict(derivation)

    needs_pd = target == "torque_residual" or "solver_pd" in job.input_columns
    physics = job.metadata.get("physics_params")
    if needs_pd:
        if not isinstance(physics, dict):
            raise ValueError("residual and solver_pd models require physics_params")
        runtime["pd_baseline"] = {
            "kp": float(physics["stiffness"]),
            "kd": float(physics["damping"]),
            "velocity_target": "zero",
        }
    if target == "torque_residual":
        assert isinstance(physics, dict)
        runtime["friction_baseline"] = {
            "dry": float(physics["dry_friction"]),
            "viscous": float(physics["viscous_friction"]),
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
    model = _build_model(torch, len(job.input_columns), len(job.target_columns), hparams).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hparams.learning_rate, weight_decay=hparams.weight_decay)
    loss_fn = torch.nn.HuberLoss(delta=hparams.huber_delta)
    best_val = math.inf
    best_state = None
    history: list[dict[str, float]] = []
    eval_sequences = val_sequences or train_sequences
    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(
            torch, model, optimizer, loss_fn, train_sequences, hparams,
            train=True, device=device, noise_std=job.noise_std,
        )
        val_loss = _run_epoch(
            torch, model, optimizer, loss_fn, eval_sequences, hparams,
            train=False, device=device, noise_std=0.0,
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    if not save_artifacts:
        return best_val, None
    job.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "hybrid_residual" if job.model == "hybrid_residual" else "gru"
    joint = _safe_name(job.joint_type)
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
        "joint_type": job.joint_type,
        "model_state_dict": model.state_dict(),
        "input_columns": job.input_columns,
        "target_columns": job.target_columns,
        "normalization": stats["normalization"],
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
    return TrainingHyperParameters(**study.best_params)


def train_stateful_gru(job: TrainingJob) -> TrainingArtifacts:
    train_raw, val_raw = _load_sequences(job)
    stats = _normalization_stats(
        train_raw,
        input_columns=job.input_columns,
        target_columns=job.target_columns,
        metadata={
            **job.metadata,
            "model": job.model,
            "joint_type": job.joint_type,
            "data": {
                "source_format": "sage" if job.train_sage_dirs else "scalar_csv",
                "train_sequence_count": len(job.train_sage_dirs or job.train_csvs),
                "val_sequence_count": len(job.val_sage_dirs or job.val_csvs),
                "train_sage_count": len(job.train_sage_dirs),
                "val_sage_count": len(job.val_sage_dirs),
                "train_csv_count": len(job.train_csvs),
                "val_csv_count": len(job.val_csvs),
                "n_streams": job.n_streams,
            },
        },
    )
    train_sequences = _normalize_sequences(
        train_raw, stats, input_columns=job.input_columns, target_columns=job.target_columns
    )
    val_sequences = _normalize_sequences(
        val_raw, stats, input_columns=job.input_columns, target_columns=job.target_columns
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
