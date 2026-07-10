"""Stateful GRU fitting over mapping-specific actuator episodes."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from anchor_bench.training.data import write_stats_json
from anchor_bench.training.episode import FeatureBlock, TrainingEpisode
from anchor_bench.training.export import build_runtime_gru, export_gru_checkpoint
from anchor_bench.training.metadata import (
    DynamicBiasDerivation,
    GRURecurrentDimensions,
    NormalizationValues,
    PreviousTorqueDerivation,
    RuntimeFeature,
    RuntimeGRUMetadata,
    RuntimeHashes,
    RuntimeMapping,
    RuntimeNormalization,
    StatValue,
)
from anchor_bench.training.schema import (
    ResolvedTrainingMapping,
    TrainingFeature,
    TrainingFit,
    TrainingNetwork,
    TrainingTarget,
)


_DYNAMIC_BIAS_SOURCE = "newton.solvers.SolverMuJoCo.eval_dynamic_bias"


@dataclass(frozen=True)
class TrainingHyperParameters:
    """Concrete architecture and optimizer values used for one fit."""

    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.0
    learning_rate: float = 1.0e-3
    weight_decay: float = 0.0
    huber_delta: float = 1.0
    chunk_length: int = 16


@dataclass(frozen=True)
class TrainingJob:
    """One shared-weight training run assembled by the phase runner."""

    train_episodes: Sequence[TrainingEpisode]
    val_episodes: Sequence[TrainingEpisode]
    output_dir: Path
    target: TrainingTarget
    mappings: Sequence[ResolvedTrainingMapping]
    mapping_runtime: Sequence[RuntimeMapping]
    input_features: Sequence[TrainingFeature]
    network: TrainingNetwork
    fit: TrainingFit
    sample_dt_s: float
    hashes: RuntimeHashes
    dynamic_bias_provenance: Mapping[str, Any] | None = None
    stats_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        train = _typed_tuple(self.train_episodes, TrainingEpisode, label="train_episodes")
        validation = _typed_tuple(self.val_episodes, TrainingEpisode, label="val_episodes", allow_empty=True)
        mappings = _typed_tuple(self.mappings, ResolvedTrainingMapping, label="mappings")
        runtime = _typed_tuple(self.mapping_runtime, RuntimeMapping, label="mapping_runtime")
        features = _typed_tuple(self.input_features, TrainingFeature, label="input_features")
        if len(runtime) != len(mappings):
            raise ValueError("mapping_runtime must contain one entry per resolved mapping")

        identities = tuple(_mapping_identity(mapping) for mapping in mappings)
        if len(set(identities)) != len(identities):
            raise ValueError("mappings must not contain duplicates")
        first_width = (len(mappings[0].input_joints), len(mappings[0].output_joints))
        for index, (mapping, constants) in enumerate(zip(mappings, runtime, strict=True)):
            identity = _mapping_identity(mapping)
            runtime_identity = (constants.input_joints, constants.output_joints)
            if runtime_identity != identity:
                raise ValueError(f"mapping_runtime[{index}] axes must exactly match mappings[{index}]")
            width = (len(mapping.input_joints), len(mapping.output_joints))
            if width != first_width:
                raise ValueError(
                    f"mapping {index} has dimensions {width[0]}->{width[1]}; "
                    f"expected {first_width[0]}->{first_width[1]}"
                )

        configured = set(identities)
        for split, episodes in (("train", train), ("validation", validation)):
            for index, episode in enumerate(episodes):
                if _mapping_identity(episode.mapping) not in configured:
                    raise ValueError(f"{split} episode {index} uses an unconfigured mapping")
        covered = {_mapping_identity(episode.mapping) for episode in train}
        if covered != configured:
            raise ValueError("training episodes must cover every configured mapping")

        feature_names = tuple(feature.name for feature in features)
        if len(set(feature_names)) != len(feature_names):
            raise ValueError("input_features must not contain duplicate names")
        if self.target not in {"torque", "torque_residual"}:
            raise ValueError(f"unsupported training target {self.target!r}")
        sample_dt_s = float(self.sample_dt_s)
        if not math.isfinite(sample_dt_s) or sample_dt_s <= 0.0:
            raise ValueError("sample_dt_s must be finite and positive")
        if not isinstance(self.network, TrainingNetwork):
            raise TypeError("network must be a TrainingNetwork")
        if not isinstance(self.fit, TrainingFit):
            raise TypeError("fit must be a TrainingFit")
        if not isinstance(self.hashes, RuntimeHashes):
            raise TypeError("hashes must be RuntimeHashes")

        provenance = self.dynamic_bias_provenance
        needs_bias = "dynamic_bias" in feature_names
        if needs_bias and not isinstance(provenance, Mapping):
            raise ValueError("dynamic_bias training requires derivation provenance")
        if not needs_bias and provenance is not None:
            raise ValueError("dynamic_bias provenance requires a dynamic_bias input feature")

        metadata = dict(self.stats_metadata)
        reserved = set(RuntimeGRUMetadata.model_fields).union({"metrics", "training"})
        collisions = sorted(reserved.intersection(metadata))
        if collisions:
            raise ValueError("stats_metadata cannot override reserved fields: " + ", ".join(collisions))

        object.__setattr__(self, "train_episodes", train)
        object.__setattr__(self, "val_episodes", validation)
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "mappings", mappings)
        object.__setattr__(self, "mapping_runtime", runtime)
        object.__setattr__(self, "input_features", features)
        object.__setattr__(self, "sample_dt_s", sample_dt_s)
        object.__setattr__(
            self,
            "dynamic_bias_provenance",
            None if provenance is None else MappingProxyType(dict(provenance)),
        )
        object.__setattr__(self, "stats_metadata", MappingProxyType(metadata))


@dataclass(frozen=True)
class TrainingArtifacts:
    """Fixed files and validated runtime contract produced by a training run."""

    checkpoint_path: Path
    stats_path: Path
    network_path: Path
    metrics: dict[str, Any]
    runtime_metadata: RuntimeGRUMetadata


def _typed_tuple(values: Any, expected_type: type, *, label: str, allow_empty: bool = False) -> tuple[Any, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{label} must be an ordered sequence")
    result = tuple(values)
    if not result and not allow_empty:
        raise ValueError(f"{label} must not be empty")
    if any(not isinstance(item, expected_type) for item in result):
        raise TypeError(f"{label} must contain {expected_type.__name__} values")
    return result


def _mapping_identity(mapping: ResolvedTrainingMapping) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return mapping.input_joints, mapping.output_joints


def _import_torch():
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "PyTorch is required for train-actuator execution. Run in the full container "
            "or install torch in the active environment."
        ) from exc
    return torch


def _seed_everything(seed: int, torch: Any | None = None) -> None:
    """Seed every stochastic library used by this module."""

    random.seed(seed)
    np.random.seed(seed)
    if torch is None:
        return
    torch.manual_seed(seed)
    if hasattr(torch, "cuda") and hasattr(torch.cuda, "manual_seed_all"):
        torch.cuda.manual_seed_all(seed)


def _runtime_for_episode(job: TrainingJob, episode: TrainingEpisode) -> RuntimeMapping:
    identity = _mapping_identity(episode.mapping)
    for mapping, runtime in zip(job.mappings, job.mapping_runtime, strict=True):
        if _mapping_identity(mapping) == identity:
            return runtime
    raise ValueError("episode uses an unconfigured mapping")


def _value_vector(value: StatValue, width: int, *, label: str) -> np.ndarray:
    raw = value if isinstance(value, tuple) else (value,)
    if len(raw) != width:
        raise ValueError(f"{label} must contain {width} value(s)")
    result = np.asarray(raw, dtype=np.float32)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must contain finite values")
    return result


def _physics_signal(blocks: Mapping[str, FeatureBlock], name: str, *, source: str) -> np.ndarray:
    block = blocks.get(name)
    if block is None:
        raise ValueError(f"{source} is missing {name!r} required by the physics baseline")
    if block.domain != "input":
        raise ValueError(f"{source} physics feature {name!r} must use domain 'input'")
    try:
        component = block.components.index("value")
    except ValueError as exc:
        raise ValueError(f"{source} physics feature {name!r} must contain a 'value' component") from exc
    return block.values[:, :, component]


def _derive_target_and_features(
    job: TrainingJob,
    episode: TrainingEpisode,
    *,
    source: str,
) -> tuple[dict[str, FeatureBlock], np.ndarray]:
    """Apply one mapping's frozen physics and autoregressive semantics."""

    runtime = _runtime_for_episode(job, episode)
    blocks = dict(episode.feature_blocks)
    target = np.array(episode.measured_torque, dtype=np.float32, copy=True)
    requested = {feature.name for feature in job.input_features}
    needs_pd = job.target == "torque_residual" or "solver_pd" in requested

    solver_pd: np.ndarray | None = None
    velocity: np.ndarray | None = None
    if needs_pd:
        baseline = runtime.pd_baseline
        if baseline is None:
            raise ValueError(f"{source} mapping requires a PD baseline")
        position_error = _physics_signal(blocks, "position_error", source=source)
        velocity = _physics_signal(blocks, "velocity", source=source)
        width = len(episode.mapping.input_joints)
        kp = _value_vector(baseline.kp, width, label=f"{source} pd_baseline.kp")
        kd = _value_vector(baseline.kd, width, label=f"{source} pd_baseline.kd")
        solver_pd = (kp * position_error - kd * velocity).astype(np.float32)
        if "solver_pd" in requested:
            feature = next(item for item in job.input_features if item.name == "solver_pd")
            if feature.components != ("value",):
                raise ValueError("solver_pd provides exactly the 'value' component")
            blocks["solver_pd"] = FeatureBlock(
                solver_pd[:, :, np.newaxis],
                "input",
                ("value",),
            )

    if job.target == "torque_residual":
        friction = runtime.friction_baseline
        if friction is None:
            raise ValueError(f"{source} residual target requires a friction baseline")
        assert solver_pd is not None and velocity is not None
        output_indices = [episode.mapping.input_joints.index(name) for name in episode.mapping.output_joints]
        output_pd = solver_pd[:, output_indices]
        output_velocity = velocity[:, output_indices]
        width = len(episode.mapping.output_joints)
        dry = _value_vector(friction.dry, width, label=f"{source} friction_baseline.dry")
        viscous = _value_vector(friction.viscous, width, label=f"{source} friction_baseline.viscous")
        friction_torque = dry * np.sign(output_velocity) + viscous * output_velocity
        target = (target - output_pd - friction_torque).astype(np.float32)

    if "previous_torque" in requested:
        previous = np.zeros_like(target, dtype=np.float32)
        previous[1:] = target[:-1]
        blocks["previous_torque"] = FeatureBlock(
            previous[:, :, np.newaxis],
            "output",
            ("value",),
        )
    if not np.all(np.isfinite(target)):
        raise ValueError(f"{source} target contains non-finite values")
    return blocks, target


def _flatten_feature_blocks(
    job: TrainingJob,
    episode: TrainingEpisode,
    blocks: Mapping[str, FeatureBlock],
    *,
    source: str,
) -> np.ndarray:
    """Flatten in feature, joint, component order."""

    rows = len(episode.timestamps)
    flattened: list[np.ndarray] = []
    for feature in job.input_features:
        block = blocks.get(feature.name)
        if block is None:
            raise ValueError(f"{source} is missing required feature {feature.name!r}")
        if block.domain != feature.domain:
            raise ValueError(
                f"{source} feature {feature.name!r} uses domain {block.domain!r}, "
                f"expected {feature.domain!r}"
            )
        if block.components != feature.components:
            raise ValueError(
                f"{source} feature {feature.name!r} components {block.components!r} "
                f"do not match {feature.components!r}"
            )
        axis = len(episode.mapping.output_joints if feature.domain == "output" else episode.mapping.input_joints)
        expected = (rows, axis, len(feature.components))
        if block.values.shape != expected:
            raise ValueError(f"{source} feature {feature.name!r} has shape {block.values.shape}, expected {expected}")
        flattened.append(block.values.reshape(rows, -1))
    features = np.concatenate(flattened, axis=1).astype(np.float32, copy=False)
    if not np.all(np.isfinite(features)):
        raise ValueError(f"{source} features contain non-finite values")
    return features


def _prepare_episode(
    job: TrainingJob,
    episode: TrainingEpisode,
    *,
    source_name: str = "training episode",
) -> tuple[np.ndarray, np.ndarray]:
    """Convert one physical episode into one independent recurrent sequence."""

    blocks, target = _derive_target_and_features(job, episode, source=source_name)
    return _flatten_feature_blocks(job, episode, blocks, source=source_name), target


def _load_sequences(
    job: TrainingJob,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[tuple[np.ndarray, np.ndarray]]]:
    train = [
        _prepare_episode(job, episode, source_name=f"training episode {index}")
        for index, episode in enumerate(job.train_episodes)
    ]
    validation = [
        _prepare_episode(job, episode, source_name=f"validation episode {index}")
        for index, episode in enumerate(job.val_episodes)
    ]
    return train, validation


def _feature_widths(job: TrainingJob) -> tuple[int, ...]:
    mapping = job.mappings[0]
    return tuple(
        len(mapping.output_joints if feature.domain == "output" else mapping.input_joints)
        * len(feature.components)
        for feature in job.input_features
    )


def _validate_sequence_shapes(
    job: TrainingJob,
    sequences: Sequence[tuple[np.ndarray, np.ndarray]],
) -> tuple[int, int]:
    input_size = sum(_feature_widths(job))
    output_size = len(job.mappings[0].output_joints)
    for index, (features, targets) in enumerate(sequences):
        if features.ndim != 2 or features.shape[1] != input_size:
            raise ValueError(f"sequence {index} input shape must be [time, {input_size}]")
        if targets.ndim != 2 or targets.shape[1] != output_size:
            raise ValueError(f"sequence {index} target shape must be [time, {output_size}]")
        if len(features) != len(targets) or not len(features):
            raise ValueError(f"sequence {index} inputs and targets must have the same non-zero row count")
        if not np.all(np.isfinite(features)) or not np.all(np.isfinite(targets)):
            raise ValueError(f"sequence {index} contains non-finite values")
    return input_size, output_size


def _stat_value(values: np.ndarray) -> StatValue:
    result = tuple(float(value) for value in values)
    return result[0] if len(result) == 1 else result


def _normalization_stats(
    job: TrainingJob,
    sequences: Sequence[tuple[np.ndarray, np.ndarray]],
) -> RuntimeNormalization:
    """Pool finite normalization statistics across training sequences only."""

    if not sequences:
        raise ValueError("normalization requires at least one training sequence")
    _validate_sequence_shapes(job, sequences)
    inputs = np.concatenate([item[0] for item in sequences], axis=0).astype(np.float64)
    targets = np.concatenate([item[1] for item in sequences], axis=0).astype(np.float64)
    input_mean = inputs.mean(axis=0)
    input_std = inputs.std(axis=0)
    target_mean = targets.mean(axis=0)
    target_std = targets.std(axis=0)
    if not np.all(np.isfinite(input_mean)) or not np.all(np.isfinite(target_mean)):
        raise ValueError("normalization means must be finite")
    if not np.all(np.isfinite(input_std)) or not np.all(np.isfinite(target_std)):
        raise ValueError("normalization standard deviations must be finite")
    if np.any(input_std == 0.0) or np.any(target_std == 0.0):
        raise ValueError("normalization standard deviations must be non-zero")

    offsets = np.cumsum((0, *_feature_widths(job)))
    input_means = {
        feature.name: _stat_value(input_mean[offsets[index] : offsets[index + 1]])
        for index, feature in enumerate(job.input_features)
    }
    input_stds = {
        feature.name: _stat_value(input_std[offsets[index] : offsets[index + 1]])
        for index, feature in enumerate(job.input_features)
    }
    return RuntimeNormalization(
        inputs=NormalizationValues(mean=input_means, std=input_stds),
        targets=NormalizationValues(
            mean={job.target: _stat_value(target_mean)},
            std={job.target: _stat_value(target_std)},
        ),
    )


def _expand_stat(value: StatValue, width: int, *, label: str) -> np.ndarray:
    return _value_vector(value, width, label=label)


def _normalization_vectors(
    job: TrainingJob,
    normalization: RuntimeNormalization,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    means: list[np.ndarray] = []
    standard_deviations: list[np.ndarray] = []
    for feature, width in zip(job.input_features, _feature_widths(job), strict=True):
        means.append(
            _expand_stat(
                normalization.inputs.mean[feature.name],
                width,
                label=f"normalization.inputs.mean.{feature.name}",
            )
        )
        standard_deviations.append(
            _expand_stat(
                normalization.inputs.std[feature.name],
                width,
                label=f"normalization.inputs.std.{feature.name}",
            )
        )
    output_width = len(job.mappings[0].output_joints)
    target_mean = _expand_stat(
        normalization.targets.mean[job.target],
        output_width,
        label=f"normalization.targets.mean.{job.target}",
    )
    target_std = _expand_stat(
        normalization.targets.std[job.target],
        output_width,
        label=f"normalization.targets.std.{job.target}",
    )
    input_mean = np.concatenate(means)
    input_std = np.concatenate(standard_deviations)
    if np.any(input_std <= 0.0) or np.any(target_std <= 0.0):
        raise ValueError("normalization standard deviations must be positive")
    return input_mean, input_std, target_mean, target_std


def _normalize_sequences(
    job: TrainingJob,
    sequences: Sequence[tuple[np.ndarray, np.ndarray]],
    normalization: RuntimeNormalization,
) -> list[tuple[np.ndarray, np.ndarray]]:
    input_mean, input_std, target_mean, target_std = _normalization_vectors(job, normalization)
    return [
        (
            ((features - input_mean) / input_std).astype(np.float32),
            ((targets - target_mean) / target_std).astype(np.float32),
        )
        for features, targets in sequences
    ]


def _run_epoch(
    torch: Any,
    model: Any,
    optimizer: Any,
    loss_fn: Any,
    sequences: Sequence[tuple[np.ndarray, np.ndarray]],
    hparams: TrainingHyperParameters,
    *,
    train: bool,
    device: Any,
    noise_std: float,
) -> float:
    """Reset state per sequence and retain it across detached TBPTT chunks."""

    model.train(train)
    losses: list[float] = []
    grad_context = torch.enable_grad() if train else torch.no_grad()
    with grad_context:
        for features, targets in sequences:
            hidden = None
            x = torch.as_tensor(features, dtype=torch.float32, device=device)
            y = torch.as_tensor(targets, dtype=torch.float32, device=device)
            for start in range(0, x.shape[0], hparams.chunk_length):
                end = start + hparams.chunk_length
                xb = x[start:end].unsqueeze(0)
                yb = y[start:end].unsqueeze(0)
                if train and noise_std > 0.0:
                    xb = xb + torch.randn_like(xb) * noise_std
                prediction, hidden = model(xb, hidden)
                loss = loss_fn(prediction, yb)
                if train:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    hidden = hidden.detach()
                losses.append(float(loss.detach().cpu()))
    return float(sum(losses) / max(1, len(losses)))


def _default_hparams(job: TrainingJob) -> TrainingHyperParameters:
    return TrainingHyperParameters(
        hidden_size=job.network.hidden_size,
        num_layers=job.network.num_layers,
        dropout=job.network.dropout,
    )


def _runtime_metadata(
    job: TrainingJob,
    normalization: RuntimeNormalization,
    hparams: TrainingHyperParameters,
) -> RuntimeGRUMetadata:
    """Build and validate the complete standalone schema-v3 contract."""

    widths = _feature_widths(job)
    feature_names = {feature.name for feature in job.input_features}
    previous = None
    if "previous_torque" in feature_names:
        previous = PreviousTorqueDerivation(
            value_space="residual_torque" if job.target == "torque_residual" else "physical_torque"
        )
    dynamic_bias = None
    if "dynamic_bias" in feature_names:
        if job.dynamic_bias_provenance is None:
            raise ValueError("dynamic_bias training requires derivation provenance")
        dynamic_bias = DynamicBiasDerivation(
            source=_DYNAMIC_BIAS_SOURCE,
            provenance=dict(job.dynamic_bias_provenance),
        )
    return RuntimeGRUMetadata(
        input_size=sum(widths),
        output_size=len(job.mappings[0].output_joints),
        recurrent=GRURecurrentDimensions(
            hidden_size=hparams.hidden_size,
            num_layers=hparams.num_layers,
            dropout=hparams.dropout,
        ),
        input_features=tuple(
            RuntimeFeature(
                name=feature.name,
                domain=feature.domain,
                components=feature.components,
                width=width,
            )
            for feature, width in zip(job.input_features, widths, strict=True)
        ),
        mappings=tuple(job.mapping_runtime),
        target_columns=(job.target,),
        sample_dt_s=job.sample_dt_s,
        normalization=normalization,
        previous_torque_derivation=previous,
        dynamic_bias_derivation=dynamic_bias,
        hashes=job.hashes,
    )


def _fit_once(
    job: TrainingJob,
    hparams: TrainingHyperParameters,
    *,
    train_sequences: Sequence[tuple[np.ndarray, np.ndarray]],
    val_sequences: Sequence[tuple[np.ndarray, np.ndarray]],
    epochs: int,
    save_artifacts: bool,
    normalization: RuntimeNormalization,
) -> tuple[float, TrainingArtifacts | None]:
    torch = _import_torch()
    _seed_everything(job.fit.seed, torch)
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    input_size, output_size = _validate_sequence_shapes(job, (*train_sequences, *val_sequences))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_runtime_gru(
        input_size=input_size,
        output_size=output_size,
        hidden_size=hparams.hidden_size,
        num_layers=hparams.num_layers,
        dropout=hparams.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hparams.learning_rate,
        weight_decay=hparams.weight_decay,
    )
    loss_fn = torch.nn.HuberLoss(delta=hparams.huber_delta)
    best_val = math.inf
    best_state = None
    history: list[dict[str, float | int]] = []
    evaluation_sequences = val_sequences or train_sequences
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
            noise_std=job.fit.noise_std,
        )
        val_loss = _run_epoch(
            torch,
            model,
            optimizer,
            loss_fn,
            evaluation_sequences,
            hparams,
            train=False,
            device=device,
            noise_std=0.0,
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if not save_artifacts:
        return best_val, None

    if best_state is not None:
        model.load_state_dict(best_state)
    metadata = _runtime_metadata(job, normalization, hparams)
    metrics: dict[str, Any] = {
        "best_val_loss": best_val,
        "history": history,
        "device": str(device),
        "hyperparameters": hparams.__dict__,
    }
    checkpoint_path = job.output_dir / "checkpoints" / "best.pt"
    stats_path = job.output_dir / "stats.json"
    network_path = job.output_dir / "network.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_size": input_size,
        "output_size": output_size,
        "runtime_metadata": metadata.model_dump(mode="json", exclude_none=True),
        "hyperparameters": hparams.__dict__,
        "metrics": metrics,
    }
    torch.save(checkpoint, checkpoint_path)
    stats = {
        **metadata.model_dump(mode="json", exclude_none=True),
        **dict(job.stats_metadata),
        "training": {
            "train_sequence_count": len(job.train_episodes),
            "validation_sequence_count": len(job.val_episodes),
        },
        "metrics": metrics,
    }
    write_stats_json(stats_path, stats)
    export_gru_checkpoint(checkpoint_path, network_path)
    return best_val, TrainingArtifacts(
        checkpoint_path=checkpoint_path,
        stats_path=stats_path,
        network_path=network_path,
        metrics=metrics,
        runtime_metadata=metadata,
    )


def _select_hparams(
    job: TrainingJob,
    train_sequences: Sequence[tuple[np.ndarray, np.ndarray]],
    val_sequences: Sequence[tuple[np.ndarray, np.ndarray]],
    normalization: RuntimeNormalization,
) -> TrainingHyperParameters:
    if job.fit.skip_optuna:
        return _default_hparams(job)
    try:
        import optuna
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("optuna is required unless skip_optuna is enabled") from exc

    sampler = optuna.samplers.TPESampler(seed=job.fit.seed)

    def objective(trial: Any) -> float:
        hparams = TrainingHyperParameters(
            hidden_size=job.network.hidden_size,
            num_layers=job.network.num_layers,
            dropout=job.network.dropout,
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
            epochs=job.fit.trial_epochs,
            save_artifacts=False,
            normalization=normalization,
        )
        return score

    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=job.fit.trials)
    return TrainingHyperParameters(
        hidden_size=job.network.hidden_size,
        num_layers=job.network.num_layers,
        dropout=job.network.dropout,
        **study.best_params,
    )


def train_stateful_gru(job: TrainingJob) -> TrainingArtifacts:
    """Fit one GRU and write the fixed immutable-run staging artifacts."""

    _seed_everything(job.fit.seed)
    train_raw, validation_raw = _load_sequences(job)
    normalization = _normalization_stats(job, train_raw)
    train_sequences = _normalize_sequences(job, train_raw, normalization)
    validation_sequences = _normalize_sequences(job, validation_raw, normalization)
    hparams = _select_hparams(job, train_sequences, validation_sequences, normalization)
    _, artifacts = _fit_once(
        job,
        hparams,
        train_sequences=train_sequences,
        val_sequences=validation_sequences,
        epochs=job.fit.final_epochs,
        save_artifacts=True,
        normalization=normalization,
    )
    assert artifacts is not None
    return artifacts
