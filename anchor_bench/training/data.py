from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from anchor_bench.core.config_model import find_duplicates, parse_gru_columns
from anchor_bench.data.motion import MotionData
from anchor_bench.data.sage import discover_sage_motions
from anchor_bench.training.dynamic_bias import DynamicBiasProvider


DEFAULT_INPUT_COLUMNS = ("position", "position_error", "velocity")
EPSILON_STD = 1.0e-8

FeatureDomain = Literal["input_joints", "output_joints"]
FEATURE_DOMAINS: frozenset[str] = frozenset({"input_joints", "output_joints"})


@dataclass(frozen=True)
class TrainingSource:
    path: Path


@dataclass(frozen=True)
class TrainingMapping:
    """One logical application of a group's shared actuator model."""

    input_joints: tuple[str, ...]
    output_joints: tuple[str, ...]

    def __post_init__(self) -> None:
        inputs = _ordered_unique(self.input_joints, label="input_joints")
        outputs = _ordered_unique(self.output_joints, label="output_joints")
        unknown_outputs = [name for name in outputs if name not in inputs]
        if unknown_outputs:
            raise ValueError(
                "output_joints must be a subset of input_joints; not present: " + ", ".join(unknown_outputs)
            )
        object.__setattr__(self, "input_joints", inputs)
        object.__setattr__(self, "output_joints", outputs)


def training_topology(input_width: int, output_width: int) -> str:
    """Classify a mapping shape as siso, miso, or mimo."""
    if input_width == 1:
        return "siso"
    if output_width == 1:
        return "miso"
    return "mimo"


@dataclass(frozen=True)
class FeatureBlock:
    """One time-aligned feature with explicit axes and channel semantics."""

    values: np.ndarray
    domain: FeatureDomain
    channels: tuple[str, ...]

    def __post_init__(self) -> None:
        values = _readonly_array(
            self.values,
            dtype=np.float32,
            ndim=3,
            label="feature block values",
        )
        domain = self.domain
        if not isinstance(domain, str) or domain not in FEATURE_DOMAINS:
            choices = ", ".join(sorted(FEATURE_DOMAINS))
            raise ValueError(f"feature block domain must be one of: {choices}")

        if isinstance(self.channels, str) or not isinstance(
            self.channels,
            Sequence,
        ):
            raise ValueError("feature block channels must be an ordered sequence")
        channels = tuple(self.channels)
        if not channels:
            raise ValueError("feature block channels must contain at least one channel")
        if any(not isinstance(channel, str) or not channel for channel in channels):
            raise ValueError("feature block channel names must be non-empty strings")
        duplicates = find_duplicates(channels)
        if duplicates:
            raise ValueError("duplicate feature block channels: " + ", ".join(duplicates))
        if values.shape[2] != len(channels):
            raise ValueError(
                f"feature block channel axis must match channels: got {values.shape[2]}, expected {len(channels)}"
            )

        object.__setattr__(self, "values", values)
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "channels", channels)


@dataclass(frozen=True)
class TrainingEpisode:
    """One ordered, time-aligned actuator-training episode.

    Feature blocks retain explicit axis and channel dimensions so the trainer
    can derive residual features before flattening them in feature-major order.
    Solver PD and previous torque are intentionally assembled by the trainer.
    """

    timestamps: np.ndarray
    mapping: TrainingMapping
    feature_blocks: Mapping[str, FeatureBlock]
    target_torque: np.ndarray
    dynamic_bias_provenance: Mapping[str, Any] | None = field(default=None)

    def __post_init__(self) -> None:
        timestamps = _readonly_array(
            self.timestamps,
            dtype=np.float64,
            ndim=1,
            label="episode timestamps",
        )
        if not len(timestamps):
            raise ValueError("episode timestamps must not be empty")
        target_torque = _readonly_array(
            self.target_torque,
            dtype=np.float32,
            ndim=2,
            label="episode target torque",
        )
        if not isinstance(self.mapping, TrainingMapping):
            raise TypeError("episode mapping must be a TrainingMapping")
        input_joints = self.mapping.input_joints
        output_joints = self.mapping.output_joints
        if len(target_torque) != len(timestamps):
            raise ValueError("episode target torque must have one row per timestamp")
        if target_torque.shape[1] != len(output_joints):
            raise ValueError("episode target torque width must match output_joints")

        blocks: dict[str, FeatureBlock] = {}
        for name, block in self.feature_blocks.items():
            if not isinstance(name, str) or not name:
                raise ValueError("episode feature names must be non-empty strings")
            if not isinstance(block, FeatureBlock):
                raise TypeError(f"episode feature {name} must be a FeatureBlock")
            if len(block.values) != len(timestamps):
                raise ValueError(f"episode feature {name} must have one row per timestamp")
            expected_axis_width = {
                "input_joints": len(input_joints),
                "output_joints": len(output_joints),
            }[block.domain]
            if block.values.shape[1] != expected_axis_width:
                raise ValueError(
                    f"episode feature {name} {block.domain} axis has width "
                    f"{block.values.shape[1]}, expected {expected_axis_width}"
                )
            blocks[name] = block

        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "feature_blocks", MappingProxyType(blocks))
        object.__setattr__(self, "target_torque", target_torque)
        if self.dynamic_bias_provenance is not None:
            object.__setattr__(
                self,
                "dynamic_bias_provenance",
                MappingProxyType(dict(self.dynamic_bias_provenance)),
            )

    @property
    def input_joints(self) -> tuple[str, ...]:
        return self.mapping.input_joints

    @property
    def output_joints(self) -> tuple[str, ...]:
        return self.mapping.output_joints


def _readonly_array(
    values: Any,
    *,
    dtype: Any,
    ndim: int,
    label: str,
) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    if array.ndim != ndim:
        raise ValueError(f"{label} must be a {ndim}-D array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} contains non-finite values")
    immutable_buffer = array.tobytes(order="C")
    return np.frombuffer(immutable_buffer, dtype=array.dtype).reshape(array.shape)


def _ordered_unique(values: Sequence[str], *, label: str) -> tuple[str, ...]:
    if isinstance(values, str) or not isinstance(values, Sequence):
        raise ValueError(f"{label} must be an ordered sequence of joint names")
    names = tuple(values)
    if not names:
        raise ValueError(f"{label} must contain at least one joint")
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError(f"{label} must contain non-empty strings")
    duplicates = find_duplicates(names)
    if duplicates:
        raise ValueError(f"duplicate {label}: {', '.join(duplicates)}")
    return names


CADENCE_MEDIAN_REL_TOL = 1.0e-2
CADENCE_INTERVAL_REL_TOL = 0.25


def _validate_episode_cadence(
    timestamps: np.ndarray,
    sample_dt_s: float,
    *,
    source: Path,
) -> None:
    if not np.isfinite(sample_dt_s) or sample_dt_s <= 0.0:
        raise ValueError("sample_dt_s must be a positive finite value")
    if len(timestamps) < 2:
        return
    intervals = np.diff(timestamps)
    median = float(np.median(intervals))
    if not math.isclose(median, sample_dt_s, rel_tol=CADENCE_MEDIAN_REL_TOL):
        raise ValueError(
            f"{source} median state cadence does not match sample_dt_s {sample_dt_s}: "
            f"median interval is {median}; set training.sample_dt_s or --sample-dt "
            "to the recording's actual sample interval"
        )
    mismatched = ~np.isclose(intervals, sample_dt_s, rtol=CADENCE_INTERVAL_REL_TOL, atol=0.0)
    if np.any(mismatched):
        first = int(np.flatnonzero(mismatched)[0])
        raise ValueError(
            f"{source} state cadence has an interval outside "
            f"{CADENCE_INTERVAL_REL_TOL:.0%} of sample_dt_s {sample_dt_s}: "
            f"interval {first}->{first + 1} is {float(intervals[first])}"
        )


@dataclass(frozen=True)
class _SourceArrays:
    """Full-width source signals interpolated onto the recorded state grid."""

    state_time: np.ndarray
    position: np.ndarray
    command: np.ndarray
    velocity: np.ndarray | None
    effort: np.ndarray


def _prepare_source_arrays(motion: MotionData, sample_dt_s: float) -> _SourceArrays:
    """Validate cadence and interpolate every source signal once per source."""

    state_time = np.asarray(motion.position.time, dtype=float)
    if not len(state_time):
        raise ValueError(f"training source has no state rows: {motion.source}")
    _validate_episode_cadence(state_time, float(sample_dt_s), source=motion.source)
    if motion.effort is None:
        raise ValueError(f"{motion.source} is missing required measured torque")
    return _SourceArrays(
        state_time=state_time,
        position=np.asarray(motion.position.values, dtype=float),
        command=motion.command_position.interpolate(state_time).values,
        velocity=None if motion.velocity is None else motion.velocity.interpolate(state_time).values,
        effort=motion.effort.interpolate(state_time).values,
    )


@lru_cache(maxsize=16)
def _prepared_training_source(path: Path, sample_dt_s: float) -> tuple[MotionData, _SourceArrays]:
    motion = _load_training_motion(path)
    return motion, _prepare_source_arrays(motion, sample_dt_s)


def _assemble_from_arrays(
    motion: MotionData,
    arrays: _SourceArrays,
    *,
    mapping: TrainingMapping,
    input_columns: Sequence[str],
    dynamic_bias_provider: DynamicBiasProvider | None,
    cache_key: str | None = None,
) -> TrainingEpisode:
    """Build one episode from prepared source arrays for a single mapping."""

    if not isinstance(mapping, TrainingMapping):
        raise TypeError("mapping must be a TrainingMapping")
    inputs = mapping.input_joints
    outputs = mapping.output_joints

    requested_columns = tuple(parse_gru_columns(list(input_columns)))

    input_indices = motion.joints.indices(inputs)
    output_indices = motion.joints.indices(outputs)
    state_time = arrays.state_time

    command = arrays.command[:, input_indices]
    position = arrays.position[:, input_indices]
    feature_blocks: dict[str, FeatureBlock] = {
        "position": FeatureBlock(
            values=position[:, :, np.newaxis],
            domain="input_joints",
            channels=("value",),
        ),
        "position_error": FeatureBlock(
            values=(command - position)[:, :, np.newaxis],
            domain="input_joints",
            channels=("value",),
        ),
    }

    needs_velocity = bool({"velocity", "solver_pd", "dynamic_bias"}.intersection(requested_columns))
    velocity_all = arrays.velocity
    if velocity_all is not None:
        feature_blocks["velocity"] = FeatureBlock(
            values=velocity_all[:, input_indices, np.newaxis],
            domain="input_joints",
            channels=("value",),
        )
    elif needs_velocity:
        raise ValueError(f"{motion.source} is missing required measured velocity")

    dynamic_bias_provenance: Mapping[str, Any] | None = None
    if "dynamic_bias" in requested_columns:
        if dynamic_bias_provider is None:
            raise ValueError("dynamic_bias training requires a DynamicBiasProvider")
        assert velocity_all is not None
        result = dynamic_bias_provider.compute(
            joint_names=motion.joints.names,
            positions=arrays.position,
            velocities=velocity_all,
            requested_joint_names=inputs,
            cache_key=cache_key,
        )
        expected_shape = (len(state_time), len(inputs))
        dynamic_bias = np.asarray(result.values, dtype=np.float32)
        if dynamic_bias.shape != expected_shape:
            raise ValueError(f"dynamic-bias provider returned shape {dynamic_bias.shape}, expected {expected_shape}")
        feature_blocks["dynamic_bias"] = FeatureBlock(
            values=dynamic_bias[:, :, np.newaxis],
            domain="input_joints",
            channels=("value",),
        )
        dynamic_bias_provenance = result.provenance

    target_torque = arrays.effort[:, output_indices]
    return TrainingEpisode(
        timestamps=state_time,
        mapping=mapping,
        feature_blocks=feature_blocks,
        target_torque=target_torque,
        dynamic_bias_provenance=dynamic_bias_provenance,
    )


def assemble_training_episode(
    motion: MotionData,
    *,
    mapping: TrainingMapping,
    input_columns: Sequence[str] = DEFAULT_INPUT_COLUMNS,
    sample_dt_s: float,
    dynamic_bias_provider: DynamicBiasProvider | None = None,
) -> TrainingEpisode:
    """Assemble one coupled episode without changing its recorded state grid."""

    arrays = _prepare_source_arrays(motion, sample_dt_s)
    return _assemble_from_arrays(
        motion,
        arrays,
        mapping=mapping,
        input_columns=input_columns,
        dynamic_bias_provider=dynamic_bias_provider,
    )


def load_training_episode(
    source: TrainingSource,
    *,
    mapping: TrainingMapping,
    input_columns: Sequence[str] = DEFAULT_INPUT_COLUMNS,
    sample_dt_s: float,
    dynamic_bias_provider: DynamicBiasProvider | None = None,
) -> TrainingEpisode:
    """Load one SAGE motion while preserving its episode boundary."""

    motion, arrays = _prepared_training_source(source.path, float(sample_dt_s))
    return _assemble_from_arrays(
        motion,
        arrays,
        mapping=mapping,
        input_columns=input_columns,
        dynamic_bias_provider=dynamic_bias_provider,
        cache_key=str(source.path),
    )


def discover_training_sources(root: str | Path) -> list[TrainingSource]:
    path = Path(root).expanduser()
    motions = discover_sage_motions(path)
    if not motions:
        raise ValueError(f"no SAGE motion folders found under {path}")
    return [TrainingSource(motion) for motion in motions]


def split_training_sources(
    sources: Sequence[TrainingSource],
    *,
    seed: int,
    val_fraction: float = 0.2,
) -> tuple[list[TrainingSource], list[TrainingSource]]:
    if not sources:
        raise ValueError("cannot split an empty training source list")
    by_path: dict[Path, TrainingSource] = {}
    for source in sources:
        by_path.setdefault(source.path, source)
    paths = list(by_path)
    random.Random(seed).shuffle(paths)
    if len(paths) == 1:
        return [by_path[paths[0]]], []
    val_count = max(1, int(round(len(paths) * val_fraction)))
    val_count = min(val_count, len(paths) - 1)
    val_paths = set(paths[:val_count])
    train = [by_path[path] for path in sorted((p for p in paths if p not in val_paths), key=str)]
    val = [by_path[path] for path in sorted((p for p in paths if p in val_paths), key=str)]
    return train, val


@lru_cache(maxsize=32)
def _load_training_motion(path: Path) -> MotionData:
    from anchor_bench.data.sage import load_sage_motion

    return load_sage_motion(path)


def write_stats_json(path: str | Path, stats: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output
