from __future__ import annotations

import json
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from anchor_bench.core.config_model import parse_gru_columns
from anchor_bench.data.motion import MotionData
from anchor_bench.data.sage import discover_sage_motions


DEFAULT_INPUT_COLUMNS = ("position", "position_error", "velocity")
DEFAULT_TARGET_COLUMN = "torque"
MODEL_CHOICES = {"gru", "hybrid_residual"}
EPSILON_STD = 1.0e-8


@dataclass(frozen=True)
class TrainingColumns:
    input_columns: list[str]
    target_columns: list[str]
    source_target_columns: list[str]


@dataclass(frozen=True)
class TrainingSource:
    path: Path
    joint: str | None = None


@dataclass(frozen=True)
class TrainingMapping:
    """One application of shared weights to ordered input and output joints."""

    input_joints: tuple[str, ...]
    output_joints: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate the recurrent tensor axes for this mapping."""

        inputs = tuple(str(name) for name in self.input_joints)
        outputs = tuple(str(name) for name in self.output_joints)
        if not inputs or not outputs:
            raise ValueError("training mappings require at least one input and output joint")
        if any(not name for name in (*inputs, *outputs)):
            raise ValueError("training mapping joint names must not be empty")
        if len(set(inputs)) != len(inputs) or len(set(outputs)) != len(outputs):
            raise ValueError("training mapping joint names must not contain duplicates")
        missing = [name for name in outputs if name not in inputs]
        if missing:
            raise ValueError("output_joints must be a subset of input_joints: " + ", ".join(missing))
        object.__setattr__(self, "input_joints", inputs)
        object.__setattr__(self, "output_joints", outputs)


@dataclass(frozen=True)
class TrainingEpisode:
    """Time-aligned signals for one source and one joint mapping."""

    position: np.ndarray
    position_error: np.ndarray
    velocity: np.ndarray | None
    torque: np.ndarray | None


def discover_training_sources(root: str | Path) -> list[TrainingSource]:
    """Discover recorded SAGE episodes without expanding joint mappings."""

    path = Path(root).expanduser()
    motions = discover_sage_motions(path)
    if not motions:
        raise ValueError(f"no SAGE motion folders found under {path}")
    return [TrainingSource(motion) for motion in motions]


def expand_joint_sources(
    sources: Sequence[TrainingSource],
    joints: Sequence[str],
) -> list[TrainingSource]:
    """Retain the legacy scalar source expansion used by older callers."""

    return [
        TrainingSource(source.path, joint)
        for source in sources
        for joint in joints
    ]


def split_training_sources(
    sources: Sequence[TrainingSource],
    *,
    seed: int,
    val_fraction: float = 0.2,
) -> tuple[list[TrainingSource], list[TrainingSource]]:
    """Split by recorded episode path so mapped sequences cannot leak."""

    if not sources:
        raise ValueError("cannot split an empty training source list")
    groups: dict[Path, list[TrainingSource]] = {}
    for source in sources:
        groups.setdefault(source.path, []).append(source)
    paths = list(groups)
    random.Random(seed).shuffle(paths)
    if len(paths) == 1:
        return list(groups[paths[0]]), []
    val_count = max(1, int(round(len(paths) * val_fraction)))
    val_count = min(val_count, len(paths) - 1)
    val_paths = set(paths[:val_count])

    def source_key(source: TrainingSource) -> tuple[str, str]:
        return str(source.path), source.joint or ""

    train = [source for path in paths if path not in val_paths for source in groups[path]]
    val = [source for path in paths if path in val_paths for source in groups[path]]
    return sorted(train, key=source_key), sorted(val, key=source_key)


@lru_cache(maxsize=32)
def load_training_motion(path: Path) -> MotionData:
    """Load and cache one full-width SAGE motion."""

    from anchor_bench.data.sage import load_sage_motion

    return load_sage_motion(path)


def read_training_header(source: TrainingSource, joint: str) -> list[str]:
    """Return scalar training columns available for one recorded joint."""

    return list(load_training_motion(source.path).training_columns(source.joint or joint))


def load_training_episode(source: TrainingSource, mapping: TrainingMapping) -> TrainingEpisode:
    """Align commands, velocity, and torque to recorded state timestamps."""

    motion = load_training_motion(source.path)
    input_indices = motion.joints.indices(mapping.input_joints)
    output_indices = motion.joints.indices(mapping.output_joints)
    state_time = np.asarray(motion.position.time, dtype=np.float64)
    if not len(state_time):
        raise ValueError(f"training source has no rows: {source.path}")
    position = np.asarray(motion.position.values[:, input_indices], dtype=np.float32)
    command = np.asarray(
        motion.command_position.interpolate(state_time).values[:, input_indices],
        dtype=np.float32,
    )
    velocity = (
        np.asarray(motion.velocity.interpolate(state_time).values[:, input_indices], dtype=np.float32)
        if motion.velocity is not None
        else None
    )
    torque = (
        np.asarray(motion.effort.interpolate(state_time).values[:, output_indices], dtype=np.float32)
        if motion.effort is not None
        else None
    )
    return TrainingEpisode(
        position=position,
        position_error=command - position,
        velocity=velocity,
        torque=torque,
    )


def load_training_table(
    source: TrainingSource,
    *,
    joint: str,
    columns: Sequence[str],
) -> dict[str, np.ndarray]:
    """Load selected scalar columns for legacy callers."""

    available = load_training_motion(source.path).training_columns(source.joint or joint)
    missing = [column for column in columns if column not in available]
    if missing:
        raise ValueError(f"{source.path} missing required training signals: {', '.join(missing)}")
    return {
        column: np.asarray(available[column], dtype=np.float32)
        for column in columns
    }


def resolve_training_columns(
    header: Iterable[str],
    *,
    model: str,
    input_columns: Sequence[str] | None = None,
    target_column: str | None = None,
    derived_columns: Sequence[str] | None = None,
) -> TrainingColumns:
    """Resolve ordered feature and target semantics for GRU training."""

    if model not in MODEL_CHOICES:
        raise ValueError(f"model {model!r} must be one of {sorted(MODEL_CHOICES)}")
    header_set = set(header)
    inputs = parse_gru_columns(input_columns or DEFAULT_INPUT_COLUMNS)
    source_target = [target_column or DEFAULT_TARGET_COLUMN]
    derived = {"dynamic_bias", "previous_torque", *(derived_columns or ())}
    missing = [
        column
        for column in [*inputs, *source_target]
        if column not in header_set and column not in derived
    ]
    if missing:
        raise ValueError(f"missing required training signals: {', '.join(missing)}")
    targets = ["torque_residual"] if model == "hybrid_residual" else source_target
    return TrainingColumns(
        input_columns=inputs,
        target_columns=targets,
        source_target_columns=source_target,
    )


def write_stats_json(path: str | Path, stats: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output
