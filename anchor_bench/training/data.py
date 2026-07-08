from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

import numpy as np

from anchor_bench.data.motion import MotionData
from anchor_bench.data.sage import discover_sage_motions


DEFAULT_INPUT_COLUMNS = ("position", "position_error", "velocity")
DEFAULT_TARGET_COLUMN = "torque"
SUPPORTED_INPUT_COLUMNS = {
    "dynamic_bias",
    "position",
    "position_error",
    "previous_torque",
    "solver_pd",
    "velocity",
}
MODEL_CHOICES = {"gru", "hybrid_residual"}
EPSILON_STD = 1.0e-8

if TYPE_CHECKING:
    from anchor_bench.training.dynamic_bias import DynamicBiasProvider


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
class TrainingEpisode:
    """One ordered, time-aligned actuator-training episode.

    Feature blocks remain two-dimensional so the trainer can derive residual
    features before flattening them in feature-major order. Solver PD and
    previous torque are intentionally assembled by the trainer.
    """

    timestamps: np.ndarray
    input_joints: tuple[str, ...]
    output_joints: tuple[str, ...]
    feature_blocks: Mapping[str, np.ndarray]
    target_torque: np.ndarray
    dynamic_bias_provenance: Mapping[str, Any] | None = field(default=None)

    def __post_init__(self) -> None:
        timestamps = _readonly_array(
            self.timestamps,
            dtype=np.float64,
            ndim=1,
            label="episode timestamps",
        )
        target_torque = _readonly_array(
            self.target_torque,
            dtype=np.float32,
            ndim=2,
            label="episode target torque",
        )
        input_joints = tuple(str(name) for name in self.input_joints)
        output_joints = tuple(str(name) for name in self.output_joints)
        if len(target_torque) != len(timestamps):
            raise ValueError("episode target torque must have one row per timestamp")
        if target_torque.shape[1] != len(output_joints):
            raise ValueError("episode target torque width must match output_joints")

        blocks: dict[str, np.ndarray] = {}
        for name, values in self.feature_blocks.items():
            block = _readonly_array(
                values,
                dtype=np.float32,
                ndim=2,
                label=f"episode feature {name}",
            )
            if len(block) != len(timestamps):
                raise ValueError(f"episode feature {name} must have one row per timestamp")
            blocks[str(name)] = block

        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "input_joints", input_joints)
        object.__setattr__(self, "output_joints", output_joints)
        object.__setattr__(self, "feature_blocks", MappingProxyType(blocks))
        object.__setattr__(self, "target_torque", target_torque)
        if self.dynamic_bias_provenance is not None:
            object.__setattr__(
                self,
                "dynamic_bias_provenance",
                MappingProxyType(dict(self.dynamic_bias_provenance)),
            )


def slice_training_episode_joint(
    episode: TrainingEpisode,
    joint: str,
) -> TrainingEpisode:
    """Return one scalar joint view while retaining episode provenance."""

    try:
        input_index = episode.input_joints.index(joint)
    except ValueError as exc:
        raise ValueError(f"episode does not contain input joint {joint!r}") from exc
    try:
        output_index = episode.output_joints.index(joint)
    except ValueError as exc:
        raise ValueError(f"episode does not contain output joint {joint!r}") from exc

    input_width = len(episode.input_joints)
    feature_blocks: dict[str, np.ndarray] = {}
    for name, block in episode.feature_blocks.items():
        if block.shape[1] != input_width:
            raise ValueError(
                f"episode feature {name!r} width must match input_joints"
            )
        feature_blocks[name] = block[:, input_index : input_index + 1]

    return TrainingEpisode(
        timestamps=episode.timestamps,
        input_joints=(joint,),
        output_joints=(joint,),
        feature_blocks=feature_blocks,
        target_torque=episode.target_torque[:, output_index : output_index + 1],
        dynamic_bias_provenance=episode.dynamic_bias_provenance,
    )

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
    result = np.array(array, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _ordered_unique(values: Sequence[str], *, label: str) -> tuple[str, ...]:
    names = tuple(str(value) for value in values)
    if not names:
        raise ValueError(f"{label} must contain at least one joint")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"duplicate {label}: {', '.join(duplicates)}")
    return names


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
    tolerance = max(1.0e-9, sample_dt_s * 1.0e-6)
    matches = np.isclose(intervals, sample_dt_s, rtol=1.0e-6, atol=tolerance)
    if not np.all(matches):
        first = int(np.flatnonzero(~matches)[0])
        raise ValueError(
            f"{source} state cadence does not match sample_dt_s {sample_dt_s}: "
            f"interval {first}->{first + 1} is {float(intervals[first])}"
        )


def assemble_training_episode(
    motion: MotionData,
    *,
    input_joints: Sequence[str],
    output_joints: Sequence[str] | None = None,
    input_columns: Sequence[str] = DEFAULT_INPUT_COLUMNS,
    sample_dt_s: float,
    dynamic_bias_provider: DynamicBiasProvider | None = None,
) -> TrainingEpisode:
    """Assemble one coupled episode without changing its recorded state grid."""

    inputs = _ordered_unique(input_joints, label="input_joints")
    outputs = _ordered_unique(
        output_joints if output_joints is not None else inputs,
        label="output_joints",
    )
    unknown_outputs = [name for name in outputs if name not in inputs]
    if unknown_outputs:
        raise ValueError(
            "output_joints must be a subset of input_joints; not present: "
            + ", ".join(unknown_outputs)
        )

    requested_columns = tuple(str(column) for column in input_columns)
    duplicate_columns = sorted(
        {column for column in requested_columns if requested_columns.count(column) > 1}
    )
    if duplicate_columns:
        raise ValueError(f"duplicate GRU input columns: {', '.join(duplicate_columns)}")
    unsupported = sorted(set(requested_columns) - SUPPORTED_INPUT_COLUMNS)
    if unsupported:
        raise ValueError(f"unsupported GRU input columns: {', '.join(unsupported)}")

    input_indices = motion.joints.indices(inputs)
    output_indices = motion.joints.indices(outputs)
    state_time = np.asarray(motion.position.time, dtype=float)
    if not len(state_time):
        raise ValueError(f"training source has no state rows: {motion.source}")
    _validate_episode_cadence(state_time, float(sample_dt_s), source=motion.source)

    if motion.effort is None:
        raise ValueError(f"{motion.source} is missing required measured torque")

    command = motion.command_position.interpolate(state_time).values[:, input_indices]
    position = motion.position.values[:, input_indices]
    feature_blocks: dict[str, np.ndarray] = {
        "position": position,
        "position_error": command - position,
    }

    needs_velocity = bool(
        {"velocity", "solver_pd", "dynamic_bias"}.intersection(requested_columns)
    )
    velocity_all: np.ndarray | None = None
    if motion.velocity is not None:
        velocity_all = motion.velocity.interpolate(state_time).values
        feature_blocks["velocity"] = velocity_all[:, input_indices]
    elif needs_velocity:
        raise ValueError(f"{motion.source} is missing required measured velocity")

    dynamic_bias_provenance: Mapping[str, Any] | None = None
    if "dynamic_bias" in requested_columns:
        if dynamic_bias_provider is None:
            raise ValueError("dynamic_bias training requires a DynamicBiasProvider")
        assert velocity_all is not None
        result = dynamic_bias_provider.compute(
            joint_names=motion.joints.names,
            positions=motion.position.values,
            velocities=velocity_all,
            requested_joint_names=inputs,
        )
        expected_shape = (len(state_time), len(inputs))
        dynamic_bias = np.asarray(result.values, dtype=np.float32)
        if dynamic_bias.shape != expected_shape:
            raise ValueError(
                f"dynamic-bias provider returned shape {dynamic_bias.shape}, "
                f"expected {expected_shape}"
            )
        if not np.all(np.isfinite(dynamic_bias)):
            raise ValueError("dynamic-bias provider returned non-finite values")
        feature_blocks["dynamic_bias"] = dynamic_bias
        dynamic_bias_provenance = result.provenance

    target_torque = motion.effort.interpolate(state_time).values[:, output_indices]
    return TrainingEpisode(
        timestamps=state_time,
        input_joints=inputs,
        output_joints=outputs,
        feature_blocks=feature_blocks,
        target_torque=target_torque,
        dynamic_bias_provenance=dynamic_bias_provenance,
    )


def load_training_episode(
    source: TrainingSource,
    *,
    input_joints: Sequence[str],
    output_joints: Sequence[str] | None = None,
    input_columns: Sequence[str] = DEFAULT_INPUT_COLUMNS,
    sample_dt_s: float,
    dynamic_bias_provider: DynamicBiasProvider | None = None,
) -> TrainingEpisode:
    """Load one SAGE motion while preserving its episode boundary."""

    return assemble_training_episode(
        _load_training_motion(source.path),
        input_joints=input_joints,
        output_joints=output_joints,
        input_columns=input_columns,
        sample_dt_s=sample_dt_s,
        dynamic_bias_provider=dynamic_bias_provider,
    )


def discover_training_sources(root: str | Path) -> list[TrainingSource]:
    path = Path(root).expanduser()
    motions = discover_sage_motions(path)
    if not motions:
        raise ValueError(f"no SAGE motion folders found under {path}")
    return [TrainingSource(motion) for motion in motions]


def expand_joint_sources(
    sources: Sequence[TrainingSource],
    joints: Sequence[str],
) -> list[TrainingSource]:
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

    train_paths = sorted(
        (path for path in paths if path not in val_paths),
        key=str,
    )
    val_paths_ordered = sorted(
        (path for path in paths if path in val_paths),
        key=str,
    )
    train = [source for path in train_paths for source in groups[path]]
    val = [source for path in val_paths_ordered for source in groups[path]]
    return train, val


@lru_cache(maxsize=32)
def _load_training_motion(path: Path) -> MotionData:
    from anchor_bench.data.sage import load_sage_motion

    return load_sage_motion(path)


def read_training_header(source: TrainingSource, joint: str) -> list[str]:
    return list(_load_training_motion(source.path).training_columns(source.joint or joint))


def load_training_table(
    source: TrainingSource,
    *,
    joint: str,
    columns: Sequence[str],
) -> dict[str, np.ndarray]:
    available = _load_training_motion(source.path).training_columns(source.joint or joint)
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
    if model not in MODEL_CHOICES:
        raise ValueError(f"model {model!r} must be one of {sorted(MODEL_CHOICES)}")
    header_set = set(header)
    inputs = list(input_columns or DEFAULT_INPUT_COLUMNS)
    duplicates = sorted({column for column in inputs if inputs.count(column) > 1})
    if duplicates:
        raise ValueError(f"duplicate GRU input columns: {', '.join(duplicates)}")
    unsupported = sorted(set(inputs) - SUPPORTED_INPUT_COLUMNS)
    if unsupported:
        raise ValueError(f"unsupported GRU input columns: {', '.join(unsupported)}")
    source_target = [target_column or DEFAULT_TARGET_COLUMN]
    derived = set(derived_columns or ())
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
