from __future__ import annotations

import json
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from anchor_bench.data.motion import MotionData
from anchor_bench.data.sage import discover_sage_motions


DEFAULT_INPUT_COLUMNS = ("position", "position_error", "velocity")
DEFAULT_TARGET_COLUMN = "torque"
SUPPORTED_INPUT_COLUMNS = {"position", "position_error", "velocity", "solver_pd"}
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

    def source_key(source: TrainingSource) -> tuple[str, str]:
        return str(source.path), source.joint or ""

    train = [source for path in paths if path not in val_paths for source in groups[path]]
    val = [source for path in paths if path in val_paths for source in groups[path]]
    return sorted(train, key=source_key), sorted(val, key=source_key)


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
