from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


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


def discover_csv_files(root: str | Path) -> list[Path]:
    path = Path(root).expanduser()
    if path.is_file():
        csvs = [path] if path.suffix.lower() == ".csv" else []
    else:
        csvs = sorted(item for item in path.rglob("*.csv") if item.is_file()) if path.exists() else []
    if not csvs:
        raise ValueError(f"no CSV files found under {path}")
    return csvs


def read_csv_header(path: str | Path) -> list[str]:
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        try:
            return [str(column) for column in next(reader)]
        except StopIteration as exc:
            raise ValueError(f"CSV file is empty: {path}") from exc


def split_csv_files(
    csv_files: Sequence[str | Path],
    *,
    seed: int,
    val_fraction: float = 0.2,
) -> tuple[list[Path], list[Path]]:
    files = [Path(path) for path in csv_files]
    if not files:
        raise ValueError("cannot split an empty CSV list")
    shuffled = list(files)
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) == 1:
        return shuffled, []
    val_count = max(1, int(round(len(shuffled) * val_fraction)))
    val_count = min(val_count, len(shuffled) - 1)
    val = sorted(shuffled[:val_count])
    train = sorted(shuffled[val_count:])
    return train, val


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
    missing = [column for column in [*inputs, *source_target] if column not in header_set and column not in derived]
    if missing:
        raise ValueError(f"missing required CSV columns: {', '.join(missing)}")
    targets = ["torque_residual"] if model == "hybrid_residual" else source_target
    return TrainingColumns(input_columns=inputs, target_columns=targets, source_target_columns=source_target)


def _iter_numeric_rows(csv_files: Sequence[Path], columns: Sequence[str]) -> Iterable[dict[str, float]]:
    for path in csv_files:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                raise ValueError(f"CSV file is empty: {path}")
            missing = [column for column in columns if column not in reader.fieldnames]
            if missing:
                raise ValueError(f"{path} missing required CSV columns: {', '.join(missing)}")
            for row in reader:
                yield {column: float(row[column]) for column in columns}


def _stats(values: dict[str, list[float]]) -> tuple[dict[str, float], dict[str, float]]:
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for column, column_values in values.items():
        if not column_values:
            raise ValueError(f"cannot compute normalization stats for empty column {column}")
        mean = sum(column_values) / len(column_values)
        variance = sum((value - mean) ** 2 for value in column_values) / len(column_values)
        means[column] = mean
        stds[column] = max(math.sqrt(variance), EPSILON_STD)
    return means, stds


def compute_normalization_stats(
    csv_files: Sequence[str | Path],
    *,
    input_columns: Sequence[str],
    target_columns: Sequence[str],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    paths = [Path(path) for path in csv_files]
    columns = [*input_columns, *target_columns]
    input_values = {column: [] for column in input_columns}
    target_values = {column: [] for column in target_columns}
    row_count = 0
    for row in _iter_numeric_rows(paths, columns):
        row_count += 1
        for column in input_columns:
            input_values[column].append(row[column])
        for column in target_columns:
            target_values[column].append(row[column])
    if row_count == 0:
        raise ValueError("cannot compute normalization stats from empty CSV rows")
    input_mean, input_std = _stats(input_values)
    target_mean, target_std = _stats(target_values)
    result: dict[str, Any] = {
        "schema_version": 1,
        "input_columns": list(input_columns),
        "target_columns": list(target_columns),
        "normalization": {
            "inputs": {"mean": input_mean, "std": input_std},
            "targets": {"mean": target_mean, "std": target_std},
        },
        "data": {"csv_count": len(paths), "row_count": row_count},
    }
    if metadata:
        result.update(metadata)
    return result


def write_stats_json(path: str | Path, stats: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output
