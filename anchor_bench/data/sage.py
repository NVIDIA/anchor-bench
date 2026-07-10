from __future__ import annotations

import ast
import csv
import shutil
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import yaml

from anchor_bench.data.motion import JointLayout, MotionData, MotionSignal


SAGE_FILES = ("control.csv", "state_motor.csv", "joint_list.txt", "metadata.yaml")


def validate_sage_dir(data_dir: str | Path) -> dict[str, Any]:
    root = Path(data_dir)
    missing = [name for name in SAGE_FILES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{root} missing required SAGE files: {', '.join(missing)}")
    metadata = yaml.safe_load((root / "metadata.yaml").read_text(encoding="utf-8")) or {}
    if not isinstance(metadata, dict):
        raise ValueError("metadata.yaml must contain a mapping")
    expected = {
        "time_unit": {"microseconds", "seconds"},
        "position_unit": {"rad"},
        "velocity_unit": {"rad/s"},
        "torque_unit": {"N*m", "Nm"},
    }
    for key, allowed in expected.items():
        if metadata.get(key) not in allowed:
            raise ValueError(f"metadata.yaml {key} must be one of {sorted(allowed)}")
    return metadata



def is_sage_motion(path: str | Path) -> bool:
    root = Path(path)
    return root.is_dir() and all((root / name).is_file() for name in SAGE_FILES)


def discover_sage_motions(path: str | Path) -> list[Path]:
    root = Path(path).expanduser()
    if is_sage_motion(root):
        return [root]
    return sorted(candidate for candidate in root.rglob("*") if is_sage_motion(candidate)) if root.exists() else []


def write_sage_motion(
    folder: str | Path,
    joint_names: Sequence[str],
    control_rows: Iterable[Sequence[Any]],
    state_rows: Iterable[Sequence[Any]],
    metadata: dict[str, Any],
) -> Path:
    root = Path(folder)
    root.mkdir(parents=True, exist_ok=True)
    (root / "joint_list.txt").write_text("\n".join(joint_names) + "\n", encoding="utf-8")
    with (root / "control.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["type", "timestamp", "positions"])
        writer.writerows(control_rows)
    with (root / "state_motor.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["type", "timestamp", "positions", "velocities", "torques"])
        writer.writerows(state_rows)
    (root / "metadata.yaml").write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")
    return root


def copy_sage_motion(source: str | Path, destination: str | Path) -> Path:
    source_root = Path(source)
    validate_sage_dir(source_root)
    destination_root = Path(destination)
    destination_root.mkdir(parents=True, exist_ok=True)
    for name in SAGE_FILES:
        shutil.copy2(source_root / name, destination_root / name)
    return destination_root


def load_joint_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_vector(value: str) -> list[float]:
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, list):
        raise ValueError(f"expected vector list, got {value!r}")
    result = [float(item) for item in parsed]
    if not np.all(np.isfinite(result)):
        raise ValueError("vector contains non-finite values")
    return result


def load_vector_csv(
    path: Path,
    columns: tuple[str, ...],
    *,
    row_type: str | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    timestamps: list[float] = []
    vectors: dict[str, list[list[float]]] = {column: [] for column in columns}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = set(reader.fieldnames or [])
        missing = [column for column in ("timestamp", *columns) if column not in fieldnames]
        if missing:
            raise ValueError(f"{path} missing required columns: {', '.join(missing)}")
        for row in reader:
            if row_type is not None and row.get("type") not in (None, "", row_type):
                continue
            timestamps.append(float(row["timestamp"]))
            for column in columns:
                vectors[column].append(parse_vector(row[column]))
    return np.asarray(timestamps, dtype=float), {
        column: np.asarray(values, dtype=float) for column, values in vectors.items()
    }


def _time_scale(metadata: dict[str, Any]) -> float:
    unit = metadata.get("time_unit")
    if unit == "seconds":
        return 1.0
    if unit == "microseconds":
        return 1e-6
    raise ValueError("metadata.yaml time_unit must be one of ['microseconds', 'seconds']")


def _sorted_signal(
    timestamps: np.ndarray,
    values: np.ndarray,
    *,
    scale: float,
    unit: str,
    name: str,
) -> MotionSignal:
    if values.ndim != 2:
        raise ValueError(f"{name} must be a 2-D vector trace")
    order = np.argsort(timestamps, kind="stable")
    return MotionSignal(
        time=np.asarray(timestamps, dtype=float)[order] * scale,
        values=np.asarray(values, dtype=float)[order],
        unit=unit,
        name=name,
    )


def load_sage_motion(
    data_dir: str | Path,
    *,
    joint_names: Sequence[str] | None = None,
    target_dt: float | None = None,
    require_velocity: bool = False,
    require_effort: bool = False,
) -> MotionData:
    """Load and normalize one SAGE motion into the shared motion model."""

    root = Path(data_dir)
    metadata = validate_sage_dir(root)
    layout = JointLayout(tuple(load_joint_list(root / "joint_list.txt")))
    control_time, control_vectors = load_vector_csv(root / "control.csv", ("positions",), row_type="CONTROL")
    state_path = root / "state_motor.csv"
    with state_path.open(newline="", encoding="utf-8") as stream:
        state_fields = set(csv.DictReader(stream).fieldnames or [])
    state_columns = ["positions"]
    for optional in ("velocities", "torques"):
        if optional in state_fields:
            state_columns.append(optional)
    if require_velocity and "velocities" not in state_columns:
        raise ValueError(f"{state_path} missing required column: velocities")
    if require_effort and "torques" not in state_columns:
        raise ValueError(f"{state_path} missing required column: torques")
    state_time, state_vectors = load_vector_csv(state_path, tuple(state_columns), row_type="STATE_MOTOR")

    scale = _time_scale(metadata)
    normalized_metadata = dict(metadata)
    normalized_metadata["source_time_unit"] = metadata["time_unit"]
    normalized_metadata["time_unit"] = "seconds"
    motion = MotionData(
        joints=layout,
        command_position=_sorted_signal(
            control_time,
            control_vectors["positions"],
            scale=scale,
            unit=str(metadata["position_unit"]),
            name="command_position",
        ),
        position=_sorted_signal(
            state_time,
            state_vectors["positions"],
            scale=scale,
            unit=str(metadata["position_unit"]),
            name="position",
        ),
        velocity=(
            _sorted_signal(
                state_time,
                state_vectors["velocities"],
                scale=scale,
                unit=str(metadata["velocity_unit"]),
                name="velocity",
            )
            if "velocities" in state_vectors
            else None
        ),
        effort=(
            _sorted_signal(
                state_time,
                state_vectors["torques"],
                scale=scale,
                unit=str(metadata["torque_unit"]),
                name="effort",
            )
            if "torques" in state_vectors
            else None
        ),
        metadata=normalized_metadata,
        source=root,
    )
    if joint_names is not None:
        motion = motion.select_joints(joint_names)
    if target_dt is not None:
        motion = motion.resample(float(target_dt))
    return motion
