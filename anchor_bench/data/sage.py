
from __future__ import annotations

import ast
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass(frozen=True)
class SageDataset:
    joint_names: list[str]
    control_timestamps: list[float]
    control_positions: list[list[float]]
    state_timestamps: list[float]
    state_positions: list[list[float]]
    state_velocities: list[list[float]]
    state_torques: list[list[float]]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SageInitialState:
    joint_names: list[str]
    positions: list[float]
    velocities: list[float] | None


def load_joint_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_vector(value: str) -> list[float]:
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, list):
        raise ValueError(f"expected vector list, got {value!r}")
    return [float(item) for item in parsed]


def load_vector_csv(path: Path, columns: tuple[str, ...]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    timestamps: list[float] = []
    vectors: dict[str, list[list[float]]] = {column: [] for column in columns}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = set(reader.fieldnames or [])
        missing = [c for c in ("timestamp", *columns) if c not in fieldnames]
        if missing:
            raise ValueError(f"{path} missing required columns: {', '.join(missing)}")
        for row in reader:
            timestamps.append(float(row["timestamp"]))
            for column in columns:
                vectors[column].append(parse_vector(row[column]))
    return np.asarray(timestamps, dtype=float), {
        column: np.asarray(values, dtype=float) for column, values in vectors.items()
    }


def _select_joints(
    vectors: list[list[float]],
    source_joints: list[str],
    requested_joints: list[str],
    column: str = "vectors",
) -> list[list[float]]:
    indices = []
    for joint in requested_joints:
        if joint not in source_joints:
            raise ValueError(f"joint '{joint}' not found in joint_list.txt")
        indices.append(source_joints.index(joint))
    for row_index, row in enumerate(vectors):
        if len(row) != len(source_joints):
            raise ValueError(
                f"{column} row {row_index} has {len(row)} values but "
                f"joint_list.txt has {len(source_joints)} joints"
            )
    return [[row[idx] for idx in indices] for row in vectors]


def _select_vector(values: list[float], source_joints: list[str], requested_joints: list[str], column: str) -> list[float]:
    if len(values) != len(source_joints):
        raise ValueError(f"{column} has {len(values)} values but joint_list.txt has {len(source_joints)} joints")
    return _select_joints([values], source_joints, requested_joints, column)[0]


def _normalize_timestamps(values: list[float], metadata: dict[str, Any]) -> list[float]:
    unit = metadata.get("time_unit")
    if unit == "seconds":
        scale = 1.0
    elif unit == "microseconds":
        scale = 1e-6
    else:
        raise ValueError("metadata.yaml time_unit must be one of ['microseconds', 'seconds']")
    return [float(value) * scale for value in values]


def _sort_rows(timestamps: list[float], vectors: list[list[float]]) -> tuple[list[float], list[list[float]]]:
    rows = sorted(zip(timestamps, vectors), key=lambda item: item[0])
    return [row[0] for row in rows], [row[1] for row in rows]


def _sort_vector_fields(
    timestamps: list[float],
    fields: dict[str, list[list[float]]],
) -> tuple[list[float], dict[str, list[list[float]]]]:
    for name, values in fields.items():
        if len(values) != len(timestamps):
            raise ValueError(
                f"state field {name!r} has {len(values)} rows but timestamps has {len(timestamps)}"
            )
    order = sorted(range(len(timestamps)), key=timestamps.__getitem__)
    return (
        [timestamps[index] for index in order],
        {name: [values[index] for index in order] for name, values in fields.items()},
    )


def _uniform_grid(start: float, end: float, target_dt: float) -> list[float]:
    if target_dt <= 0.0:
        raise ValueError("target_dt must be positive")
    span = end - start
    step_count = int(math.floor(span / target_dt + 1.0e-9))
    return [
        round(float(start + index * target_dt), 12)
        for index in range(step_count + 1)
    ]


def _interpolate_vectors(timestamps: list[float], positions: list[list[float]], target_times: list[float]) -> list[list[float]]:
    if not target_times:
        return []
    source_t = np.asarray(timestamps, dtype=float)
    values = np.asarray(positions, dtype=float)
    if values.ndim != 2:
        raise ValueError("values must be a 2-D vector trace")
    target_t = np.asarray(target_times, dtype=float)
    columns = [np.interp(target_t, source_t, values[:, idx]) for idx in range(values.shape[1])]
    matrix = np.column_stack(columns) if columns else np.empty((len(target_times), 0), dtype=float)
    return [[float(round(value, 12)) for value in row] for row in matrix.tolist()]


def _hold_vectors(
    timestamps: list[float],
    values: list[list[float]],
    target_times: list[float],
) -> list[list[float]]:
    """Sample a commanded vector using previous-sample (zero-order) hold."""
    if not target_times:
        return []
    source_t = np.asarray(timestamps, dtype=float)
    source_values = np.asarray(values, dtype=float)
    if source_values.ndim != 2:
        raise ValueError("values must be a 2-D vector trace")
    indices = np.searchsorted(source_t, np.asarray(target_times, dtype=float), side="right") - 1
    indices = np.clip(indices, 0, len(source_t) - 1)
    return [[float(value) for value in row] for row in source_values[indices].tolist()]


def validate_sage_dir(data_dir: str | Path) -> dict[str, Any]:
    root = Path(data_dir)
    required = ["control.csv", "state_motor.csv", "joint_list.txt", "metadata.yaml"]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"{root} missing required SAGE files: {', '.join(missing)}")
    metadata = yaml.safe_load((root / "metadata.yaml").read_text(encoding="utf-8")) or {}
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


def load_initial_state(data_dir: str | Path, joint_names: list[str] | None = None) -> SageInitialState:
    """Load the first real motor state row in requested joint order."""
    root = Path(data_dir)
    validate_sage_dir(root)
    source_joints = load_joint_list(root / "joint_list.txt")
    requested_joints = list(joint_names or source_joints)

    with (root / "state_motor.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = set(reader.fieldnames or [])
        if "positions" not in fieldnames:
            raise ValueError(f"{root / 'state_motor.csv'} missing required column: positions")
        for row in reader:
            row_type = row.get("type")
            if row_type and row_type != "STATE_MOTOR":
                continue
            positions = parse_vector(row["positions"])
            velocities = parse_vector(row["velocities"]) if row.get("velocities") else None
            return SageInitialState(
                joint_names=requested_joints,
                positions=_select_vector(positions, source_joints, requested_joints, "positions"),
                velocities=(
                    _select_vector(velocities, source_joints, requested_joints, "velocities")
                    if velocities is not None else None
                ),
            )
    raise ValueError(f"{root / 'state_motor.csv'} has no STATE_MOTOR rows")


def load_real_data(
    data_dir: str | Path,
    target_dt: float | None = None,
    joint_names: list[str] | None = None,
    require_state_dynamics: bool = False,
    hold_controls: bool = False,
) -> SageDataset:
    root = Path(data_dir)
    metadata = validate_sage_dir(root)
    source_joints = load_joint_list(root / "joint_list.txt")
    requested_joints = list(joint_names or source_joints)

    control_t_array, control_fields_array = load_vector_csv(root / "control.csv", ("positions",))
    state_path = root / "state_motor.csv"
    with state_path.open(newline="", encoding="utf-8") as stream:
        state_header = set(csv.DictReader(stream).fieldnames or [])
    dynamic_columns = ("velocities", "torques")
    if require_state_dynamics:
        missing = [column for column in dynamic_columns if column not in state_header]
        if missing:
            raise ValueError(
                f"{state_path} missing required actuator-training columns: {', '.join(missing)}"
            )
    state_columns = ("positions",) + tuple(
        column for column in dynamic_columns if column in state_header
    )
    state_t_array, state_fields_array = load_vector_csv(
        state_path,
        state_columns,
    )
    control_t = _normalize_timestamps(control_t_array.tolist(), metadata)
    state_t = _normalize_timestamps(state_t_array.tolist(), metadata)
    control_pos = _select_joints(
        control_fields_array["positions"].tolist(),
        source_joints,
        requested_joints,
        "control positions",
    )
    state_fields = {
        name: _select_joints(values.tolist(), source_joints, requested_joints, name)
        for name, values in state_fields_array.items()
    }
    control_t, control_pos = _sort_rows(control_t, control_pos)
    state_t, state_fields = _sort_vector_fields(state_t, state_fields)

    state_pos = state_fields["positions"]
    state_vel = state_fields.get("velocities", [])
    state_torque = state_fields.get("torques", [])

    normalized_metadata = dict(metadata)
    normalized_metadata["source_time_unit"] = metadata.get("time_unit")
    normalized_metadata["time_unit"] = "seconds"

    if target_dt is not None:
        if not control_t or not state_t:
            raise ValueError(f"{root} has no rows to interpolate")
        overlap_start = max(control_t[0], state_t[0])
        overlap_end = min(control_t[-1], state_t[-1])
        if overlap_end < overlap_start:
            raise ValueError("control and state traces do not overlap in time")
        target_times = _uniform_grid(overlap_start, overlap_end, float(target_dt))
        normalized_metadata["target_dt"] = float(target_dt)
        return SageDataset(
            joint_names=requested_joints,
            control_timestamps=target_times,
            control_positions=(
                _hold_vectors(control_t, control_pos, target_times)
                if hold_controls
                else _interpolate_vectors(control_t, control_pos, target_times)
            ),
            state_timestamps=target_times,
            state_positions=_interpolate_vectors(state_t, state_pos, target_times),
            state_velocities=(
                _interpolate_vectors(state_t, state_vel, target_times)
                if state_vel
                else []
            ),
            state_torques=(
                _interpolate_vectors(state_t, state_torque, target_times)
                if state_torque
                else []
            ),
            metadata=normalized_metadata,
        )

    if abs(len(control_pos) - len(state_pos)) > 1:
        raise ValueError(
            f"control/state row counts differ by more than one row: control={len(control_pos)}, state={len(state_pos)}. "
            "Align asynchronous streams by timestamp before loading."
        )
    n = min(len(control_pos), len(state_pos))
    return SageDataset(
        joint_names=requested_joints,
        control_timestamps=control_t[:n],
        control_positions=control_pos[:n],
        state_timestamps=state_t[:n],
        state_positions=state_pos[:n],
        state_velocities=state_vel[:n],
        state_torques=state_torque[:n],
        metadata=normalized_metadata,
    )
