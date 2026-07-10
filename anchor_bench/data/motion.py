from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Self, Sequence

import numpy as np


def canonicalize_recorded_joint_names(
    names: Sequence[str],
    *,
    canonical_joint_names: Sequence[str],
    aliases: Mapping[str, str],
    context: str = "recorded joint axis",
    permit_unmapped: bool = True,
) -> tuple[str, ...]:
    """Resolve known robot columns while preserving optional context columns."""

    canonical = tuple(canonical_joint_names)
    if not canonical or any(
        not isinstance(name, str) or not name or name != name.strip()
        for name in canonical
    ):
        raise ValueError("canonical_joint_names must contain non-empty names")
    if len(canonical) != len(set(canonical)):
        raise ValueError("canonical_joint_names must be unique")
    canonical_set = set(canonical)
    for alias, target in aliases.items():
        if (
            not isinstance(alias, str)
            or not alias
            or alias != alias.strip()
            or not isinstance(target, str)
            or not target
            or target != target.strip()
        ):
            raise ValueError("joint aliases must map non-empty names")
    bad_targets = sorted({target for target in aliases.values() if target not in canonical_set})
    if bad_targets:
        raise ValueError("joint aliases target unknown canonical joints: " + ", ".join(bad_targets))
    collisions = sorted(set(aliases).intersection(canonical_set))
    if collisions:
        raise ValueError("joint aliases collide with canonical names: " + ", ".join(collisions))

    result: list[str] = []
    owner_by_joint: dict[str, str] = {}
    raw_names: set[str] = set()
    for raw_name in names:
        if (
            not isinstance(raw_name, str)
            or not raw_name
            or raw_name != raw_name.strip()
        ):
            raise ValueError(f"{context} contains an invalid joint name")
        if raw_name in raw_names:
            raise ValueError(f"{context} contains duplicate joint name {raw_name!r}")
        raw_names.add(raw_name)
        resolved = aliases.get(raw_name, raw_name)
        if resolved not in canonical_set:
            if not permit_unmapped:
                raise ValueError(f"{context} contains unknown robot joint {raw_name!r}")
            result.append(raw_name)
            continue
        previous = owner_by_joint.get(resolved)
        if previous is not None:
            raise ValueError(
                f"{context} maps columns {previous!r} and {raw_name!r} "
                f"to robot joint {resolved!r}"
            )
        owner_by_joint[resolved] = raw_name
        result.append(resolved)
    return tuple(result)


def _readonly_array(values: Any, *, ndim: int, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != ndim:
        raise ValueError(f"{label} must be a {ndim}-D array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} contains non-finite values")
    array = np.array(array, dtype=float, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class JointLayout:
    """Canonical joint order plus a reusable source-to-requested permutation."""

    names: tuple[str, ...]

    def __post_init__(self) -> None:
        names = tuple(str(name) for name in self.names)
        if not names:
            raise ValueError("joint layout must contain at least one joint")
        empty = [index for index, name in enumerate(names) if not name]
        if empty:
            raise ValueError(f"joint names must not be empty (indices: {empty})")
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate joint names: {', '.join(duplicates)}")
        object.__setattr__(self, "names", names)

    def indices(self, requested: Sequence[str]) -> np.ndarray:
        requested_names = tuple(str(name) for name in requested)
        if len(set(requested_names)) != len(requested_names):
            raise ValueError("requested joints must be unique")
        lookup = {name: index for index, name in enumerate(self.names)}
        missing = [name for name in requested_names if name not in lookup]
        if missing:
            raise ValueError(f"joints not found in joint_list.txt: {', '.join(missing)}")
        return np.asarray([lookup[name] for name in requested_names], dtype=int)

    def select(self, requested: Sequence[str]) -> tuple[JointLayout, np.ndarray]:
        requested_names = tuple(str(name) for name in requested)
        return JointLayout(requested_names), self.indices(requested_names)


@dataclass(frozen=True)
class MotionSignal:
    """A time-indexed joint signal. Time is always normalized to seconds."""

    time: np.ndarray
    values: np.ndarray
    unit: str
    name: str

    def __post_init__(self) -> None:
        time = _readonly_array(self.time, ndim=1, label=f"{self.name} timestamps")
        values = _readonly_array(self.values, ndim=2, label=self.name)
        if len(time) != len(values):
            raise ValueError(f"{self.name} has {len(values)} rows but {len(time)} timestamps")
        if len(time) > 1 and np.any(np.diff(time) <= 0.0):
            raise ValueError(f"{self.name} timestamps must be strictly increasing")
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "values", values)

    def select_columns(self, indices: np.ndarray) -> Self:
        return replace(self, values=self.values[:, indices])

    def interpolate(self, target_time: np.ndarray) -> Self:
        target = _readonly_array(target_time, ndim=1, label="target timestamps")
        if not len(target):
            return replace(self, time=target, values=np.empty((0, self.values.shape[1])))
        if not len(self.time):
            raise ValueError(f"cannot interpolate empty {self.name} signal")
        columns = [np.interp(target, self.time, self.values[:, index]) for index in range(self.values.shape[1])]
        values = np.column_stack(columns) if columns else np.empty((len(target), 0), dtype=float)
        return replace(self, time=target, values=values)


def uniform_time_grid(start: float, end: float, sample_dt: float) -> np.ndarray:
    if sample_dt <= 0.0:
        raise ValueError("sample_dt must be positive")
    span = end - start
    count = int(round(span / sample_dt)) + 1
    drift = abs(span - (count - 1) * sample_dt)
    if drift > 0.5 * sample_dt:
        raise ValueError(
            f"timestamp span {span} is not a multiple of sample_dt {sample_dt} (drift {drift} exceeds half a step)"
        )
    return np.asarray([round(float(start + index * sample_dt), 12) for index in range(count)], dtype=float)


@dataclass(frozen=True)
class MotionInitialState:
    joints: JointLayout
    positions: np.ndarray
    velocities: np.ndarray | None


@dataclass(frozen=True)
class MotionData:
    """Canonical in-memory representation shared by all data-consuming workflows."""

    joints: JointLayout
    command_position: MotionSignal
    position: MotionSignal
    velocity: MotionSignal | None
    effort: MotionSignal | None
    metadata: Mapping[str, Any]
    source: Path

    def __post_init__(self) -> None:
        width = len(self.joints.names)
        for signal in (self.command_position, self.position, self.velocity, self.effort):
            if signal is not None and signal.values.shape[1] != width:
                raise ValueError(
                    f"{signal.name} has {signal.values.shape[1]} values per row but joint_list.txt has {width} joints"
                )
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "source", Path(self.source))

    @property
    def joint_names(self) -> list[str]:
        return list(self.joints.names)

    def select_joints(self, requested: Sequence[str]) -> Self:
        layout, indices = self.joints.select(requested)
        return replace(
            self,
            joints=layout,
            command_position=self.command_position.select_columns(indices),
            position=self.position.select_columns(indices),
            velocity=self.velocity.select_columns(indices) if self.velocity is not None else None,
            effort=self.effort.select_columns(indices) if self.effort is not None else None,
        )

    def resample(self, sample_dt: float) -> Self:
        signals = [self.command_position, self.position]
        if self.velocity is not None:
            signals.append(self.velocity)
        if self.effort is not None:
            signals.append(self.effort)
        if any(not len(signal.time) for signal in signals):
            raise ValueError("cannot resample an empty motion signal")
        start = max(float(signal.time[0]) for signal in signals)
        end = min(float(signal.time[-1]) for signal in signals)
        if end < start:
            raise ValueError("control and state traces do not overlap in time")
        target = uniform_time_grid(start, end, float(sample_dt))
        metadata = dict(self.metadata)
        metadata["target_dt"] = float(sample_dt)
        return replace(
            self,
            command_position=self.command_position.interpolate(target),
            position=self.position.interpolate(target),
            velocity=self.velocity.interpolate(target) if self.velocity is not None else None,
            effort=self.effort.interpolate(target) if self.effort is not None else None,
            metadata=metadata,
        )

    def initial_state(self) -> MotionInitialState:
        if not len(self.position.time):
            raise ValueError(f"{self.source} has no motor state rows")
        velocities = self.velocity.values[0] if self.velocity is not None else None
        return MotionInitialState(self.joints, self.position.values[0], velocities)

    def training_columns(self, joint: str) -> dict[str, np.ndarray]:
        joint_index = int(self.joints.indices([joint])[0])
        state_time = self.position.time
        command = self.command_position.interpolate(state_time).values[:, joint_index]
        position = self.position.values[:, joint_index]
        columns = {
            "timestamp": state_time,
            "command_position": command,
            "position": position,
            "position_error": command - position,
        }
        if self.velocity is not None:
            columns["velocity"] = self.velocity.interpolate(state_time).values[:, joint_index]
        if self.effort is not None:
            columns["torque"] = self.effort.interpolate(state_time).values[:, joint_index]
        return columns
