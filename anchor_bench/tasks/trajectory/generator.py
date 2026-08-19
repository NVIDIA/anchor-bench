# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate long-form commanded-position datasets from trajectory configs."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from anchor_bench.schema.trajectory import (
    ChirpPiece,
    HoldPiece,
    PoseReference,
    SinePiece,
    Trajectory,
    TrajectoryConfig,
    TransitionPiece,
)

NANOSECONDS_PER_SECOND = 1_000_000_000


def generate_dataset(config: TrajectoryConfig) -> pd.DataFrame:
    """Generate every configured trajectory in the actuator dataset format."""

    frames = [
        _trajectory_frame(config, trajectory) for trajectory in config.trajectories
    ]
    return pd.concat(frames, ignore_index=True)


def generate_positions(
    config: TrajectoryConfig,
    trajectory: Trajectory,
) -> np.ndarray:
    """Evaluate one trajectory as samples-by-actuators position values."""

    sample_rate_hz = config.sample_rate_hz
    actuator_indices = {
        actuator.name: index for index, actuator in enumerate(config.actuators)
    }
    current = _resolve_pose(config, trajectory.initial_pose)
    samples: list[np.ndarray] = []

    for piece in trajectory.pieces:
        sample_count = _piece_sample_count(
            duration_s=float(piece.duration_s),
            sample_rate_hz=sample_rate_hz,
        )

        if isinstance(piece, HoldPiece):
            if piece.pose is not None:
                current = _resolve_pose(config, piece.pose)
            piece_samples = np.repeat(current[None, :], sample_count, axis=0)
        elif isinstance(piece, TransitionPiece):
            target = _resolve_pose(config, piece.to)
            progress = np.arange(sample_count, dtype=np.float64) / sample_count
            if piece.interpolation == "quintic_smoothstep":
                progress = progress**3 * (progress * (progress * 6.0 - 15.0) + 10.0)
            piece_samples = current + progress[:, None] * (target - current)
            current = target
        elif isinstance(piece, SinePiece):
            center = (
                current.copy()
                if piece.center is None
                else _resolve_pose(config, piece.center)
            )
            times = np.arange(sample_count, dtype=np.float64) / sample_rate_hz
            offsets = np.zeros((sample_count, len(config.actuators)))
            for component in piece.components:
                phase = component.phase_rad + (
                    2.0 * np.pi * float(component.frequency_hz) * times
                )
                offsets[:, actuator_indices[component.actuator]] += float(
                    component.amplitude
                ) * np.sin(phase)
            envelope = _ramp_envelope(
                times,
                duration_s=float(piece.duration_s),
                ramp_s=float(piece.ramp_s),
            )
            piece_samples = center + envelope[:, None] * offsets
            current = _sine_endpoint(piece, center, actuator_indices)
        else:
            center = (
                current.copy()
                if piece.center is None
                else _resolve_pose(config, piece.center)
            )
            times = np.arange(sample_count, dtype=np.float64) / sample_rate_hz
            offsets = np.zeros((sample_count, len(config.actuators)))
            for component in piece.components:
                phase = _chirp_phase(
                    times,
                    duration_s=float(piece.duration_s),
                    start_frequency_hz=float(component.start_frequency_hz),
                    end_frequency_hz=float(component.end_frequency_hz),
                    phase_rad=float(component.phase_rad),
                )
                offsets[:, actuator_indices[component.actuator]] += float(
                    component.amplitude
                ) * np.sin(phase)
            envelope = _ramp_envelope(
                times,
                duration_s=float(piece.duration_s),
                ramp_s=float(piece.ramp_s),
            )
            piece_samples = center + envelope[:, None] * offsets
            current = _chirp_endpoint(piece, center, actuator_indices)

        samples.append(piece_samples)

    if not samples or sum(len(piece_samples) for piece_samples in samples) == 0:
        return np.empty((0, len(config.actuators)), dtype=np.float64)
    return np.concatenate(samples)


def _piece_sample_count(*, duration_s: float, sample_rate_hz: int) -> int:
    """Return the rounded piece length, with one sample for positive durations."""

    return max(1, round(duration_s * sample_rate_hz))


def _trajectory_frame(
    config: TrajectoryConfig,
    trajectory: Trajectory,
) -> pd.DataFrame:
    positions = generate_positions(config, trajectory)
    sample_count, actuator_count = positions.shape
    sample_indices = np.arange(sample_count, dtype=np.float64)
    time_ns = np.rint(
        sample_indices * NANOSECONDS_PER_SECOND / config.sample_rate_hz
    ).astype(np.int64)
    repeated_time = np.repeat(time_ns, actuator_count)
    fields = np.tile(
        [f"{actuator.name}/command_q" for actuator in config.actuators],
        sample_count,
    )

    return pd.DataFrame(
        {
            "time_ns": repeated_time,
            "time_utc": pd.to_datetime(repeated_time, unit="ns", utc=True),
            "experiment": pd.Series(
                [trajectory.name] * (sample_count * actuator_count),
                dtype="string",
            ),
            "field": pd.Series(fields, dtype="string"),
            "value": positions.reshape(-1),
        }
    )


def _resolve_pose(
    config: TrajectoryConfig,
    reference: PoseReference,
) -> np.ndarray:
    pose = config.poses[reference] if isinstance(reference, str) else reference
    return np.asarray(pose, dtype=np.float64)


def _ramp_envelope(
    times: np.ndarray,
    *,
    duration_s: float,
    ramp_s: float,
) -> np.ndarray:
    if ramp_s == 0.0:
        return np.ones_like(times)
    return np.minimum.reduce(
        (
            np.ones_like(times),
            times / ramp_s,
            np.maximum(0.0, duration_s - times) / ramp_s,
        )
    )


def _sine_endpoint(
    piece: SinePiece,
    center: np.ndarray,
    actuator_indices: dict[str, int],
) -> np.ndarray:
    endpoint = center.copy()
    envelope = _ramp_envelope(
        np.asarray([float(piece.duration_s)]),
        duration_s=float(piece.duration_s),
        ramp_s=float(piece.ramp_s),
    )[0]
    for component in piece.components:
        phase = float(component.phase_rad) + (
            2.0 * np.pi * float(component.frequency_hz) * float(piece.duration_s)
        )
        endpoint[actuator_indices[component.actuator]] += (
            envelope * float(component.amplitude) * np.sin(phase)
        )
    return endpoint


def _chirp_endpoint(
    piece: ChirpPiece,
    center: np.ndarray,
    actuator_indices: dict[str, int],
) -> np.ndarray:
    endpoint = center.copy()
    duration_s = float(piece.duration_s)
    envelope = _ramp_envelope(
        np.asarray([duration_s]),
        duration_s=duration_s,
        ramp_s=float(piece.ramp_s),
    )[0]
    for component in piece.components:
        phase = _chirp_phase(
            np.asarray([duration_s]),
            duration_s=duration_s,
            start_frequency_hz=float(component.start_frequency_hz),
            end_frequency_hz=float(component.end_frequency_hz),
            phase_rad=float(component.phase_rad),
        )[0]
        endpoint[actuator_indices[component.actuator]] += (
            envelope * float(component.amplitude) * np.sin(phase)
        )
    return endpoint


def _chirp_phase(
    times: Sequence[float] | np.ndarray,
    *,
    duration_s: float,
    start_frequency_hz: float,
    end_frequency_hz: float,
    phase_rad: float,
) -> np.ndarray:
    time_values = np.asarray(times, dtype=np.float64)
    sweep_rate = (end_frequency_hz - start_frequency_hz) / duration_s
    cycles = start_frequency_hz * time_values + 0.5 * sweep_rate * time_values**2
    return phase_rad + 2.0 * np.pi * cycles
