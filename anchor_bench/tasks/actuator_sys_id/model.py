# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Domain objects shared by the actuator system-identification pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

Candidate = Dict[str, Dict[str, float]]


@dataclass(frozen=True)
class Trajectory:
    """One aligned command/measurement episode."""

    experiment: object
    joint_names: List[str]
    commanded_position: np.ndarray
    measured_position: np.ndarray

    def __post_init__(self) -> None:
        commanded = np.asarray(self.commanded_position, dtype=np.float64)
        measured = np.asarray(self.measured_position, dtype=np.float64)
        expected_width = len(self.joint_names)
        if commanded.ndim != 2 or measured.ndim != 2:
            raise ValueError("trajectory positions must be two-dimensional")
        if commanded.shape != measured.shape:
            raise ValueError(
                "commanded and measured trajectories must have equal shape"
            )
        if commanded.shape[0] < 2:
            raise ValueError("trajectory must contain at least two samples")
        if commanded.shape[1] != expected_width:
            raise ValueError(
                "trajectory width does not match the configured joint count"
            )
        if not np.isfinite(commanded).all() or not np.isfinite(measured).all():
            raise ValueError("trajectory positions must be finite")
        commanded = np.array(commanded, copy=True)
        measured = np.array(measured, copy=True)
        commanded.setflags(write=False)
        measured.setflags(write=False)
        object.__setattr__(self, "commanded_position", commanded)
        object.__setattr__(self, "measured_position", measured)


@dataclass(frozen=True)
class GenerationRecord:
    """Scores recorded after one optimizer generation."""

    generation: int
    best_mse: float
    population_mean_mse: float


@dataclass(frozen=True)
class SysIdResult:
    """Completed system-identification result and generated artifacts."""

    best_parameters: Candidate
    best_mse: float
    generations: int
    output_dir: Path
    best_parameters_path: Path
    optimization_log_path: Path
    run_summary_path: Path
    identified_usd_path: Path
