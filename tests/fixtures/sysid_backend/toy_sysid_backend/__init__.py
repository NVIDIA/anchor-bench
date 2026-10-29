# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A USD-backed installed backend for exercising the real sysID CLI."""

from __future__ import annotations

import json
from collections.abc import Sequence
from importlib import import_module
from pathlib import Path
from typing import Any, Literal

import numpy as np

from anchor_bench.core.simulation import SimulationContext
from anchor_bench.core.simulation.registry import register_simulation_backend
from anchor_bench.schema.actuator_sys_id import ActuatorSysIdConfig
from anchor_bench.schema.simulation import SysIdSimulationConfig
from anchor_bench.tasks.actuator_sys_id.model import (
    Candidate,
    IdentifiedModelArtifact,
    Trajectory,
)
from anchor_bench.tasks.actuator_sys_id.optimizer import FloatArray
from anchor_bench.tasks.actuator_sys_id.parameter_space import ParameterSpace


class ToyConfig(SysIdSimulationConfig):
    type: Literal["toy-sysid"] = "toy-sysid"  # pyright: ignore[reportIncompatibleVariableOverride]
    gain: float
    model_name: str


class ToyContext(SimulationContext[float, float]):
    """Load the scalar response scale from the shared USD input."""

    def _load(self, usd_path: Path) -> tuple[float, float]:
        usd = import_module("pxr.Usd")
        stage = usd.Stage.Open(str(usd_path))
        if stage is None:
            raise ValueError("Could not open the response USD")
        scale = stage.GetPrimAtPath("/response").GetAttribute("gain_scale").Get()
        return float(scale), 0.0


class ToyEvaluator:
    def __init__(self, context: ToyContext, gain: float) -> None:
        self._gain = context.model * gain

    def evaluate(
        self,
        population: Sequence[Candidate],
        trajectories: Sequence[Trajectory],
    ) -> FloatArray:
        scores = []
        for candidate in population:
            squared_errors = []
            for trajectory in trajectories:
                stiffness = candidate[trajectory.joint_names[0]]["stiffness"]
                predicted = trajectory.commanded_position * stiffness / self._gain
                squared_errors.extend(
                    np.square(predicted - trajectory.measured_position).ravel()
                )
            scores.append(float(np.mean(squared_errors)))
        return np.asarray(scores, dtype=np.float64)


def create_context(config: ActuatorSysIdConfig) -> ToyContext:
    simulation = config.simulation
    if not isinstance(simulation, ToyConfig):
        raise TypeError("toy backend requires its registered config model")
    return ToyContext(simulation.usd_path)


def create_evaluator(
    config: ActuatorSysIdConfig,
    space: ParameterSpace,
    context: SimulationContext[Any, Any],
) -> ToyEvaluator:
    if not isinstance(context, ToyContext):
        raise TypeError("toy evaluator requires its registered context")
    if not isinstance(config.simulation, ToyConfig):
        raise TypeError("toy evaluator requires its registered config model")
    return ToyEvaluator(context, config.simulation.gain)


def export_model(
    config: ActuatorSysIdConfig,
    best: Candidate,
    context: SimulationContext[Any, Any],
    output: Path,
) -> IdentifiedModelArtifact:
    path = output / "identified.json"
    path.write_text(
        json.dumps(
            {
                "parameters": best,
                "simulation": config.simulation.model_dump(mode="json"),
                "gain_scale": context.model,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return IdentifiedModelArtifact(path=path, format="json")


def provide_json() -> None:
    register_simulation_backend(
        "toy-sysid",
        ToyContext,
        config_model=ToyConfig,
        create_context=create_context,
        create_evaluator=create_evaluator,
        export_model=export_model,
    )


def provide_no_export() -> None:
    register_simulation_backend(
        "toy-sysid",
        ToyContext,
        config_model=ToyConfig,
        create_context=create_context,
        create_evaluator=create_evaluator,
    )
