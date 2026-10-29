# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Newton evaluation and native USD export for system identification."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from anchor_bench.core.simulation import SimulationContext
from anchor_bench.core.simulation.newton import NewtonSimulationContext
from anchor_bench.schema.actuator_sys_id import ActuatorSysIdConfig
from anchor_bench.schema.simulation import JsonValue, NewtonSimulationConfig
from anchor_bench.tasks.actuator_sys_id.model import Candidate, IdentifiedModelArtifact
from anchor_bench.tasks.actuator_sys_id.objective import PopulationEvaluator
from anchor_bench.tasks.actuator_sys_id.parameter_space import ParameterSpace
from anchor_bench.tasks.actuator_sys_id.simulation import NewtonPopulationEvaluator


def _simulation_config(config: ActuatorSysIdConfig) -> NewtonSimulationConfig:
    if not isinstance(config.simulation, NewtonSimulationConfig):
        raise TypeError("Newton backend requires NewtonSimulationConfig")
    return config.simulation


def create_context(config: ActuatorSysIdConfig) -> NewtonSimulationContext:
    """Resolve the fixed-base Newton resource selected by the run."""
    simulation = _simulation_config(config)
    return NewtonSimulationContext(simulation.usd_path, fixed_base=True)


def create_evaluator(
    config: ActuatorSysIdConfig,
    space: ParameterSpace,
    context: SimulationContext[Any, Any],
) -> NewtonPopulationEvaluator:
    """Reuse the selected context for population evaluation."""
    if not isinstance(context, NewtonSimulationContext):
        raise TypeError("Newton backend requires a Newton simulation context")
    return NewtonPopulationEvaluator(config, space, simulation_context=context)


def export_model(
    config: ActuatorSysIdConfig,
    best: Candidate,
    context: SimulationContext[Any, Any],
    output: Path,
) -> IdentifiedModelArtifact:
    """Author fitted actuator properties as the existing native USD layer."""
    from anchor_bench.tasks.actuator_sys_id.usd import write_identified_usd_layer

    simulation = _simulation_config(config)
    path = output / "identified.usda"
    write_identified_usd_layer(
        simulation.usd_path,
        path,
        best,
        {
            joint: group.controller.clamps
            for group in config.joint_parameters
            for joint in group.names
        },
        config.model_dump_json(),
    )
    return IdentifiedModelArtifact(path, "usd")


def describe_runtime(evaluator: PopulationEvaluator) -> Mapping[str, JsonValue]:
    """Describe Newton timing and solver configuration used by the evaluator."""
    if not isinstance(evaluator, NewtonPopulationEvaluator):
        raise TypeError("Newton runtime description requires its population evaluator")
    return evaluator.simulation_config.model_dump(mode="json")
