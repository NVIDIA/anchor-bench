# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit simulation context registration and sysID callbacks."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, TypedDict

from anchor_bench.core.simulation import SimulationContext
from anchor_bench.core.simulation.newton import NewtonSimulationContext
from anchor_bench.errors import SimulationBackendError

if TYPE_CHECKING:
    from anchor_bench.schema.simulation import JsonValue, SysIdSimulationConfig
    from anchor_bench.tasks.actuator_sys_id.model import IdentifiedModelArtifact
    from anchor_bench.tasks.actuator_sys_id.objective import PopulationEvaluator


class _SysIdCallbacks(TypedDict):
    config_model: type[SysIdSimulationConfig]
    create_context: Callable[..., SimulationContext[Any, Any]]
    create_evaluator: Callable[..., PopulationEvaluator]
    export_model: Callable[..., IdentifiedModelArtifact | None] | None
    describe_runtime: Callable[..., Mapping[str, JsonValue]] | None


SIMULATION_CONTEXTS: dict[str, type[SimulationContext[Any, Any]]] = {
    "newton": NewtonSimulationContext,
}
_SYSID_CALLBACKS: dict[str, _SysIdCallbacks] = {}


def register_simulation_backend(
    name: str,
    context_type: type[SimulationContext[Any, Any]],
    *,
    config_model: type[SysIdSimulationConfig],
    create_context: Callable[..., SimulationContext[Any, Any]],
    create_evaluator: Callable[..., PopulationEvaluator],
    export_model: Callable[..., IdentifiedModelArtifact | None] | None = None,
    describe_runtime: Callable[..., Mapping[str, JsonValue]] | None = None,
) -> None:
    """Register a simulator and the callbacks used to run sysID with it."""
    if name in SIMULATION_CONTEXTS:
        raise SimulationBackendError(
            f"Simulation backend {name!r} is already registered."
        )
    SIMULATION_CONTEXTS[name] = context_type
    _SYSID_CALLBACKS[name] = {
        "config_model": config_model,
        "create_context": create_context,
        "create_evaluator": create_evaluator,
        "export_model": export_model,
        "describe_runtime": describe_runtime,
    }


def get_sysid_callbacks(name: str) -> _SysIdCallbacks:
    """Return explicitly registered callbacks, including the built-in Newton ones."""
    if name == "newton" and name not in _SYSID_CALLBACKS:
        from anchor_bench.schema.simulation import NewtonSimulationConfig
        from anchor_bench.tasks.actuator_sys_id.backends import newton

        _SYSID_CALLBACKS[name] = {
            "config_model": NewtonSimulationConfig,
            "create_context": newton.create_context,
            "create_evaluator": newton.create_evaluator,
            "export_model": newton.export_model,
            "describe_runtime": newton.describe_runtime,
        }
    if name not in SIMULATION_CONTEXTS:
        raise SimulationBackendError(f"Unknown simulation backend {name!r}.")
    if name not in _SYSID_CALLBACKS:
        raise SimulationBackendError(
            f"Simulation backend {name!r} does not support system identification."
        )
    return _SYSID_CALLBACKS[name]


def create_simulation_context(
    backend: str,
    usd_path: str,
    *,
    fixed_base: bool = False,
) -> SimulationContext[Any, Any]:
    """Create the context registered for a configured simulator."""
    try:
        context_type = SIMULATION_CONTEXTS[backend]
    except KeyError as error:
        raise ValueError(f"Unknown simulation backend {backend!r}.") from error
    return context_type(usd_path, fixed_base=fixed_base)
