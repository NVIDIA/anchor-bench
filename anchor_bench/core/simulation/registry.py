# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit simulation context registration."""

from typing import Any

from anchor_bench.core.simulation import SimulationContext
from anchor_bench.core.simulation.newton import NewtonSimulationContext
from anchor_bench.errors import SimulationBackendError

SIMULATION_CONTEXTS: dict[str, type[SimulationContext[Any, Any]]] = {
    "newton": NewtonSimulationContext,
}


def register_simulation_backend(
    name: str, context_type: type[SimulationContext[Any, Any]]
) -> None:
    """Register a simulator context class."""
    if name in SIMULATION_CONTEXTS:
        raise SimulationBackendError(
            f"Simulation backend {name!r} is already registered."
        )
    SIMULATION_CONTEXTS[name] = context_type


def get_simulation_context_type(name: str) -> type[SimulationContext[Any, Any]]:
    """Return the context class registered for a simulator."""
    try:
        return SIMULATION_CONTEXTS[name]
    except KeyError as error:
        raise ValueError(f"Unknown simulation backend {name!r}.") from error


def create_simulation_context(
    backend: str, usd_path: str, *, fixed_base: bool = False
) -> SimulationContext[Any, Any]:
    """Create the context registered for a configured simulator."""
    return get_simulation_context_type(backend)(usd_path, fixed_base=fixed_base)
