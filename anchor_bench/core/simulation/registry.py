# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Supported simulation context implementations."""

from typing import Any

from anchor_bench.core.simulation import SimulationContext
from anchor_bench.core.simulation.newton import NewtonSimulationContext

SIMULATION_CONTEXTS: dict[str, type[SimulationContext[Any, Any]]] = {
    "newton": NewtonSimulationContext,
}


def get_simulation_context_type(name: str) -> type[SimulationContext[Any, Any]]:
    """Return the context class configured for a simulator."""
    try:
        return SIMULATION_CONTEXTS[name]
    except KeyError as error:
        raise ValueError(f"Unknown simulation backend {name!r}.") from error


def create_simulation_context(
    backend: str, usd_path: str, *, fixed_base: bool = False
) -> SimulationContext[Any, Any]:
    """Create the context selected for a configured simulator."""
    return get_simulation_context_type(backend)(usd_path, fixed_base=fixed_base)
