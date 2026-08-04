"""Simulation backend registry."""

from __future__ import annotations

from anchor_bench.core.simulation import SimulationContext
from anchor_bench.core.simulation.newton import NewtonSimulationContext

SIMULATION_CONTEXTS: dict[str, type[SimulationContext]] = {
    "newton": NewtonSimulationContext,
}


def create_simulation_context(backend: str, usd_path: str) -> SimulationContext:
    """Create the simulation context registered for a configured backend."""

    try:
        context_type = SIMULATION_CONTEXTS[backend]
    except KeyError as error:
        raise ValueError(f"Unknown simulation backend {backend!r}.") from error
    return context_type(usd_path)
