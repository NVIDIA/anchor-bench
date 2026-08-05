"""Newton simulation context."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any, Tuple

from anchor_bench.core.simulation import SimulationContext


class NewtonSimulationContext(SimulationContext):
    """Load simulation models with Newton."""

    def _load(self, usd_path: Path) -> Tuple[Any, Any]:
        """Build a Newton model and initial state from a USD file."""

        try:
            newton = import_module("newton")
        except ImportError as error:
            raise RuntimeError(
                "NewtonSimulationContext requires the Newton runtime."
            ) from error

        builder = newton.ModelBuilder()
        solver_type = getattr(getattr(newton, "solvers", None), "SolverMuJoCo", None)
        register = getattr(solver_type, "register_custom_attributes", None)
        if callable(register):
            register(builder)
        builder.add_usd(source=str(usd_path))
        model = builder.finalize()
        return model, model.state()
