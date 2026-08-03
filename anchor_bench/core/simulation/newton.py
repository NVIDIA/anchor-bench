"""Newton simulation context."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any, Tuple

from anchor_bench.core.simulation import SimulationContext


class NewtonSimulationContext(SimulationContext):
    """Load simulation models with Newton."""

    def load(self, usd_path: str) -> Tuple[Any, Any]:
        """Build a Newton model and initial state from a USD file."""

        source = Path(usd_path).expanduser()
        if not source.is_file():
            raise FileNotFoundError(f"USD file does not exist: {source}.")

        try:
            newton = import_module("newton")
        except ImportError as error:
            raise RuntimeError(
                "NewtonSimulationContext requires the Newton runtime."
            ) from error

        builder = newton.ModelBuilder()
        builder.add_usd(source=str(source))
        model = builder.finalize()
        return model, model.state()
