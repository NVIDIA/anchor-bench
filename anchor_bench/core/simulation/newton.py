"""Newton simulation context."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Tuple

import newton

from anchor_bench.core.simulation import SimulationContext


class NewtonSimulationContext(SimulationContext[newton.Model, newton.State]):
    """Load simulation models with Newton."""

    def create_replicated_model(
        self,
        *,
        world_count: int,
        configure_builder: Callable[[newton.ModelBuilder], None] | None = None,
    ) -> newton.Model:
        """Build configurable copies of the context's USD model."""

        if (
            isinstance(world_count, bool)
            or not isinstance(world_count, int)
            or world_count < 1
        ):
            raise ValueError("world_count must be a positive integer.")

        template = self._create_usd_builder()
        if configure_builder is not None:
            configure_builder(template)

        builder = self._create_builder()
        builder.replicate(template, world_count=world_count)
        return builder.finalize()

    def create_mujoco_solver(
        self,
        model: newton.Model,
        **options: Any,
    ) -> newton.solvers.SolverMuJoCo:
        """Create Newton's MuJoCo solver for a model."""

        return newton.solvers.SolverMuJoCo(model, **options)

    def _load(self, usd_path: Path) -> Tuple[newton.Model, newton.State]:
        """Build a Newton model and initial state from a USD file."""

        builder = self._create_usd_builder()
        model = builder.finalize()
        return model, model.state()

    def _create_usd_builder(self) -> newton.ModelBuilder:
        """Create a configured builder and load the context's USD asset."""

        builder = self._create_builder()
        builder.add_usd(source=str(self.usd_path))
        return builder

    def _create_builder(self) -> newton.ModelBuilder:
        """Create a builder with Newton solver attributes registered."""

        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        return builder
