"""Newton simulation context."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Tuple

from anchor_bench.core.simulation import SimulationContext


class NewtonSimulationContext(SimulationContext):
    """Load simulation models with Newton."""

    @property
    def runtime(self) -> Any:
        """Return the lazily imported Newton runtime."""

        runtime = getattr(self, "_runtime", None)
        if runtime is None:
            try:
                runtime = import_module("newton")
            except ImportError as error:
                raise RuntimeError(
                    "NewtonSimulationContext requires the Newton runtime."
                ) from error
            self._runtime = runtime
        return runtime

    def create_replicated_model(
        self,
        *,
        world_count: int,
        configure_builder: Callable[[Any], None] | None = None,
    ) -> Any:
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

    def create_mujoco_solver(self, model: Any, **options: Any) -> Any:
        """Create Newton's MuJoCo solver for a model."""

        solver_type = getattr(
            getattr(self.runtime, "solvers", None),
            "SolverMuJoCo",
            None,
        )
        if solver_type is None:
            raise RuntimeError(
                "NewtonSimulationContext requires newton.solvers.SolverMuJoCo."
            )
        return solver_type(model, **options)

    def _load(self, usd_path: Path) -> Tuple[Any, Any]:
        """Build a Newton model and initial state from a USD file."""

        builder = self._create_usd_builder()
        model = builder.finalize()
        return model, model.state()

    def _create_usd_builder(self) -> Any:
        """Create a configured builder and load the context's USD asset."""

        builder = self._create_builder()
        builder.add_usd(source=str(self.usd_path))
        return builder

    def _create_builder(self) -> Any:
        """Create a builder with Newton solver attributes registered."""

        builder = self.runtime.ModelBuilder()
        solver_type = getattr(
            getattr(self.runtime, "solvers", None),
            "SolverMuJoCo",
            None,
        )
        register = getattr(solver_type, "register_custom_attributes", None)
        if callable(register):
            register(builder)
        return builder
