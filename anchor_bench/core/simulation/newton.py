"""Newton simulation context."""

from __future__ import annotations

from collections.abc import Hashable
from math import isfinite
from pathlib import Path
from typing import Any, Callable, Mapping, Tuple

import newton

from anchor_bench.core.simulation import SimulationContext


class NewtonSimulationContext(SimulationContext[newton.Model, newton.State]):
    """Own one initialized Newton simulation for the process.

    Call :meth:`initialize` before reading resources when more than one world is
    required. Resource access before explicit initialization creates the default
    single-world simulation and prevents later initialization with another world
    count.
    """

    _initialized: bool
    _world_count: int
    _configuration_key: Hashable | None
    _solver_options: dict[str, Any]
    _state_in: newton.State
    _state_out: newton.State
    _control: newton.Control
    _contacts: newton.Contacts
    _solver: newton.solvers.SolverMuJoCo

    def __init__(
        self,
        usd_path: str,
        *,
        fixed_base: bool = False,
    ) -> None:
        """Initialize instance bookkeeping without resetting the singleton."""

        super().__init__(usd_path, fixed_base=fixed_base)
        if not hasattr(self, "_initialized"):
            self._initialized = False

    def initialize(
        self,
        *,
        world_count: int = 1,
        configure_builder: Callable[[newton.ModelBuilder], None] | None = None,
        configuration_key: Hashable | None = None,
        solver_options: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize and own one model, solver, and set of simulation buffers.

        ``configuration_key`` must stably identify any mutations performed by
        ``configure_builder`` so equivalent callbacks can reuse the context while
        incompatible builder configurations are rejected.
        """

        if (
            isinstance(world_count, bool)
            or not isinstance(world_count, int)
            or world_count < 1
        ):
            raise ValueError("world_count must be a positive integer.")
        if configure_builder is not None and configuration_key is None:
            raise ValueError(
                "configuration_key is required when configure_builder is provided."
            )
        if configure_builder is None and configuration_key is not None:
            raise ValueError(
                "configure_builder is required when configuration_key is provided."
            )
        requested_solver_options = dict(solver_options or {})
        if self._initialized:
            if (
                self._world_count != world_count
                or self._configuration_key != configuration_key
                or self._solver_options != requested_solver_options
            ):
                raise RuntimeError(
                    "Newton simulation context is already initialized with a "
                    "different world count or configuration."
                )
            return

        template = self._create_usd_builder()
        if configure_builder is not None:
            configure_builder(template)

        builder = self._create_builder()
        builder.replicate(template, world_count=world_count)
        model = builder.finalize()
        state_in = model.state()
        state_out = model.state()
        control = model.control()
        contacts = model.contacts()
        solver = newton.solvers.SolverMuJoCo(model, **requested_solver_options)

        self._model = model
        self._state = state_in
        self._state_in = state_in
        self._state_out = state_out
        self._control = control
        self._contacts = contacts
        self._solver = solver
        self._world_count = world_count
        self._configuration_key = configuration_key
        self._solver_options = requested_solver_options
        self._initialized = True

    @property
    def world_count(self) -> int:
        """Return the number of worlds in the owned model."""

        self._ensure_initialized()
        return self._world_count

    @property
    def state_in(self) -> newton.State:
        """Return the current input state for all worlds."""

        self._ensure_initialized()
        return self._state_in

    @property
    def state_out(self) -> newton.State:
        """Return the current output state for all worlds."""

        self._ensure_initialized()
        return self._state_out

    @property
    def control(self) -> newton.Control:
        """Return the shared control buffer for all worlds."""

        self._ensure_initialized()
        return self._control

    @property
    def contacts(self) -> newton.Contacts:
        """Return the shared contacts buffer for all worlds."""

        self._ensure_initialized()
        return self._contacts

    @property
    def solver(self) -> newton.solvers.SolverMuJoCo:
        """Return the solver permanently bound to the owned model."""

        self._ensure_initialized()
        return self._solver

    def step(self, *, dt: float) -> None:
        """Advance every world one step and rotate the shared state buffers."""

        if not isfinite(dt) or dt <= 0.0:
            raise ValueError("simulation timestep must be finite and positive.")
        state_in = self.state_in
        state_out = self.state_out
        state_in.clear_forces()
        self.solver.step(
            state_in=state_in,
            state_out=state_out,
            control=self.control,
            contacts=self.contacts,
            dt=dt,
        )
        self._state_in, self._state_out = state_out, state_in
        self._state = self._state_in

    def _load(self, usd_path: Path) -> Tuple[newton.Model, newton.State]:
        """Return the owned model and input state, initializing if needed."""

        self._ensure_initialized()
        model = self._model
        if model is None:
            raise RuntimeError("Newton simulation context did not initialize a model.")
        return model, self._state_in

    def _ensure_initialized(self) -> None:
        """Initialize the default simulation if no explicit request was made."""

        if not self._initialized:
            self.initialize()

    def _create_usd_builder(self) -> newton.ModelBuilder:
        """Create a configured builder and load the context's USD asset."""

        builder = self._create_builder()
        builder.add_usd(
            source=str(self.usd_path),
            floating=not self.fixed_base,
        )
        return builder

    def _create_builder(self) -> newton.ModelBuilder:
        """Create a builder with Newton solver attributes registered."""

        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        return builder
