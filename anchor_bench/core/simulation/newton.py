# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Newton simulation context."""

from __future__ import annotations

import logging
from collections.abc import Hashable
from pathlib import Path
from typing import Any, Callable, Mapping, Tuple

import newton

from anchor_bench.core.simulation import SimulationContext
from anchor_bench.core.simulation.newton_stepper import NewtonStepExecutor

logger = logging.getLogger(__name__)


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
    _control: newton.Control
    _collision_pipeline: newton.CollisionPipeline
    _contacts: newton.Contacts
    _solver: newton.solvers.SolverMuJoCo
    _step_executor: NewtonStepExecutor

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

        logger.info(
            "Initializing Newton simulation path=%s worlds=%d fixed_base=%s",
            self.usd_path,
            world_count,
            self.fixed_base,
        )
        template = self._create_usd_builder()
        if configure_builder is not None:
            configure_builder(template)

        builder = self._create_builder()
        builder.replicate(template, copy_count=world_count)
        model = builder.finalize()
        state = model.state()
        state_next = model.state()
        control = model.control()
        collision_pipeline = newton.CollisionPipeline(model, broad_phase="explicit")
        contacts = collision_pipeline.contacts()
        solver = newton.solvers.SolverMuJoCo(model, **requested_solver_options)
        step_executor = NewtonStepExecutor(
            model=model,
            solver=solver,
            state=state,
            state_next=state_next,
            control=control,
            contacts=contacts,
        )

        self._model = model
        self._state = state
        self._control = control
        self._collision_pipeline = collision_pipeline
        self._contacts = contacts
        self._solver = solver
        self._step_executor = step_executor
        self._world_count = world_count
        self._configuration_key = configuration_key
        self._solver_options = requested_solver_options
        self._initialized = True
        logger.info(
            "Initialized Newton simulation path=%s worlds=%d fixed_base=%s",
            self.usd_path,
            world_count,
            self.fixed_base,
        )

    @property
    def world_count(self) -> int:
        """Return the number of worlds in the owned model."""

        self._ensure_initialized()
        return self._world_count

    @property
    def state_next(self) -> newton.State:
        """Return the next state buffer for all worlds."""

        self._ensure_initialized()
        return self._step_executor.state_next

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

    def capture_step_graph(self, *, dt: float, substeps: int = 1) -> None:
        """Capture a reusable Warp graph for one simulation interval."""

        self._ensure_initialized()
        self._step_executor.capture_graph(dt=dt, substeps=substeps)

    def step(self, *, dt: float, substeps: int = 1) -> None:
        """Advance every world over one eager or graph-captured interval."""

        self._ensure_initialized()
        self._step_executor.step(dt=dt, substeps=substeps)
        self._state = self._step_executor.state

    def _load(self, usd_path: Path) -> Tuple[newton.Model, newton.State]:
        """Return the owned model and current state, initializing if needed."""

        self._ensure_initialized()
        model = self._model
        if model is None:
            raise RuntimeError("Newton simulation context did not initialize a model.")
        return model, self._step_executor.state

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
