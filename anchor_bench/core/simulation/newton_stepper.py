# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared eager and graph-captured Newton solver stepping."""

from __future__ import annotations

import logging
from math import isfinite

import newton
import warp as wp

logger = logging.getLogger(__name__)


class NewtonStepExecutor:
    """Advance a Newton solver while preserving stable state-buffer identities.

    Every interval starts and ends in ``state``. For odd substep counts,
    the final result is copied from ``state_next`` back to that canonical buffer.
    The same interval routine is used eagerly and during Warp graph capture.
    """

    def __init__(
        self,
        *,
        model: newton.Model,
        solver: newton.solvers.SolverBase,
        state: newton.State,
        state_next: newton.State,
        control: newton.Control | None,
        contacts: newton.Contacts | None,
        clear_forces: bool = True,
    ) -> None:
        self._model = model
        self._solver = solver
        self._state = state
        self._state_next = state_next
        self._control = control
        self._contacts = contacts
        self._clear_forces = clear_forces
        self._graph: wp.Graph | None = None
        self._graph_key: tuple[float, int] | None = None

    @property
    def state(self) -> newton.State:
        """Return the current state."""

        return self._state

    @property
    def state_next(self) -> newton.State:
        """Return the buffer for the next state."""

        return self._state_next

    def capture_graph(self, *, dt: float, substeps: int = 1) -> None:
        """Capture one reusable interval graph when supported."""

        physics_dt = self._physics_dt(dt, substeps)
        key = (float(dt), substeps)
        if self._graph is not None and self._graph_key == key:
            return
        if not self._supports_graph_capture():
            logger.debug(
                "Newton step graph capture unsupported; using eager stepping "
                "dt=%s substeps=%d solver=%s",
                dt,
                substeps,
                type(self._solver).__name__,
            )
            self._graph = None
            self._graph_key = None
            return

        logger.debug(
            "Capturing Newton step graph dt=%s substeps=%d device=%s",
            dt,
            substeps,
            self._model.device,
        )
        with wp.ScopedCapture(device=self._model.device) as capture:
            self._simulate_interval(physics_dt, substeps)
        graph = capture.graph
        if graph is None:
            raise RuntimeError("Warp did not return a captured simulation graph.")
        self._graph = graph
        self._graph_key = key
        logger.debug(
            "Captured Newton step graph dt=%s substeps=%d device=%s",
            dt,
            substeps,
            self._model.device,
        )

    def step(self, *, dt: float, substeps: int = 1) -> None:
        """Launch a matching captured graph or advance the interval eagerly."""

        physics_dt = self._physics_dt(dt, substeps)
        graph = self._graph
        if graph is not None and self._graph_key == (float(dt), substeps):
            wp.capture_launch(graph)
            return
        self._simulate_interval(physics_dt, substeps)

    def _supports_graph_capture(self) -> bool:
        if getattr(self._solver, "use_mujoco_cpu", False):
            return False
        if not hasattr(self._model, "device"):
            return False
        # SolverMuJoCo evaluates this interval using a Python-side step counter.
        # Values other than 0 or 1 would freeze a non-repeatable update pattern
        # into the graph.
        return getattr(self._solver, "update_data_interval", 1) in (0, 1)

    def _simulate_interval(self, physics_dt: float, substeps: int) -> None:
        canonical = self._state
        state = canonical
        state_next = self._state_next
        for index in range(substeps):
            self._step_once(state, state_next, physics_dt)
            if substeps % 2 and index == substeps - 1:
                canonical.assign(state_next)
            else:
                state, state_next = state_next, state

    def _step_once(
        self,
        state: newton.State,
        state_next: newton.State,
        dt: float,
    ) -> None:
        if self._clear_forces:
            state.clear_forces()
        self._solver.step(
            state_in=state,
            state_out=state_next,
            control=self._control,
            contacts=self._contacts,
            dt=dt,
        )

    @staticmethod
    def _physics_dt(dt: float, substeps: int) -> float:
        if not isfinite(dt) or dt <= 0.0:
            raise ValueError("simulation timestep must be finite and positive.")
        if isinstance(substeps, bool) or not isinstance(substeps, int) or substeps < 1:
            raise ValueError("simulation substeps must be a positive integer.")
        return dt / substeps
