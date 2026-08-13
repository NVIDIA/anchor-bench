"""Shared eager and graph-captured Newton solver stepping."""

from __future__ import annotations

from math import isfinite

import newton
import warp as wp


class NewtonStepExecutor:
    """Advance a Newton solver while preserving stable state-buffer identities.

    Every interval starts and ends in ``state_in``. For odd substep counts, the
    final result is copied from ``state_out`` back to that canonical buffer. The
    same interval routine is used eagerly and during Warp graph capture.
    """

    def __init__(
        self,
        *,
        model: newton.Model,
        solver: newton.solvers.SolverBase,
        state_in: newton.State,
        state_out: newton.State,
        control: newton.Control | None,
        contacts: newton.Contacts | None,
        clear_forces: bool = True,
    ) -> None:
        self._model = model
        self._solver = solver
        self._state_in = state_in
        self._state_out = state_out
        self._control = control
        self._contacts = contacts
        self._clear_forces = clear_forces
        self._graph: wp.Graph | None = None
        self._graph_key: tuple[float, int] | None = None

    @property
    def state_in(self) -> newton.State:
        """Return the current input state."""

        return self._state_in

    @property
    def state_out(self) -> newton.State:
        """Return the available output state."""

        return self._state_out

    def capture_graph(self, *, dt: float, substeps: int = 1) -> None:
        """Capture one reusable interval graph when supported."""

        physics_dt = self._physics_dt(dt, substeps)
        key = (float(dt), substeps)
        if self._graph is not None and self._graph_key == key:
            return
        if not self._supports_graph_capture():
            self._graph = None
            self._graph_key = None
            return

        with wp.ScopedCapture(device=self._model.device) as capture:
            self._simulate_interval(physics_dt, substeps)
        graph = capture.graph
        if graph is None:
            raise RuntimeError("Warp did not return a captured simulation graph.")
        self._graph = graph
        self._graph_key = key

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
        canonical = self._state_in
        state_in = canonical
        state_out = self._state_out
        for index in range(substeps):
            self._step_once(state_in, state_out, physics_dt)
            if substeps % 2 and index == substeps - 1:
                canonical.assign(state_out)
            else:
                state_in, state_out = state_out, state_in

    def _step_once(
        self,
        state_in: newton.State,
        state_out: newton.State,
        dt: float,
    ) -> None:
        if self._clear_forces:
            state_in.clear_forces()
        self._solver.step(
            state_in=state_in,
            state_out=state_out,
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
