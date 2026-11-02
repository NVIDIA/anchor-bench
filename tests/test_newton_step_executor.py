# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, Callable, cast

import newton
import pytest
from typing_extensions import Self

from anchor_bench.core.simulation import newton_stepper
from anchor_bench.core.simulation.newton_stepper import NewtonStepExecutor


class GraphRecorder:
    """Record fake device operations and replay them like a Warp graph."""

    def __init__(self) -> None:
        self.active = False
        self.capture_count = 0
        self.operations: list[Callable[[], None]] = []

    def record(self, operation: Callable[[], None]) -> None:
        if self.active:
            self.operations.append(operation)
        else:
            operation()

    def launch(self, graph: tuple[Callable[[], None], ...]) -> None:
        for operation in graph:
            operation()


class FakeCapture:
    """Expose recorded operations through the ScopedCapture protocol."""

    def __init__(self, recorder: GraphRecorder) -> None:
        self.recorder = recorder
        self.graph: tuple[Callable[[], None], ...] | None = None

    def __enter__(self) -> Self:
        self.recorder.capture_count += 1
        self.recorder.operations = []
        self.recorder.active = True
        return self

    def __exit__(self, *args: object) -> None:
        self.recorder.active = False
        self.graph = tuple(self.recorder.operations)


class FakeState:
    """Hold one scalar value manipulated through recorded operations."""

    def __init__(self, recorder: GraphRecorder, value: float = 0.0) -> None:
        self.recorder = recorder
        self.value = value
        self.clear_count = 0
        self.assign_count = 0

    def clear_forces(self) -> None:
        self.recorder.record(self._clear_forces)

    def _clear_forces(self) -> None:
        self.clear_count += 1

    def assign(self, other: FakeState) -> None:
        self.recorder.record(lambda: self._assign(other))

    def _assign(self, other: FakeState) -> None:
        self.value = other.value
        self.assign_count += 1


class FakeSolver:
    """Integrate a mutable scalar control into fake state buffers."""

    use_mujoco_cpu = False
    update_data_interval = 1

    def __init__(self, recorder: GraphRecorder) -> None:
        self.recorder = recorder
        self.steps: list[float] = []

    def step(
        self,
        *,
        state_in: FakeState,
        state_out: FakeState,
        control: Any,
        contacts: object,
        dt: float,
    ) -> None:
        self.recorder.record(
            lambda: self._step(state_in, state_out, control.increment, dt)
        )

    def _step(
        self,
        state_in: FakeState,
        state_out: FakeState,
        increment: float,
        dt: float,
    ) -> None:
        state_out.value = state_in.value + increment
        self.steps.append(dt)


def _fake_executor(
    *,
    state: FakeState,
    state_next: FakeState,
    solver: FakeSolver,
    control: Any,
) -> NewtonStepExecutor:
    """Cast test doubles once at the concrete Newton API boundary."""

    return NewtonStepExecutor(
        model=cast(newton.Model, SimpleNamespace(device="cpu")),
        solver=cast(newton.solvers.SolverBase, solver),
        state=cast(newton.State, state),
        state_next=cast(newton.State, state_next),
        control=cast(newton.Control, control),
        contacts=None,
    )


@pytest.mark.parametrize("substeps", [2, 3])
def test_eager_steps_keep_the_current_state_canonical(substeps: int) -> None:
    """Keep eager results in the stable current-state buffer."""

    recorder = GraphRecorder()
    state = FakeState(recorder)
    state_next = FakeState(recorder)
    solver = FakeSolver(recorder)
    executor = _fake_executor(
        state=state,
        state_next=state_next,
        solver=solver,
        control=SimpleNamespace(increment=1.0),
    )

    executor.step(dt=0.1 * substeps, substeps=substeps)

    assert executor.state is state
    assert executor.state_next is state_next
    assert state.value == pytest.approx(substeps)
    assert solver.steps == pytest.approx([0.1] * substeps)
    assert state.assign_count == (1 if substeps % 2 else 0)


@pytest.mark.parametrize("substeps", [2, 3])
def test_captured_steps_reuse_canonical_state_and_live_controls(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    substeps: int,
) -> None:
    """Reuse stable state buffers and live controls in captured steps."""

    recorder = GraphRecorder()
    monkeypatch.setattr(
        newton_stepper.wp,
        "ScopedCapture",
        lambda *, device: FakeCapture(recorder),
    )
    monkeypatch.setattr(newton_stepper.wp, "capture_launch", recorder.launch)
    state = FakeState(recorder)
    state_next = FakeState(recorder)
    control = SimpleNamespace(increment=1.0)
    solver = FakeSolver(recorder)
    executor = _fake_executor(
        state=state,
        state_next=state_next,
        solver=solver,
        control=control,
    )
    dt = 0.1 * substeps

    with caplog.at_level(logging.DEBUG, logger=newton_stepper.__name__):
        executor.capture_graph(dt=dt, substeps=substeps)
    assert state.value == 0.0
    executor.step(dt=dt, substeps=substeps)
    control.increment = 2.0
    executor.step(dt=dt, substeps=substeps)

    assert recorder.capture_count == 1
    assert executor.state is state
    assert executor.state_next is state_next
    assert state.value == pytest.approx(3.0 * substeps)
    assert solver.steps == pytest.approx([0.1] * (2 * substeps))
    assert state.assign_count == (2 if substeps % 2 else 0)
    assert "Capturing Newton step graph" in caplog.text
    assert "Captured Newton step graph" in caplog.text


def test_step_uses_eager_execution_without_a_matching_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use eager execution when the captured interval does not match."""

    recorder = GraphRecorder()
    monkeypatch.setattr(
        newton_stepper.wp,
        "ScopedCapture",
        lambda *, device: FakeCapture(recorder),
    )
    monkeypatch.setattr(newton_stepper.wp, "capture_launch", recorder.launch)
    state = FakeState(recorder)
    state_next = FakeState(recorder)
    executor = _fake_executor(
        state=state,
        state_next=state_next,
        solver=FakeSolver(recorder),
        control=SimpleNamespace(increment=1.0),
    )

    executor.step(dt=0.2)
    executor.capture_graph(dt=0.1)
    executor.step(dt=0.2)
    executor.step(dt=0.1)

    assert recorder.capture_count == 1
    assert executor.state is state
    assert state.value == 3.0


@pytest.mark.parametrize(
    ("use_mujoco_cpu", "update_data_interval"),
    [(True, 1), (False, 2)],
)
def test_capture_falls_back_for_unsupported_solver_modes(
    caplog: pytest.LogCaptureFixture,
    use_mujoco_cpu: bool,
    update_data_interval: int,
) -> None:
    """Fall back to eager execution for unsupported solver modes."""

    recorder = GraphRecorder()
    state = FakeState(recorder)
    state_next = FakeState(recorder)
    solver = FakeSolver(recorder)
    solver.use_mujoco_cpu = use_mujoco_cpu
    solver.update_data_interval = update_data_interval
    executor = _fake_executor(
        state=state,
        state_next=state_next,
        solver=solver,
        control=SimpleNamespace(increment=1.0),
    )

    with caplog.at_level(logging.DEBUG, logger=newton_stepper.__name__):
        executor.capture_graph(dt=0.1)
    executor.step(dt=0.1)

    assert recorder.capture_count == 0
    assert executor.state is state
    assert state.value == 1.0
    assert "graph capture unsupported; using eager stepping" in caplog.text


@pytest.mark.parametrize(
    ("dt", "substeps", "message"),
    [
        (0.0, 1, "simulation timestep must be finite and positive"),
        (float("nan"), 1, "simulation timestep must be finite and positive"),
        (0.1, 0, "simulation substeps must be a positive integer"),
        (0.1, True, "simulation substeps must be a positive integer"),
    ],
)
@pytest.mark.parametrize("method_name", ["step", "capture_graph"])
def test_rejects_invalid_interval_parameters_without_solver_calls(
    dt: float,
    substeps: int,
    message: str,
    method_name: str,
) -> None:
    """Reject invalid intervals before invoking the solver."""

    recorder = GraphRecorder()
    solver = FakeSolver(recorder)
    executor = _fake_executor(
        state=FakeState(recorder),
        state_next=FakeState(recorder),
        solver=solver,
        control=SimpleNamespace(increment=1.0),
    )

    method = getattr(executor, method_name)
    with pytest.raises(ValueError, match=message):
        method(dt=dt, substeps=substeps)

    assert solver.steps == []


def test_before_substep_runs_before_every_solver_step() -> None:
    """Run the callback with each substep's input state and timestep."""

    recorder = GraphRecorder()
    state = FakeState(recorder)
    state_next = FakeState(recorder)
    control = SimpleNamespace(increment=1.0)
    solver = FakeSolver(recorder)
    executor = _fake_executor(
        state=state,
        state_next=state_next,
        solver=solver,
        control=control,
    )
    calls: list[tuple[object, object, float, int]] = []

    def before(s: newton.State, c: newton.Control | None, dt: float) -> None:
        calls.append((s, c, dt, len(solver.steps)))

    executor.set_before_substep(before)
    executor.step(dt=0.3, substeps=3)

    assert [call[0] for call in calls] == [state, state_next, state]
    assert all(call[1] is control for call in calls)
    assert [call[2] for call in calls] == pytest.approx([0.1] * 3)
    # The callback sees the solver having completed exactly the prior substeps.
    assert [call[3] for call in calls] == [0, 1, 2]
    assert solver.steps == pytest.approx([0.1] * 3)


def test_before_substep_is_recorded_into_captured_graphs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replay the callback's recorded work on every graph launch."""

    recorder = GraphRecorder()
    monkeypatch.setattr(
        newton_stepper.wp,
        "ScopedCapture",
        lambda *, device: FakeCapture(recorder),
    )
    monkeypatch.setattr(newton_stepper.wp, "capture_launch", recorder.launch)
    state = FakeState(recorder)
    executor = _fake_executor(
        state=state,
        state_next=FakeState(recorder),
        solver=FakeSolver(recorder),
        control=SimpleNamespace(increment=1.0),
    )
    ran: list[float] = []

    def before(s: newton.State, c: newton.Control | None, dt: float) -> None:
        recorder.record(lambda: ran.append(dt))

    executor.capture_graph(dt=0.2, substeps=2)
    executor.set_before_substep(before)
    executor.capture_graph(dt=0.2, substeps=2)
    assert recorder.capture_count == 2
    assert ran == []

    executor.step(dt=0.2, substeps=2)
    executor.step(dt=0.2, substeps=2)

    assert ran == pytest.approx([0.1] * 4)


def test_setting_a_callback_drops_a_captured_graph() -> None:
    """Invalidate captured graphs so the next capture includes the callback."""

    recorder = GraphRecorder()
    executor = _fake_executor(
        state=FakeState(recorder),
        state_next=FakeState(recorder),
        solver=FakeSolver(recorder),
        control=None,
    )
    executor._graph = cast(Any, object())
    executor._graph_key = (0.1, 1)

    executor.set_before_substep(None)

    assert executor._graph is None and executor._graph_key is None
