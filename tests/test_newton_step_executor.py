from __future__ import annotations

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
    state_in: FakeState,
    state_out: FakeState,
    solver: FakeSolver,
    control: Any,
) -> NewtonStepExecutor:
    """Cast test doubles once at the concrete Newton API boundary."""

    return NewtonStepExecutor(
        model=cast(newton.Model, SimpleNamespace(device="cpu")),
        solver=cast(newton.solvers.SolverBase, solver),
        state_in=cast(newton.State, state_in),
        state_out=cast(newton.State, state_out),
        control=cast(newton.Control, control),
        contacts=None,
    )


@pytest.mark.parametrize("substeps", [2, 3])
def test_eager_steps_keep_the_input_state_canonical(substeps: int) -> None:
    recorder = GraphRecorder()
    state_in = FakeState(recorder)
    state_out = FakeState(recorder)
    solver = FakeSolver(recorder)
    executor = _fake_executor(
        state_in=state_in,
        state_out=state_out,
        solver=solver,
        control=SimpleNamespace(increment=1.0),
    )

    executor.step(dt=0.1 * substeps, substeps=substeps)

    assert executor.state_in is state_in
    assert executor.state_out is state_out
    assert state_in.value == pytest.approx(substeps)
    assert solver.steps == pytest.approx([0.1] * substeps)
    assert state_in.assign_count == (1 if substeps % 2 else 0)


@pytest.mark.parametrize("substeps", [2, 3])
def test_captured_steps_reuse_canonical_state_and_live_controls(
    monkeypatch: pytest.MonkeyPatch,
    substeps: int,
) -> None:
    recorder = GraphRecorder()
    monkeypatch.setattr(
        newton_stepper.wp,
        "ScopedCapture",
        lambda *, device: FakeCapture(recorder),
    )
    monkeypatch.setattr(newton_stepper.wp, "capture_launch", recorder.launch)
    state_in = FakeState(recorder)
    state_out = FakeState(recorder)
    control = SimpleNamespace(increment=1.0)
    solver = FakeSolver(recorder)
    executor = _fake_executor(
        state_in=state_in,
        state_out=state_out,
        solver=solver,
        control=control,
    )
    dt = 0.1 * substeps

    executor.capture_graph(dt=dt, substeps=substeps)
    assert state_in.value == 0.0
    executor.step(dt=dt, substeps=substeps)
    control.increment = 2.0
    executor.step(dt=dt, substeps=substeps)

    assert recorder.capture_count == 1
    assert executor.state_in is state_in
    assert executor.state_out is state_out
    assert state_in.value == pytest.approx(3.0 * substeps)
    assert solver.steps == pytest.approx([0.1] * (2 * substeps))
    assert state_in.assign_count == (2 if substeps % 2 else 0)


def test_step_uses_eager_execution_without_a_matching_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = GraphRecorder()
    monkeypatch.setattr(
        newton_stepper.wp,
        "ScopedCapture",
        lambda *, device: FakeCapture(recorder),
    )
    monkeypatch.setattr(newton_stepper.wp, "capture_launch", recorder.launch)
    state_in = FakeState(recorder)
    state_out = FakeState(recorder)
    executor = _fake_executor(
        state_in=state_in,
        state_out=state_out,
        solver=FakeSolver(recorder),
        control=SimpleNamespace(increment=1.0),
    )

    executor.step(dt=0.2)
    executor.capture_graph(dt=0.1)
    executor.step(dt=0.2)
    executor.step(dt=0.1)

    assert recorder.capture_count == 1
    assert executor.state_in is state_in
    assert state_in.value == 3.0


@pytest.mark.parametrize(
    ("use_mujoco_cpu", "update_data_interval"),
    [(True, 1), (False, 2)],
)
def test_capture_falls_back_for_unsupported_solver_modes(
    use_mujoco_cpu: bool,
    update_data_interval: int,
) -> None:
    recorder = GraphRecorder()
    state_in = FakeState(recorder)
    state_out = FakeState(recorder)
    solver = FakeSolver(recorder)
    solver.use_mujoco_cpu = use_mujoco_cpu
    solver.update_data_interval = update_data_interval
    executor = _fake_executor(
        state_in=state_in,
        state_out=state_out,
        solver=solver,
        control=SimpleNamespace(increment=1.0),
    )

    executor.capture_graph(dt=0.1)
    executor.step(dt=0.1)

    assert recorder.capture_count == 0
    assert executor.state_in is state_in
    assert state_in.value == 1.0
