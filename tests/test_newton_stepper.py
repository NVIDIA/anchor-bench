from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest

from anchor_bench.train.validators import newton_stepper
from anchor_bench.train.validators.newton_stepper import _NewtonStepper


class FakeArray:
    """Provide the Newton array operations used by the stepper."""

    def __init__(self, values: list[float]) -> None:
        """Store a mutable NumPy value array."""

        self.values = np.asarray(values, dtype=np.float32)

    def numpy(self) -> np.ndarray:
        """Return the current host values."""

        return self.values.copy()

    def assign(self, values: np.ndarray) -> None:
        """Replace the current values."""

        self.values = np.asarray(values, dtype=np.float32).copy()


class FakeState:
    """Hold fake Newton generalized coordinates."""

    def __init__(self) -> None:
        """Create scalar position and velocity arrays."""

        self.joint_q = FakeArray([0.0])
        self.joint_qd = FakeArray([0.0])


class FakeControl:
    """Hold fake Newton target and effort arrays."""

    def __init__(self) -> None:
        """Create scalar control arrays."""

        self.joint_target_q = FakeArray([0.0])
        self.joint_target_qd = FakeArray([0.0])
        self.joint_f = FakeArray([0.0])


class FakeNewtonModel:
    """Expose the scalar-joint model fields used by replay."""

    def __init__(self) -> None:
        """Create one authored position-controlled joint."""

        self.joint_label = ["joint"]
        self.joint_q_start = FakeArray([0.0, 1.0])
        self.joint_qd_start = FakeArray([0.0, 1.0])
        self.joint_target_q_start = FakeArray([0.0, 1.0])
        self.joint_target_mode = FakeArray([3.0])
        self.joint_target_ke = FakeArray([4.0])
        self.joint_target_kd = FakeArray([0.5])
        self.joint_q = FakeArray([0.0])
        self.joint_dof_count = 1
        self.last_control: FakeControl | None = None

    def state(self) -> FakeState:
        """Create independent simulation state."""

        return FakeState()

    def control(self) -> FakeControl:
        """Create and retain one control object."""

        self.last_control = FakeControl()
        return self.last_control


class FakeSolver:
    """Advance the fake state using effort as velocity."""

    instances: ClassVar[list[FakeSolver]] = []

    def __init__(self, model: FakeNewtonModel, **options: object) -> None:
        """Record one created solver."""

        self.reset_count = 0
        self.steps: list[float] = []
        self.instances.append(self)

    def reset(self, state: FakeState) -> None:
        """Record a solver reset."""

        self.reset_count += 1

    def step(
        self,
        state_0: FakeState,
        state_1: FakeState,
        control: FakeControl,
        contacts: object,
        dt: float,
    ) -> None:
        """Integrate the effort over one physical step."""

        velocity = control.joint_f.numpy()
        state_1.joint_q.assign(state_0.joint_q.numpy() + velocity * dt)
        state_1.joint_qd.assign(velocity)
        self.steps.append(dt)


@pytest.fixture
def fake_newton(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the delayed Newton import with the fake solver API."""

    module = SimpleNamespace(
        solvers=SimpleNamespace(SolverMuJoCo=FakeSolver),
        JointTargetMode=SimpleNamespace(EFFORT=4),
    )
    FakeSolver.instances.clear()
    monkeypatch.setattr(newton_stepper, "import_module", lambda name: module)


def _context() -> SimpleNamespace:
    """Build one fake shared model and initial state."""

    model = FakeNewtonModel()
    return SimpleNamespace(model=model, state=model.state())


def test_stepper_resets_steps_and_restores_direct_effort_mode(
    fake_newton: None,
) -> None:
    """Own stepping in the validator while leaving the shared model reusable."""

    context = _context()
    stepper = _NewtonStepper(
        context,
        output_joints=("joint",),
        residual=False,
        substeps=2,
    )

    assert context.model.joint_target_mode.numpy().tolist() == [4.0]
    assert context.model.joint_target_ke.numpy().tolist() == [0.0]
    assert context.model.joint_target_kd.numpy().tolist() == [0.0]
    assert stepper.solver_pd(
        "joint",
        3.0,
        newton_stepper._ReplayState(
            position={"joint": 1.0},
            velocity={"joint": 2.0},
        ),
    ) == pytest.approx(7.0)
    initial = stepper.reset(
        position={"joint": 1.0},
        velocity={"joint": 0.0},
        command={"joint": 3.0},
    )
    result = stepper.step(
        command={"joint": 3.0},
        effort={"joint": 2.0},
        dt=0.2,
    )

    solver = FakeSolver.instances[0]
    assert solver.reset_count == 1
    assert solver.steps == pytest.approx([0.1, 0.1])
    assert initial.position == {"joint": 1.0}
    assert result.position["joint"] == pytest.approx(1.4)
    assert result.velocity == {"joint": 2.0}
    assert context.model.last_control is not None
    assert context.model.last_control.joint_target_q.numpy().tolist() == [3.0]
    assert context.model.last_control.joint_f.numpy().tolist() == [2.0]

    stepper.close()

    assert context.model.joint_target_mode.numpy().tolist() == [3.0]
    assert context.model.joint_target_ke.numpy().tolist() == [4.0]
    assert context.model.joint_target_kd.numpy().tolist() == [0.5]


def test_stepper_retains_usd_actuation_for_residual_effort(
    fake_newton: None,
) -> None:
    """Leave the authored PD and friction path enabled for residual output."""

    context = _context()
    stepper = _NewtonStepper(
        context,
        output_joints=("joint",),
        residual=True,
        substeps=1,
    )

    assert context.model.joint_target_mode.numpy().tolist() == [3.0]
    stepper.close()
    assert context.model.joint_target_mode.numpy().tolist() == [3.0]


def test_stepper_uses_layout_aware_position_target_index() -> None:
    """Index position targets independently from generalized coordinates."""

    stepper = object.__new__(_NewtonStepper)
    stepper._indices = {"joint": (1, 0, 0)}
    stepper._control = FakeControl()

    stepper._write_control({"joint": 3.0}, {})

    assert stepper._control.joint_target_q.numpy().tolist() == [3.0]
