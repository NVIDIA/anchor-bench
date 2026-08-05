from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest

import anchor_bench.train.validators as validator_registry
import anchor_bench.train.validators.newton as newton_validator
from anchor_bench.core import SimulationContext
from anchor_bench.schema.training import TrainingValidatorConfig
from anchor_bench.tasks.train import (
    MappedTrainingSequence,
    ValidationDataset,
    ValidatorRunner,
)
from anchor_bench.train.validators.newton import (
    NewtonReplayValidator,
)


class FakeModel:
    """Record evaluation mode changes for replay tests."""

    def __init__(self) -> None:
        """Start outside evaluation mode."""

        self.evaluated = False
        self.hidden_inputs: list[object] = []

    def eval(self) -> None:
        """Record that validation selected evaluation mode."""

        self.evaluated = True


class FakeStepper:
    """Provide deterministic stepping without importing Newton."""

    instances: ClassVar[list[FakeStepper]] = []

    def __init__(
        self,
        simulation_context: Any,
        *,
        output_joints: tuple[str, ...],
        residual: bool,
        substeps: int,
    ) -> None:
        """Record construction arguments and initialize state."""

        self.simulation_context = simulation_context
        self.output_joints = output_joints
        self.residual = residual
        self.substeps = substeps
        self.position: dict[str, float] = {}
        self.velocity: dict[str, float] = {}
        self.steps: list[tuple[dict[str, float], dict[str, float], float]] = []
        self.reset_count = 0
        self.closed = False
        self.instances.append(self)

    def reset(
        self,
        *,
        position: dict[str, float],
        velocity: dict[str, float],
        command: dict[str, float],
    ) -> newton_validator._ReplayState:
        """Reset to the supplied measured state."""

        self.reset_count += 1
        self.position = dict(position)
        self.velocity = dict(velocity)
        return self._state()

    def step(
        self,
        *,
        command: dict[str, float],
        effort: dict[str, float],
        dt: float,
    ) -> newton_validator._ReplayState:
        """Integrate effort directly for predictable test trajectories."""

        self.steps.append((dict(command), dict(effort), dt))
        for joint, value in effort.items():
            self.position[joint] += value * dt
            self.velocity[joint] = value
        return self._state()

    def solver_pd(
        self,
        joint: str,
        command: float,
        state: newton_validator._ReplayState,
    ) -> float:
        """Return a simple live position error."""

        return command - state.position[joint]

    def close(self) -> None:
        """Record resource cleanup."""

        self.closed = True

    def _state(self) -> newton_validator._ReplayState:
        """Return the current fake state."""

        return newton_validator._ReplayState(
            position=dict(self.position),
            velocity=dict(self.velocity),
        )


@pytest.fixture(autouse=True)
def replay_dependencies(monkeypatch: pytest.MonkeyPatch) -> object:
    """Provide the singleton context and fake Newton stepper."""

    context = object()
    FakeStepper.instances.clear()
    monkeypatch.setattr(SimulationContext, "current", lambda: context)
    monkeypatch.setattr(newton_validator, "_NewtonStepper", FakeStepper)

    def predict(
        model: FakeModel,
        features: np.ndarray,
        hidden: int | None,
    ) -> tuple[np.ndarray, int]:
        """Return the live feature while recording recurrent-state flow."""

        model.hidden_inputs.append(hidden)
        next_hidden = 1 if hidden is None else hidden + 1
        return features[:, 0, :1], next_hidden

    monkeypatch.setattr(
        NewtonReplayValidator,
        "_predict",
        staticmethod(predict),
    )
    return context


def _sequence(
    target_feature: str = "actual_effort",
    *,
    experiment: str = "wave",
) -> MappedTrainingSequence:
    """Build one replay whose live position error produces the real trajectory."""

    index = pd.MultiIndex.from_product(
        [[experiment], [0, 1_000_000_000, 2_000_000_000]],
        names=["experiment", "time_ns"],
    )
    return MappedTrainingSequence(
        experiment=experiment,
        name="joint",
        inputs=pd.DataFrame(
            {"joint/position_error": [99.0, 99.0, 99.0]},
            index=index,
        ),
        targets=pd.DataFrame(
            {f"joint/{target_feature}": [0.0, 0.0, 0.0]},
            index=index,
        ),
    )


def _validation_dataset(
    *sequences: MappedTrainingSequence,
) -> ValidationDataset:
    """Keep recorded replay signals separate from configured model inputs."""

    frames = [
        pd.DataFrame(
            {
                "joint/actual_q": [0.0, 1.0, 2.0],
                "joint/actual_dq": [0.0, 1.0, 1.0],
                "joint/command_q": [1.0, 2.0, 3.0],
            },
            index=sequence.inputs.index,
        )
        for sequence in sequences
    ]
    return ValidationDataset(
        sequences=tuple(sequences),
        dataframe=pd.concat(frames),
    )


def test_registry_exposes_newton_replay_validator() -> None:
    """Resolve the built-in validator through its YAML entrypoint key."""

    assert validator_registry.newton_replay is NewtonReplayValidator


def test_runner_executes_registered_newton_replay_validator() -> None:
    """Execute the registered validator through its YAML configuration."""

    results = ValidatorRunner.run(
        [
            TrainingValidatorConfig(
                name="rollout",
                type="simulation",
                entrypoint="newton_replay",
                config={"signals": ["position"]},
            )
        ],
        model=FakeModel(),
        validation_dataset=_validation_dataset(_sequence()),
    )

    assert len(results) == 1
    assert results[0].name == "rollout"
    assert results[0].type == "simulation"
    assert results[0].metrics["aggregate/position/rmse"] == 0.0


def test_replays_live_inputs_and_reports_physical_metrics() -> None:
    """Reset, step, and compare simulated position and velocity trajectories."""

    model = FakeModel()
    validator = NewtonReplayValidator(
        name="rollout",
        type="simulation",
        config={"signals": ["position", "velocity"], "substeps": 4},
    )

    result = validator.validate(model, _validation_dataset(_sequence()))

    stepper = FakeStepper.instances[0]
    assert model.evaluated
    assert model.hidden_inputs == [None, 1]
    assert stepper.residual is False
    assert stepper.substeps == 4
    assert stepper.reset_count == 1
    assert [step[1] for step in stepper.steps] == [
        {"joint": 1.0},
        {"joint": 1.0},
    ]
    assert [step[2] for step in stepper.steps] == [1.0, 1.0]
    assert stepper.closed
    assert result.metrics["aggregate/position/rmse"] == 0.0
    assert result.metrics["aggregate/velocity/mae"] == 0.0
    assert result.metrics["experiment/wave/position/joint/rmse"] == 0.0


def test_resets_hidden_state_between_experiments() -> None:
    """Start every validation experiment with an empty recurrent state."""

    model = FakeModel()
    validator = NewtonReplayValidator(
        name="rollout",
        type="simulation",
        config={"signals": ["position"]},
    )

    validator.validate(
        model,
        _validation_dataset(_sequence(), _sequence(experiment="step")),
    )

    assert model.hidden_inputs == [None, 1, None, 1]
    assert FakeStepper.instances[0].reset_count == 2


def test_residual_target_keeps_newton_baseline_enabled() -> None:
    """Select residual replay from the configured training target."""

    validator = NewtonReplayValidator(
        name="residual rollout",
        type="simulation",
        config={"signals": ["position"]},
    )

    validator.validate(FakeModel(), _validation_dataset(_sequence("torque_residual")))

    assert FakeStepper.instances[0].residual is True


@pytest.mark.parametrize(
    "config",
    [
        {"signals": []},
        {"signals": ["torque"]},
        {"signals": ["position", "position"]},
        {"substeps": 0},
        {"substeps": True},
    ],
)
def test_rejects_invalid_replay_configuration(config: dict[str, object]) -> None:
    """Reject unsupported signals and invalid physical substeps."""

    with pytest.raises(ValueError):
        NewtonReplayValidator(
            name="invalid",
            type="simulation",
            config=config,
        )


@pytest.mark.parametrize(
    ("timestamps", "message"),
    [
        ([0, 0, 1], "timestamps must increase"),
        ([0, 1, 3], "timestamps must be uniformly spaced"),
    ],
)
def test_rejects_invalid_timestamps(
    timestamps: list[int],
    message: str,
) -> None:
    """Require positive, constant replay timesteps."""

    sequence = _sequence()
    bad_index = pd.MultiIndex.from_product(
        [["wave"], timestamps],
        names=["experiment", "time_ns"],
    )
    invalid = MappedTrainingSequence(
        experiment=sequence.experiment,
        name=sequence.name,
        inputs=sequence.inputs.set_axis(bad_index),
        targets=sequence.targets.set_axis(bad_index),
    )
    validator = NewtonReplayValidator(
        name="rollout",
        type="simulation",
        config={},
    )

    with pytest.raises(ValueError, match=message):
        validator.validate(FakeModel(), _validation_dataset(invalid))
