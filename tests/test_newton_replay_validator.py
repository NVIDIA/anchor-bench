# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, ClassVar, Mapping

import pandas as pd
import pytest

import anchor_bench.tasks.train.validators as validator_registry
import anchor_bench.tasks.train.validators.newton as newton_validator
from anchor_bench.core import SimulationContext
from anchor_bench.schema.training import TrainingValidatorConfig
from anchor_bench.tasks.train import (
    MappedTrainingSequence,
    ValidationDataset,
    ValidatorRunner,
)
from anchor_bench.tasks.train.validators.newton import (
    NewtonReplayValidator,
)


class FakeModel:
    """Record evaluation mode changes for replay tests."""

    def __init__(self) -> None:
        """Start outside evaluation mode."""

        self.evaluated = False
        self.state_inputs: list[object] = []

    def eval(self) -> None:
        """Record that validation selected evaluation mode."""

        self.evaluated = True


class FakeReplayAdapter(newton_validator._NewtonReplayAdapter):
    """Provide deterministic stepping without importing Newton."""

    instances: ClassVar[list[FakeReplayAdapter]] = []

    def __init__(
        self,
        simulation_context: Any,
        *,
        output_joints: tuple[str, ...],
        residual: bool,
        substeps: int,
        disable_contacts: bool = False,
    ) -> None:
        """Record construction arguments and initialize state."""

        self.simulation_context = simulation_context
        self.output_joints = output_joints
        self.residual = residual
        self.substeps = substeps
        self.disable_contacts = disable_contacts
        self.position: dict[str, float] = {}
        self.velocity: dict[str, float] = {}
        self.steps: list[tuple[dict[str, float], dict[str, float], float]] = []
        self.reset_count = 0
        self.captured_dts: list[float] = []
        self.closed = False
        self.instances.append(self)

    def reset(
        self,
        *,
        position: Mapping[str, float],
        velocity: Mapping[str, float],
        command: Mapping[str, float],
    ) -> newton_validator._ReplayState:
        """Reset to the supplied measured state."""

        self.reset_count += 1
        self.position = dict(position)
        self.velocity = dict(velocity)
        return self._state()

    def capture_step_graph(self, *, dt: float) -> None:
        """Record explicit graph capture after the replay timestep is known."""

        self.captured_dts.append(dt)

    def step(
        self,
        *,
        command: Mapping[str, float],
        effort: Mapping[str, float],
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

    def dynamic_bias(
        self,
        joint: str,
        state: newton_validator._ReplayState,
    ) -> float:
        """Return a deterministic live dynamic-bias value."""

        return state.position[joint] + 7.0

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
    """Provide the singleton context and fake Newton replay adapter."""

    context = object()
    FakeReplayAdapter.instances.clear()
    monkeypatch.setattr(SimulationContext, "current", lambda: context)
    monkeypatch.setattr(newton_validator, "_NewtonReplayAdapter", FakeReplayAdapter)

    @contextmanager
    def configure(model, dataset, adapter):
        adapter.fake_model = model
        adapter.fake_state = None
        yield

    def predict(adapter, features, command, columns, dt):
        adapter.fake_model.state_inputs.append(adapter.fake_state)
        adapter.fake_state = 1 if adapter.fake_state is None else adapter.fake_state + 1
        return {
            joint: float(features[row, 0, 0])
            for row, joint in enumerate(adapter.output_joints)
        }

    original_reset = FakeReplayAdapter.reset

    def reset(adapter, **kwargs):
        adapter.fake_state = None
        return original_reset(adapter, **kwargs)

    monkeypatch.setattr(FakeReplayAdapter, "reset", reset)
    monkeypatch.setattr(
        NewtonReplayValidator, "_configure_actuator", staticmethod(configure)
    )
    monkeypatch.setattr(
        NewtonReplayValidator, "_actuator_effort", staticmethod(predict)
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
        config={
            "signals": ["position", "velocity"],
            "substeps": 4,
            "disable_contacts": True,
        },
    )

    result = validator.validate(model, _validation_dataset(_sequence()))

    adapter = FakeReplayAdapter.instances[0]
    assert model.evaluated
    assert model.state_inputs == [None, 1]
    assert adapter.residual is False
    assert adapter.substeps == 4
    assert adapter.disable_contacts is True
    assert adapter.reset_count == 1
    assert adapter.captured_dts == [1.0]
    assert [step[1] for step in adapter.steps] == [
        {"joint": 1.0},
        {"joint": 1.0},
    ]
    assert [step[2] for step in adapter.steps] == [1.0, 1.0]
    assert adapter.closed
    assert result.metrics["aggregate/position/rmse"] == 0.0
    assert result.metrics["aggregate/velocity/mae"] == 0.0
    assert result.metrics["experiment/wave/position/joint/rmse"] == 0.0


def test_resets_model_state_between_experiments() -> None:
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

    assert model.state_inputs == [None, 1, None, 1]
    assert FakeReplayAdapter.instances[0].reset_count == 2


def test_residual_target_keeps_newton_baseline_enabled() -> None:
    """Select residual replay from the configured training target."""

    validator = NewtonReplayValidator(
        name="residual rollout",
        type="simulation",
        config={"signals": ["position"]},
    )

    validator.validate(FakeModel(), _validation_dataset(_sequence("torque_residual")))

    assert FakeReplayAdapter.instances[0].residual is True


@pytest.mark.parametrize(
    "config",
    [
        {"signals": []},
        {"signals": ["torque"]},
        {"signals": ["position", "position"]},
        {"substeps": 0},
        {"substeps": True},
        {"disable_contacts": 1},
    ],
)
def test_rejects_invalid_replay_configuration(config: dict[str, object]) -> None:
    """Reject unsupported signals and invalid physical options."""

    with pytest.raises((TypeError, ValueError)):
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


def test_rebuilds_dynamic_bias_and_previous_prediction_inputs() -> None:
    """Build full-torque runtime features from live replay state."""

    adapter = FakeReplayAdapter(
        object(),
        output_joints=("joint",),
        residual=False,
        substeps=1,
    )
    state = newton_validator._ReplayState(
        position={"joint": 2.0},
        velocity={"joint": 3.0},
    )

    values = NewtonReplayValidator._live_inputs(
        ["joint/qfrc_bias", "joint/previous_torque"],
        state,
        {"joint": 4.0},
        adapter,
        {"joint": 5.0},
    )

    assert values.tolist() == [9.0, 5.0]
