from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

import pytest

import anchor_bench.train.validators as validator_registry
from anchor_bench.core import SimulationContext
from anchor_bench.tasks.train import ValidatorRunner
from anchor_bench.train import BaseValidator


class RecordingValidator(BaseValidator):
    """Provide an inert concrete validator for runner tests."""

    def validate(self, model: object, validation_dataset: object) -> object:
        """Return the stored configuration for an observable test result."""

        return self.config


@dataclass(frozen=True)
class ValidatorConfig:
    """Match the fields exposed by TrainingValidatorConfig."""

    name: str
    type: Literal["offline", "simulation"]
    entrypoint: str
    config: Mapping[str, Any]


def _register(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    validator_type: type[BaseValidator],
) -> None:
    """Temporarily add a test validator to the fixed built-in registry."""

    monkeypatch.setitem(
        validator_registry.VALIDATORS,
        name,
        (__name__, validator_type.__name__),
    )


def test_instantiates_validators_in_configuration_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve every entrypoint and retain its name, type, and config."""

    _register(monkeypatch, "first", RecordingValidator)
    _register(monkeypatch, "second", RecordingValidator)
    simulation_context = object()
    monkeypatch.setattr(SimulationContext, "current", lambda: simulation_context)
    configs = [
        ValidatorConfig(
            name="held-out loss",
            type="offline",
            entrypoint="first",
            config={"split": "test"},
        ),
        ValidatorConfig(
            name="rollout tracking",
            type="simulation",
            entrypoint="second",
            config={"duration": 5.0},
        ),
    ]

    validators = ValidatorRunner.instantiate(configs)

    assert [validator.name for validator in validators] == [
        "held-out loss",
        "rollout tracking",
    ]
    assert [validator.type for validator in validators] == [
        "offline",
        "simulation",
    ]
    assert [validator.config for validator in validators] == [
        {"split": "test"},
        {"duration": 5.0},
    ]
    assert all(
        validator.simulation_context is simulation_context for validator in validators
    )


def test_empty_configuration_returns_no_validators() -> None:
    """Allow the constructor helper to preserve an empty validator sequence."""

    assert ValidatorRunner.instantiate([]) == []


def test_rejects_unknown_entrypoint() -> None:
    """Treat entrypoints only as keys in the built-in registry."""

    config = ValidatorConfig(
        name="external",
        type="offline",
        entrypoint="package.module:validator",
        config={},
    )

    with pytest.raises(AttributeError, match="Unknown validator"):
        ValidatorRunner.instantiate([config])


def test_rejects_registered_non_validator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Require fixed registry entries to resolve to validator subclasses."""

    monkeypatch.setitem(
        validator_registry.VALIDATORS,
        "invalid",
        (__name__, "ValidatorConfig"),
    )
    config = ValidatorConfig(
        name="invalid",
        type="offline",
        entrypoint="invalid",
        config={},
    )

    with pytest.raises(TypeError, match="must inherit from BaseValidator"):
        ValidatorRunner.instantiate([config])
