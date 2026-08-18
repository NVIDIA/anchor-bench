from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

import pytest

import anchor_bench.train.validators as validator_registry
from anchor_bench.core import SimulationContext
from anchor_bench.tasks.train import ValidatorRunner
from anchor_bench.train import BaseValidator, ValidationResult


class RecordingValidator(BaseValidator):
    """Provide an inert concrete validator for runner tests."""

    def validate(self, model: object, validation_dataset: object) -> ValidationResult:
        """Return one observable scalar metric."""

        return ValidationResult(
            name=self.name,
            type=self.type,
            metrics={"score": float(self.config.get("score", 0.0))},
        )


@dataclass(frozen=True)
class ValidatorConfig:
    """Match the fields exposed by TrainingValidatorConfig."""

    name: str
    type: Literal["offline", "simulation"]
    entrypoint: str
    config: Mapping[str, Any]


@pytest.fixture(autouse=True)
def simulation_context(
    monkeypatch: pytest.MonkeyPatch,
) -> object:
    """Provide the shared simulation context required by every validator."""

    context = object()
    monkeypatch.setattr(SimulationContext, "current", lambda: context)
    return context


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
    simulation_context: object,
) -> None:
    """Resolve every entrypoint and retain its name, type, and config."""

    _register(monkeypatch, "first", RecordingValidator)
    _register(monkeypatch, "second", RecordingValidator)
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


def test_runs_validators_in_configuration_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Execute configured validators and collect results in declaration order."""

    _register(monkeypatch, "first", RecordingValidator)
    _register(monkeypatch, "second", RecordingValidator)
    configs = [
        ValidatorConfig(
            name="held-out loss",
            type="offline",
            entrypoint="first",
            config={"score": 0.25},
        ),
        ValidatorConfig(
            name="rollout tracking",
            type="simulation",
            entrypoint="second",
            config={"score": 0.75},
        ),
    ]

    results = ValidatorRunner.run(
        configs,
        model=object(),
        validation_dataset=object(),
    )

    assert results == [
        ValidationResult(
            name="held-out loss",
            type="offline",
            metrics={"score": 0.25},
        ),
        ValidationResult(
            name="rollout tracking",
            type="simulation",
            metrics={"score": 0.75},
        ),
    ]


def test_run_rejects_non_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Require every validator to return the common result contract."""

    _register(monkeypatch, "invalid-result", RecordingValidator)
    monkeypatch.setattr(
        RecordingValidator,
        "validate",
        lambda self, model, validation_dataset: object(),
    )
    config = ValidatorConfig(
        name="invalid result",
        type="offline",
        entrypoint="invalid-result",
        config={},
    )

    with pytest.raises(TypeError, match="must return a ValidationResult"):
        ValidatorRunner.run(
            [config],
            model=object(),
            validation_dataset=object(),
        )


def test_run_rejects_mismatched_result_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep collected result metadata aligned with its configured validator."""

    _register(monkeypatch, "mismatch", RecordingValidator)
    monkeypatch.setattr(
        RecordingValidator,
        "validate",
        lambda self, model, validation_dataset: ValidationResult(
            name="another validator",
            type=self.type,
            metrics={},
        ),
    )
    config = ValidatorConfig(
        name="configured validator",
        type="offline",
        entrypoint="mismatch",
        config={},
    )

    with pytest.raises(ValueError, match="returned result metadata"):
        ValidatorRunner.run(
            [config],
            model=object(),
            validation_dataset=object(),
        )
