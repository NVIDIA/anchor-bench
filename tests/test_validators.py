# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest

import anchor_bench.train.validators as validator_registry
from anchor_bench.core import SimulationContext
from anchor_bench.train import BaseValidator, ValidationResult


class RecordingValidator(BaseValidator):
    """Provide an inert concrete validator for scaffold tests."""

    def validate(self, model: object, validation_dataset: object) -> ValidationResult:
        """Return an inert structured validation result."""

        return ValidationResult(
            name=self.name,
            type=self.type,
            metrics={},
        )


def test_base_validator_stores_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retain the schema fields needed when validation is integrated later."""

    source_config = {"split": "test"}
    simulation_context = object()
    monkeypatch.setattr(SimulationContext, "current", lambda: simulation_context)

    validator = RecordingValidator(
        name="held-out loss",
        type="offline",
        config=source_config,
    )

    assert validator.name == "held-out loss"
    assert validator.type == "offline"
    assert validator.config == source_config
    assert validator.config is not source_config
    assert validator.simulation_context is simulation_context


def test_base_validator_cannot_be_instantiated() -> None:
    """Keep validation behavior mandatory for concrete implementations."""

    with pytest.raises(TypeError):
        BaseValidator(  # pyright: ignore[reportAbstractUsage]
            name="abstract",
            type="offline",
            config={},
        )


def test_registry_imports_validator_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    """Delay importing a validator module until its registry key is resolved."""

    imports: list[tuple[str, str]] = []

    def import_validator(module_name: str, package: str) -> object:
        imports.append((module_name, package))
        return SimpleNamespace(RecordingValidator=RecordingValidator)

    monkeypatch.setitem(
        validator_registry.VALIDATORS,
        "recording",
        (".recording", "RecordingValidator"),
    )
    monkeypatch.setattr(validator_registry, "import_module", import_validator)

    assert imports == []
    assert validator_registry.recording is RecordingValidator
    assert imports == [(".recording", validator_registry.__name__)]


def test_registry_rejects_unknown_validator() -> None:
    """Reject keys outside the explicit built-in registry."""

    with pytest.raises(AttributeError, match="Unknown validator 'missing'"):
        _ = validator_registry.missing
