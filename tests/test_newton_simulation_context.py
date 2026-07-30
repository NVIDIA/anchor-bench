from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar, Optional

import pytest

from anchor_bench.core import NewtonSimulationContext


class FakeModel:
    """Record state creation for a fake Newton model."""

    def __init__(self) -> None:
        self.initial_state = object()

    def state(self):
        """Return the model's initial state."""

        return self.initial_state


class FakeModelBuilder:
    """Record calls made through Newton's model builder API."""

    last_instance: ClassVar[Optional["FakeModelBuilder"]] = None

    def __init__(self) -> None:
        self.usd_source = None
        self.model = FakeModel()
        FakeModelBuilder.last_instance = self

    def add_usd(self, *, source: str) -> None:
        """Record the USD source passed to Newton."""

        self.usd_source = source

    def finalize(self) -> FakeModel:
        """Return the finalized fake model."""

        return self.model


def test_loads_newton_model_and_initial_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "robot.usda"
    source.write_text("#usda 1.0", encoding="utf-8")
    fake_newton = SimpleNamespace(ModelBuilder=FakeModelBuilder)
    monkeypatch.setattr(
        "anchor_bench.core.simulation.import_module",
        lambda name: fake_newton,
    )

    model, state = NewtonSimulationContext().load(str(source))

    builder = FakeModelBuilder.last_instance
    assert builder is not None
    assert builder.usd_source == str(source)
    assert model is builder.model
    assert state is builder.model.initial_state


def test_rejects_missing_usd_file(tmp_path: Path) -> None:
    source = tmp_path / "missing.usda"

    with pytest.raises(FileNotFoundError, match="USD file does not exist"):
        NewtonSimulationContext().load(str(source))
