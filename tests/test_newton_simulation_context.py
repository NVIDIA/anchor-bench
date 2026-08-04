from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, Optional, Tuple

import pytest

from anchor_bench.core import NewtonSimulationContext, SimulationContext


class FakeModel:
    """Record state creation for a fake Newton model."""

    def __init__(self) -> None:
        self.initial_state = object()

    def state(self):
        """Return the model initial state."""

        return self.initial_state


class FakeModelBuilder:
    """Record calls made through Newton model builder API."""

    last_instance: ClassVar[Optional["FakeModelBuilder"]] = None
    instance_count: ClassVar[int] = 0

    def __init__(self) -> None:
        self.usd_source = None
        self.model = FakeModel()
        FakeModelBuilder.last_instance = self
        FakeModelBuilder.instance_count += 1

    def add_usd(self, *, source: str) -> None:
        """Record the USD source passed to Newton."""

        self.usd_source = source

    def finalize(self) -> FakeModel:
        """Return the finalized fake model."""

        return self.model


class OtherSimulationContext(SimulationContext):
    """Represent a second backend for singleton enforcement tests."""

    def _load(self, usd_path: Path) -> Tuple[Any, Any]:
        """Return placeholder backend objects."""

        return object(), object()


@pytest.fixture(autouse=True)
def reset_simulation_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate the process-wide context between tests."""

    monkeypatch.setattr(
        SimulationContext,
        "_instance",
        None,
    )


def test_lazily_loads_one_newton_model_and_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "robot.usda"
    source.write_text("#usda 1.0", encoding="utf-8")
    fake_newton = SimpleNamespace(ModelBuilder=FakeModelBuilder)
    imported = []

    def import_fake_newton(name: str):
        """Record the delayed Newton import."""

        imported.append(name)
        return fake_newton

    monkeypatch.setattr(
        "anchor_bench.core.simulation.newton.import_module",
        import_fake_newton,
    )
    FakeModelBuilder.instance_count = 0

    context = NewtonSimulationContext(str(source))
    same_context = NewtonSimulationContext(str(source))

    assert context is same_context
    assert SimulationContext.current() is context
    assert imported == []

    model = context.model
    state = context.state
    loaded_model, loaded_state = context.load()

    builder = FakeModelBuilder.last_instance
    assert builder is not None
    assert builder.usd_source == str(source)
    assert model is builder.model
    assert state is builder.model.initial_state
    assert loaded_model is model
    assert loaded_state is state
    assert imported == ["newton"]
    assert FakeModelBuilder.instance_count == 1


def test_rejects_access_before_context_creation() -> None:
    """Require one concrete context to establish the singleton."""

    with pytest.raises(RuntimeError, match="has not been created"):
        SimulationContext.current()


def test_rejects_missing_usd_file(tmp_path: Path) -> None:
    source = tmp_path / "missing.usda"

    with pytest.raises(FileNotFoundError, match="USD file does not exist"):
        NewtonSimulationContext(str(source))


def test_rejects_a_different_usd_file(tmp_path: Path) -> None:
    """Keep one USD assigned for the process lifetime."""

    first = tmp_path / "first.usda"
    second = tmp_path / "second.usda"
    first.write_text("#usda 1.0", encoding="utf-8")
    second.write_text("#usda 1.0", encoding="utf-8")

    NewtonSimulationContext(str(first))

    with pytest.raises(RuntimeError, match="already uses"):
        NewtonSimulationContext(str(second))


def test_rejects_a_different_backend(tmp_path: Path) -> None:
    """Keep one simulation backend assigned for the process lifetime."""

    source = tmp_path / "robot.usda"
    source.write_text("#usda 1.0", encoding="utf-8")

    NewtonSimulationContext(str(source))

    with pytest.raises(RuntimeError, match="Only one simulation backend"):
        OtherSimulationContext(str(source))
