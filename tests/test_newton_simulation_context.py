from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, Tuple

import pytest

from anchor_bench.core import NewtonSimulationContext, SimulationContext


class FakeModel:
    """Record simulation buffer creation for a fake Newton model."""

    def __init__(self) -> None:
        self.states = []
        self.control_buffer = object()
        self.contacts_buffer = object()

    def state(self):
        """Return a new fake state."""

        state = FakeState()
        self.states.append(state)
        return state

    def control(self):
        """Return the model control buffer."""

        return self.control_buffer

    def contacts(self):
        """Return the model contacts buffer."""

        return self.contacts_buffer


class FakeState:
    """Record force clearing for a fake Newton state."""

    def __init__(self) -> None:
        self.clear_count = 0
        self.assign_count = 0

    def clear_forces(self) -> None:
        self.clear_count += 1

    def assign(self, other: "FakeState") -> None:
        self.assign_count += 1


class FakeModelBuilder:
    """Record calls made through Newton model builder API."""

    instances: ClassVar[list["FakeModelBuilder"]] = []
    instance_count: ClassVar[int] = 0

    def __init__(self) -> None:
        self.usd_source = None
        self.replicated_template = None
        self.world_count = 1
        self.world_gravity = []
        self.floating = None
        self.options = {}
        self.model = FakeModel()
        FakeModelBuilder.instances.append(self)
        FakeModelBuilder.instance_count += 1

    def add_usd(
        self,
        *,
        source: str,
        floating: bool,
        **options: object,
    ) -> None:
        """Record the USD source passed to Newton."""

        self.usd_source = source
        self.floating = floating
        self.options = options

    def finalize(self) -> FakeModel:
        """Return the finalized fake model."""

        return self.model

    def replicate(
        self,
        template: "FakeModelBuilder",
        *,
        world_count: int,
    ) -> None:
        """Record a replicated model request."""

        self.replicated_template = template
        self.world_count = world_count


class FakeSolver:
    """Record solver registration, construction, and steps."""

    registered_builders: ClassVar[list[FakeModelBuilder]] = []
    instances: ClassVar[list["FakeSolver"]] = []

    @classmethod
    def register_custom_attributes(cls, builder: FakeModelBuilder) -> None:
        cls.registered_builders.append(builder)

    def __init__(self, model: FakeModel, **options: Any) -> None:
        self.model = model
        self.options = options
        self.steps = []
        self.instances.append(self)

    def step(self, **values: Any) -> None:
        self.steps.append(values)


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


def _install_fake_newton(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install and reset the fake Newton construction API."""

    fake_newton = SimpleNamespace(
        ModelBuilder=FakeModelBuilder,
        solvers=SimpleNamespace(SolverMuJoCo=FakeSolver),
    )
    monkeypatch.setattr(
        "anchor_bench.core.simulation.newton.newton",
        fake_newton,
    )
    FakeModelBuilder.instance_count = 0
    FakeModelBuilder.instances = []
    FakeSolver.registered_builders = []
    FakeSolver.instances = []


@pytest.fixture
def fake_newton_usd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Create a USD asset and install the fake Newton construction API."""

    source = tmp_path / "robot.usda"
    source.write_text("#usda 1.0", encoding="utf-8")
    _install_fake_newton(monkeypatch)
    return source


def test_lazily_builds_one_newton_model_and_state(
    fake_newton_usd: Path,
) -> None:
    source = fake_newton_usd

    context = NewtonSimulationContext(str(source))
    same_context = NewtonSimulationContext(str(source))

    assert context is same_context
    assert SimulationContext.current() is context
    assert FakeModelBuilder.instance_count == 0

    model = context.model
    state = context.state
    loaded_model, loaded_state = context.load()

    template, population = FakeModelBuilder.instances
    fake_model = population.model
    assert template.usd_source == str(source)
    assert template.floating is True
    assert template.options == {}
    assert template.world_gravity == []
    assert population.replicated_template is template
    assert population.world_count == 1
    assert model is fake_model
    assert state is fake_model.states[0]
    assert loaded_model is model
    assert loaded_state is state
    assert context.state is fake_model.states[0]
    assert context.state_next is fake_model.states[1]
    assert context.control is fake_model.control_buffer
    assert context.contacts is fake_model.contacts_buffer
    assert context.solver is FakeSolver.instances[0]
    assert context.world_count == 1
    assert FakeModelBuilder.instance_count == 2


def test_initializes_one_configurable_replicated_simulation(
    fake_newton_usd: Path,
) -> None:
    """Keep USD loading and population replication in the shared context."""

    source = fake_newton_usd
    configured = []
    configure = configured.append

    context = NewtonSimulationContext(str(source))
    context.initialize(
        world_count=4,
        configure_builder=configure,
        configuration_key=("actuators", 1),
        solver_options={"iterations": 5},
    )
    context.initialize(
        world_count=4,
        configure_builder=configure,
        configuration_key=("actuators", 1),
        solver_options={"iterations": 5},
    )

    template, population = FakeModelBuilder.instances
    assert FakeSolver.registered_builders == [template, population]
    assert template.usd_source == str(source)
    assert configured == [template]
    assert population.replicated_template is template
    assert population.world_count == 4
    assert context.model is population.model
    fake_solver = FakeSolver.instances[0]
    assert context.solver is fake_solver
    assert fake_solver.model is population.model
    assert fake_solver.options == {"iterations": 5}
    assert FakeModelBuilder.instance_count == 2
    assert len(FakeSolver.instances) == 1


def test_rejects_conflicting_reinitialization(
    fake_newton_usd: Path,
) -> None:
    source = fake_newton_usd
    configure = lambda builder: None
    context = NewtonSimulationContext(str(source))
    context.initialize(
        world_count=4,
        configure_builder=configure,
        configuration_key=("actuators", 1),
        solver_options={"iterations": 5},
    )

    with pytest.raises(RuntimeError, match="different world count or configuration"):
        context.initialize(
            world_count=8,
            configure_builder=configure,
            configuration_key=("actuators", 1),
            solver_options={"iterations": 5},
        )
    context.initialize(
        world_count=4,
        configure_builder=lambda builder: None,
        configuration_key=("actuators", 1),
        solver_options={"iterations": 5},
    )
    with pytest.raises(RuntimeError, match="different world count or configuration"):
        context.initialize(
            world_count=4,
            configure_builder=configure,
            configuration_key=("actuators", 2),
            solver_options={"iterations": 5},
        )
    with pytest.raises(RuntimeError, match="different world count or configuration"):
        context.initialize(
            world_count=4,
            configure_builder=configure,
            configuration_key=("actuators", 1),
            solver_options={"iterations": 6},
        )


def test_rejects_invalid_world_count(
    fake_newton_usd: Path,
) -> None:
    context = NewtonSimulationContext(str(fake_newton_usd))

    with pytest.raises(ValueError, match="positive integer"):
        context.initialize(world_count=0)


def test_requires_a_stable_key_for_builder_configuration(
    fake_newton_usd: Path,
) -> None:
    context = NewtonSimulationContext(str(fake_newton_usd))

    with pytest.raises(ValueError, match="configuration_key is required"):
        context.initialize(configure_builder=lambda builder: None)


def test_rejects_a_key_without_builder_configuration(
    fake_newton_usd: Path,
) -> None:
    context = NewtonSimulationContext(str(fake_newton_usd))
    configured = []
    configuration_key = ("actuators", 1)

    with pytest.raises(ValueError, match="configure_builder is required"):
        context.initialize(configuration_key=configuration_key)

    context.initialize(
        configure_builder=configured.append,
        configuration_key=configuration_key,
    )

    assert configured == [FakeModelBuilder.instances[0]]


def test_context_steps_with_stable_owned_states(
    fake_newton_usd: Path,
) -> None:
    context = NewtonSimulationContext(str(fake_newton_usd))
    context.initialize()
    fake_model = FakeModelBuilder.instances[1].model
    state, state_next = fake_model.states
    assert context.state is state
    assert context.state_next is state_next

    context.step(dt=0.01)

    assert state.clear_count == 1
    assert state.assign_count == 1
    assert context.state is state
    assert context.state_next is state_next
    assert context.state is state
    assert FakeSolver.instances[0].steps == [
        {
            "state_in": state,
            "state_out": state_next,
            "control": context.control,
            "contacts": context.contacts,
            "dt": 0.01,
        }
    ]

    with pytest.raises(ValueError, match="finite and positive"):
        context.step(dt=float("nan"))


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


def test_rejects_different_import_configuration(tmp_path: Path) -> None:
    """Keep the fixed-base setting process-wide."""

    source = tmp_path / "robot.usda"
    source.write_text("#usda 1.0", encoding="utf-8")
    NewtonSimulationContext(str(source), fixed_base=True)

    with pytest.raises(RuntimeError, match="different base mode"):
        NewtonSimulationContext(str(source), fixed_base=False)
