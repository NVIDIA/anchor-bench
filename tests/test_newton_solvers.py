# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, ClassVar

import newton
import pytest

from anchor_bench.core.simulation import newton_solvers
from anchor_bench.core.simulation.newton_solvers import (
    NEWTON_SOLVERS,
    MuJoCoAdapter,
    NewtonSolverAdapter,
    VBDAdapter,
    get_solver_adapter,
)


class FakeSolver:
    registered: ClassVar[list[object]] = []

    def __init__(self, model, *, iterations: int = 1, integrator: str = "euler"):
        self.model = model
        self.iterations = iterations
        self.integrator = integrator

    @classmethod
    def register_custom_attributes(cls, builder) -> None:
        cls.registered.append(builder)


# The fakes stand in for Newton objects, so they are passed untyped.
MODEL: Any = "model"
BUILDER: Any = "builder"


class FakeAdapter(NewtonSolverAdapter):
    solver_class = FakeSolver  # type: ignore[assignment]


def test_matching_settings_reach_the_constructor():
    solver, recorded = FakeAdapter().create_solver(
        MODEL, {"iterations": 7, "integrator": "implicitfast"}
    )

    assert isinstance(solver, FakeSolver)
    assert (solver.model, solver.iterations, solver.integrator) == (
        "model",
        7,
        "implicitfast",
    )
    assert recorded == {"iterations": 7, "integrator": "implicitfast"}


def test_unknown_setting_warns_and_is_recorded_as_not_set():
    with pytest.warns(UserWarning, match="simulation.solver.settings.itertions"):
        solver, recorded = FakeAdapter().create_solver(
            MODEL, {"itertions": 20, "iterations": 3}
        )

    assert isinstance(solver, FakeSolver)
    assert solver.iterations == 3
    assert recorded == {
        "itertions": "NOT SET: not a FakeSolver parameter",
        "iterations": 3,
    }


def test_managed_arguments_are_never_taken_from_settings():
    with pytest.warns(UserWarning, match="settings.model"):
        solver, recorded = FakeAdapter().create_solver(MODEL, {"model": "other"})

    assert isinstance(solver, FakeSolver)
    assert solver.model == "model"
    assert recorded == {"model": "NOT SET: set by the simulation context"}


def test_register_attributes_delegates_to_the_solver_class():
    FakeSolver.registered.clear()
    FakeAdapter().register_attributes(BUILDER)
    assert FakeSolver.registered == ["builder"]


def test_default_step_contacts_come_from_a_shared_pipeline(monkeypatch):
    class FakePipeline:
        def __init__(self, model, *, broad_phase):
            self.args = (model, broad_phase)

        def contacts(self):
            return ("contacts", self.args)

    monkeypatch.setattr(newton_solvers.newton, "CollisionPipeline", FakePipeline)
    assert FakeAdapter().step_contacts(MODEL) == (
        "contacts",
        ("model", "explicit"),
    )


def test_mujoco_is_the_registered_default_solver():
    assert NEWTON_SOLVERS["warp_mujoco"] is MuJoCoAdapter
    assert MuJoCoAdapter.solver_class is newton.solvers.SolverMuJoCo
    assert isinstance(get_solver_adapter("warp_mujoco"), MuJoCoAdapter)


def test_unknown_solver_type_lists_available_solvers():
    with pytest.raises(ValueError, match=r"'featherstone'.*warp_mujoco"):
        get_solver_adapter("featherstone")


def test_vbd_is_registered():
    assert NEWTON_SOLVERS["vbd"] is VBDAdapter
    assert VBDAdapter.solver_class is newton.solvers.SolverVBD


def test_vbd_colors_the_builder_before_finalize():
    calls = []
    builder: Any = type("B", (), {"color": lambda self: calls.append(1)})()
    VBDAdapter().prepare_builder(builder)
    assert calls == [1]


def test_vbd_owns_its_collision_pipeline_and_steps_without_contacts(monkeypatch):
    class FakePipeline:
        def __init__(self, model, *, broad_phase, contact_matching):
            self.args = (model, broad_phase, contact_matching)

    class FakeVBD:
        def __init__(self, model, *, iterations=10, collision_pipeline=None):
            self.model = model
            self.iterations = iterations
            self.collision_pipeline = collision_pipeline

    monkeypatch.setattr(newton_solvers.newton, "CollisionPipeline", FakePipeline)
    monkeypatch.setattr(VBDAdapter, "solver_class", FakeVBD)

    with pytest.warns(UserWarning, match="settings.collision_pipeline"):
        solver, recorded = VBDAdapter().create_solver(
            MODEL, {"iterations": 4, "collision_pipeline": "mine"}
        )

    assert isinstance(solver, FakeVBD)
    assert solver.iterations == 4
    pipeline: Any = solver.collision_pipeline
    assert pipeline.args == ("model", "explicit", "latest")
    assert str(recorded["collision_pipeline"]).startswith("NOT SET: ")
    assert VBDAdapter().step_contacts(MODEL) is None
