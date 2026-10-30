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
    assert recorded == {"model": "NOT SET: not a FakeSolver parameter"}


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
