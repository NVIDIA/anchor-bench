# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Solver adapters used by the Newton simulation context."""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Mapping
from typing import Any, ClassVar

import newton
from pydantic import JsonValue

NOT_SET_PREFIX = "NOT SET: "

_KEYWORD_KINDS = (
    inspect.Parameter.POSITIONAL_OR_KEYWORD,
    inspect.Parameter.KEYWORD_ONLY,
)


class NewtonSolverAdapter:
    """Build and drive one Newton solver type inside the simulation context.

    Subclasses set ``solver_class`` and override only the lifecycle steps in
    which their solver differs from these defaults.
    """

    solver_class: ClassVar[type[newton.solvers.SolverBase]]
    # Constructor arguments the adapter supplies itself.
    managed_arguments: ClassVar[frozenset[str]] = frozenset({"model"})

    def register_attributes(self, builder: newton.ModelBuilder) -> None:
        """Declare the solver's custom attributes so USD can author them."""
        self.solver_class.register_custom_attributes(builder)

    def prepare_builder(self, builder: newton.ModelBuilder) -> None:
        """Adjust the replicated builder before it is finalized."""

    def create_solver(
        self, model: newton.Model, settings: Mapping[str, JsonValue]
    ) -> tuple[newton.solvers.SolverBase, dict[str, JsonValue]]:
        """Construct the solver and return it with the recorded settings."""
        kwargs, recorded = self.match_settings(settings)
        return self.solver_class(model, **kwargs), recorded

    def step_contacts(self, model: newton.Model) -> newton.Contacts | None:
        """Return the contacts passed to every solver step."""
        return newton.CollisionPipeline(model, broad_phase="explicit").contacts()

    def match_settings(
        self, settings: Mapping[str, JsonValue]
    ) -> tuple[dict[str, Any], dict[str, JsonValue]]:
        """Split settings into constructor arguments and a provenance record."""
        parameters = inspect.signature(self.solver_class).parameters
        accepted = {
            name
            for name, parameter in parameters.items()
            if parameter.kind in _KEYWORD_KINDS
        } - self.managed_arguments
        kwargs: dict[str, Any] = {}
        recorded: dict[str, JsonValue] = {}
        for name, value in settings.items():
            if name in accepted:
                kwargs[name] = value
                recorded[name] = value
                continue
            reason = f"not a {self.solver_class.__name__} parameter"
            warnings.warn(
                f"simulation.solver.settings.{name}: {reason}; it was not set",
                UserWarning,
                stacklevel=2,
            )
            recorded[name] = f"{NOT_SET_PREFIX}{reason}"
        return kwargs, recorded


class MuJoCoAdapter(NewtonSolverAdapter):
    solver_class = newton.solvers.SolverMuJoCo


class VBDAdapter(NewtonSolverAdapter):
    solver_class = newton.solvers.SolverVBD
    managed_arguments = NewtonSolverAdapter.managed_arguments | {"collision_pipeline"}

    def prepare_builder(self, builder: newton.ModelBuilder) -> None:
        # VBD solves bodies in parallel by graph color.
        builder.color()

    def create_solver(
        self, model: newton.Model, settings: Mapping[str, JsonValue]
    ) -> tuple[newton.solvers.SolverBase, dict[str, JsonValue]]:
        kwargs, recorded = self.match_settings(settings)
        # VBD runs collision detection itself; its contact history needs matching.
        pipeline = newton.CollisionPipeline(
            model, broad_phase="explicit", contact_matching="latest"
        )
        return self.solver_class(model, collision_pipeline=pipeline, **kwargs), recorded

    def step_contacts(self, model: newton.Model) -> newton.Contacts | None:
        return None


NEWTON_SOLVERS: dict[str, type[NewtonSolverAdapter]] = {
    "warp_mujoco": MuJoCoAdapter,
    "vbd": VBDAdapter,
}


def get_solver_adapter(solver_type: str) -> NewtonSolverAdapter:
    """Return a fresh adapter for a configured Newton solver type."""
    try:
        return NEWTON_SOLVERS[solver_type]()
    except KeyError as error:
        available = ", ".join(sorted(NEWTON_SOLVERS))
        raise ValueError(
            f"Unknown Newton solver {solver_type!r}; available: {available}"
        ) from error
