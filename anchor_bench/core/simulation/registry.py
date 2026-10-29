# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Simulation contexts and their optional system-identification callbacks."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
from importlib.metadata import entry_points
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

from anchor_bench.core.simulation import SimulationContext
from anchor_bench.core.simulation.newton import NewtonSimulationContext
from anchor_bench.errors import SimulationBackendError

if TYPE_CHECKING:
    from anchor_bench.schema.actuator_sys_id import ActuatorSysIdConfig
    from anchor_bench.schema.simulation import JsonValue, SysIdSimulationConfig
    from anchor_bench.tasks.actuator_sys_id.model import (
        Candidate,
        IdentifiedModelArtifact,
    )
    from anchor_bench.tasks.actuator_sys_id.objective import PopulationEvaluator
    from anchor_bench.tasks.actuator_sys_id.parameter_space import ParameterSpace


class _SysIdCallbacks(TypedDict):
    config_model: type[SysIdSimulationConfig]
    create_context: Callable[[ActuatorSysIdConfig], SimulationContext[Any, Any]]
    create_evaluator: Callable[
        [ActuatorSysIdConfig, ParameterSpace, SimulationContext[Any, Any]],
        PopulationEvaluator,
    ]
    export_model: (
        Callable[
            [ActuatorSysIdConfig, Candidate, SimulationContext[Any, Any], Path],
            IdentifiedModelArtifact | None,
        ]
        | None
    )
    describe_runtime: Callable[[PopulationEvaluator], Mapping[str, JsonValue]] | None


SIMULATION_CONTEXTS: dict[str, type[SimulationContext[Any, Any]]] = {
    "newton": NewtonSimulationContext,
}
_SYSID_CALLBACKS: dict[str, _SysIdCallbacks] = {}
_DISCOVERED = False
_DISCOVERING = False


def _ensure_builtin() -> None:
    if "newton" in _SYSID_CALLBACKS:
        return
    from anchor_bench.schema.simulation import NewtonSimulationConfig
    from anchor_bench.tasks.actuator_sys_id.backends import newton

    _SYSID_CALLBACKS["newton"] = {
        "config_model": NewtonSimulationConfig,
        "create_context": newton.create_context,
        "create_evaluator": newton.create_evaluator,
        "export_model": newton.export_model,
        "describe_runtime": newton.describe_runtime,
    }


def register_simulation_backend(
    name: str,
    context_type: type[SimulationContext[Any, Any]],
    *,
    config_model: type[SysIdSimulationConfig] | None = None,
    create_context: Callable[[ActuatorSysIdConfig], SimulationContext[Any, Any]]
    | None = None,
    create_evaluator: Callable[
        [ActuatorSysIdConfig, ParameterSpace, SimulationContext[Any, Any]],
        PopulationEvaluator,
    ]
    | None = None,
    export_model: Callable[
        [ActuatorSysIdConfig, Candidate, SimulationContext[Any, Any], Path],
        IdentifiedModelArtifact | None,
    ]
    | None = None,
    describe_runtime: Callable[[PopulationEvaluator], Mapping[str, JsonValue]]
    | None = None,
) -> None:
    """Register a context class, optionally with the callbacks needed for sysID."""
    from anchor_bench.schema.simulation import SysIdSimulationConfig

    if not isinstance(name, str) or not name.strip():
        raise SimulationBackendError(
            "Simulation backend name must be a non-empty string."
        )
    if name in SIMULATION_CONTEXTS:
        raise SimulationBackendError(
            f"Simulation backend {name!r} is already registered."
        )
    if not isinstance(context_type, type) or not issubclass(
        context_type, SimulationContext
    ):
        raise SimulationBackendError(
            "Backend context_type must extend SimulationContext."
        )
    callbacks = None
    if any(
        value is not None
        for value in (
            config_model,
            create_context,
            create_evaluator,
            export_model,
            describe_runtime,
        )
    ):
        if config_model is None or create_context is None or create_evaluator is None:
            raise SimulationBackendError(
                "SysID registration requires config_model, create_context, and create_evaluator."
            )
        if not isinstance(config_model, type) or not issubclass(
            config_model, SysIdSimulationConfig
        ):
            raise SimulationBackendError(
                "Backend config_model must extend SysIdSimulationConfig."
            )
        for label, factory in (
            ("create_context", create_context),
            ("create_evaluator", create_evaluator),
            ("export_model", export_model),
            ("describe_runtime", describe_runtime),
        ):
            if factory is None and label in ("export_model", "describe_runtime"):
                continue
            if not callable(factory):
                raise SimulationBackendError(f"Backend {label} must be callable.")
        callbacks = _SysIdCallbacks(
            config_model=config_model,
            create_context=create_context,
            create_evaluator=create_evaluator,
            export_model=export_model,
            describe_runtime=describe_runtime,
        )
    SIMULATION_CONTEXTS[name] = context_type
    if callbacks is not None:
        _SYSID_CALLBACKS[name] = callbacks


def discover_sysid_backends() -> None:
    """Run installed registration functions, rolling back a failed discovery."""
    global _DISCOVERED, _DISCOVERING
    if _DISCOVERED:
        return
    if _DISCOVERING:
        raise SimulationBackendError(
            "Recursive simulation backend discovery is not supported."
        )
    _ensure_builtin()
    contexts_before = SIMULATION_CONTEXTS.copy()
    callbacks_before = _SYSID_CALLBACKS.copy()
    _DISCOVERING = True
    try:
        entries = sorted(
            entry_points(group="anchor_bench.sysid_backends"),
            key=lambda entry: entry.name,
        )
        duplicates = sorted(
            name
            for name, count in Counter(entry.name for entry in entries).items()
            if count > 1
        )
        if duplicates:
            raise SimulationBackendError(
                f"Duplicate simulation backend entry-point names: {', '.join(duplicates)}."
            )
        for entry in entries:
            if entry.name in SIMULATION_CONTEXTS:
                raise SimulationBackendError(
                    f"Simulation backend {entry.name!r} is already registered."
                )
        for entry in entries:
            names_before = set(SIMULATION_CONTEXTS)
            try:
                provider = entry.load()
                if not callable(provider):
                    raise SimulationBackendError(
                        "Entry-point provider must be callable."
                    )
                if provider() is not None:
                    raise SimulationBackendError(
                        "Entry-point provider must register its context and return None."
                    )
                if set(SIMULATION_CONTEXTS) - names_before != {entry.name}:
                    raise SimulationBackendError(
                        f"Entry-point provider must register exactly its name {entry.name!r}."
                    )
            except Exception as error:
                raise SimulationBackendError(
                    f"Could not load simulation backend {entry.name!r}: {error}"
                ) from error
        _DISCOVERED = True
    except Exception as error:
        SIMULATION_CONTEXTS.clear()
        SIMULATION_CONTEXTS.update(contexts_before)
        _SYSID_CALLBACKS.clear()
        _SYSID_CALLBACKS.update(callbacks_before)
        if isinstance(error, SimulationBackendError):
            raise
        raise SimulationBackendError(
            f"Could not discover simulation backends: {error}"
        ) from error
    finally:
        _DISCOVERING = False


def get_sysid_callbacks(name: str) -> _SysIdCallbacks:
    """Look up the sysID callbacks attached to a registered context."""
    discover_sysid_backends()
    if name not in SIMULATION_CONTEXTS:
        raise SimulationBackendError(f"Unknown simulation backend {name!r}.")
    if name not in _SYSID_CALLBACKS:
        raise SimulationBackendError(
            f"Simulation backend {name!r} does not support system identification."
        )
    return _SYSID_CALLBACKS[name]


def create_sysid_context(config: ActuatorSysIdConfig) -> SimulationContext[Any, Any]:
    """Create and verify the registered context using its sysID factory."""
    callbacks = get_sysid_callbacks(config.simulation.type)
    context = callbacks["create_context"](config)
    expected = SIMULATION_CONTEXTS[config.simulation.type]
    if type(context) is not expected:
        raise SimulationBackendError(
            f"Backend {config.simulation.type!r} returned context type {type(context).__name__}; expected {expected.__name__}."
        )
    return context


def create_simulation_context(
    backend: str,
    usd_path: str,
    *,
    fixed_base: bool = False,
) -> SimulationContext[Any, Any]:
    """Create the context registered for a configured simulator."""
    try:
        context_type = SIMULATION_CONTEXTS[backend]
    except KeyError as error:
        raise ValueError(f"Unknown simulation backend {backend!r}.") from error
    return context_type(usd_path, fixed_base=fixed_base)
