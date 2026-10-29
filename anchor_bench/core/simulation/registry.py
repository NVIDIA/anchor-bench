# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Simulation resources and typed system-identification capabilities."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from importlib.metadata import entry_points
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from anchor_bench.core.simulation import SimulationContext
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


@dataclass(frozen=True)
class SysIdBackend:
    """Typed factories and optional model export for an identification backend."""

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
    ) = None
    describe_runtime: (
        Callable[[PopulationEvaluator], Mapping[str, JsonValue]] | None
    ) = None


@dataclass(frozen=True)
class SimulationBackend:
    """One registered resource class and its optional identification capability."""

    name: str
    context_type: type[SimulationContext[Any, Any]]
    sysid: SysIdBackend | None = None


_BACKENDS: dict[str, SimulationBackend] = {}
_DISCOVERED = False
_PENDING: dict[str, SimulationBackend] | None = None


def _newton_create_context(config: ActuatorSysIdConfig) -> SimulationContext[Any, Any]:
    from anchor_bench.tasks.actuator_sys_id.backends.newton import create_context

    return create_context(config)


def _newton_create_evaluator(
    config: ActuatorSysIdConfig,
    space: ParameterSpace,
    context: SimulationContext[Any, Any],
) -> PopulationEvaluator:
    from anchor_bench.tasks.actuator_sys_id.backends.newton import create_evaluator

    return create_evaluator(config, space, context)


def _newton_export_model(
    config: ActuatorSysIdConfig,
    best: Candidate,
    context: SimulationContext[Any, Any],
    output: Path,
) -> IdentifiedModelArtifact | None:
    from anchor_bench.tasks.actuator_sys_id.backends.newton import export_model

    return export_model(config, best, context, output)


def _newton_describe_runtime(evaluator: PopulationEvaluator) -> Mapping[str, JsonValue]:
    from anchor_bench.tasks.actuator_sys_id.backends.newton import describe_runtime

    return describe_runtime(evaluator)


def _ensure_builtin() -> None:
    if "newton" in _BACKENDS:
        return
    from anchor_bench.core.simulation.newton import NewtonSimulationContext
    from anchor_bench.schema.simulation import NewtonSimulationConfig

    _BACKENDS["newton"] = SimulationBackend(
        "newton",
        NewtonSimulationContext,
        SysIdBackend(
            NewtonSimulationConfig,
            _newton_create_context,
            _newton_create_evaluator,
            _newton_export_model,
            _newton_describe_runtime,
        ),
    )


def _validate_registration(registration: object) -> SimulationBackend:
    from anchor_bench.schema.simulation import SysIdSimulationConfig

    if not isinstance(registration, SimulationBackend):
        raise SimulationBackendError(
            "Provider must return a SimulationBackend registration."
        )
    if not isinstance(registration.name, str) or not registration.name.strip():
        raise SimulationBackendError(
            "Simulation backend name must be a non-empty string."
        )
    if not isinstance(registration.context_type, type) or not issubclass(
        registration.context_type, SimulationContext
    ):
        raise SimulationBackendError(
            "Backend context_type must extend SimulationContext."
        )
    capability = registration.sysid
    if capability is None:
        return registration
    if not isinstance(capability, SysIdBackend):
        raise SimulationBackendError("Backend sysid capability must be a SysIdBackend.")
    if not isinstance(capability.config_model, type) or not issubclass(
        capability.config_model, SysIdSimulationConfig
    ):
        raise SimulationBackendError(
            "Backend config_model must extend SysIdSimulationConfig."
        )
    for name in (
        "create_context",
        "create_evaluator",
        "export_model",
        "describe_runtime",
    ):
        factory = getattr(capability, name)
        if (
            factory is None and name in ("export_model", "describe_runtime")
        ) or callable(factory):
            continue
        raise SimulationBackendError(f"Backend {name} must be callable.")
    return registration


def register_simulation_backend(registration: SimulationBackend) -> None:
    """Register a backend once, rejecting invalid capabilities and name collisions."""

    _ensure_builtin()
    _validate_registration(registration)
    target = _BACKENDS if _PENDING is None else _PENDING
    if registration.name in target:
        raise SimulationBackendError(
            f"Simulation backend {registration.name!r} is already registered."
        )
    target[registration.name] = registration


def discover_sysid_backends() -> None:
    """Load sorted entry points atomically, allowing a failed discovery to retry."""

    global _DISCOVERED, _PENDING
    if _DISCOVERED:
        return
    if _PENDING is not None:
        raise SimulationBackendError(
            "Recursive simulation backend discovery is not supported."
        )
    _ensure_builtin()
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
            if entry.name in _BACKENDS:
                raise SimulationBackendError(
                    f"Simulation backend {entry.name!r} is already registered."
                )
        _PENDING = _BACKENDS.copy()
        for entry in entries:
            try:
                provider = entry.load()
                if not callable(provider):
                    raise SimulationBackendError(
                        "Entry-point provider must be callable."
                    )
                registration = _validate_registration(provider())
                if registration.name != entry.name:
                    raise SimulationBackendError(
                        f"Entry-point name {entry.name!r} does not match registration name {registration.name!r}."
                    )
                register_simulation_backend(registration)
            except Exception as error:
                raise SimulationBackendError(
                    f"Could not load simulation backend {entry.name!r}: {error}"
                ) from error
        _BACKENDS.update(_PENDING)
        _DISCOVERED = True
    except SimulationBackendError:
        raise
    except Exception as error:
        raise SimulationBackendError(
            f"Could not discover simulation backends: {error}"
        ) from error
    finally:
        _PENDING = None


def get_sysid_backend(name: str) -> SysIdBackend:
    """Return a discovered backend's identification capability or fail early."""

    discover_sysid_backends()
    registration = _BACKENDS.get(name)
    if registration is None:
        raise SimulationBackendError(f"Unknown simulation backend {name!r}.")
    if registration.sysid is None:
        raise SimulationBackendError(
            f"Simulation backend {name!r} does not support system identification."
        )
    return registration.sysid


def create_sysid_context(config: ActuatorSysIdConfig) -> SimulationContext[Any, Any]:
    """Create and verify the concrete resource promised by a backend registration."""

    backend = get_sysid_backend(config.simulation.type)
    context = backend.create_context(config)
    expected = _BACKENDS[config.simulation.type].context_type
    if type(context) is not expected:
        raise SimulationBackendError(
            f"Backend {config.simulation.type!r} returned context type {type(context).__name__}; expected {expected.__name__}."
        )
    return context


class _SimulationContexts(Mapping[str, type[SimulationContext[Any, Any]]]):
    """Compatibility view over the single canonical backend registry."""

    def __getitem__(self, name: str) -> type[SimulationContext[Any, Any]]:
        _ensure_builtin()
        return _BACKENDS[name].context_type

    def __iter__(self) -> Iterator[str]:
        _ensure_builtin()
        return iter(_BACKENDS)

    def __len__(self) -> int:
        _ensure_builtin()
        return len(_BACKENDS)


SIMULATION_CONTEXTS: Mapping[str, type[SimulationContext[Any, Any]]] = (
    _SimulationContexts()
)


def create_simulation_context(
    backend: str,
    usd_path: str,
    *,
    fixed_base: bool = False,
) -> SimulationContext[Any, Any]:
    """Preserve the training adapter's path-based construction API."""

    try:
        context_type = SIMULATION_CONTEXTS[backend]
    except KeyError as error:
        raise ValueError(f"Unknown simulation backend {backend!r}.") from error
    constructor = cast(Callable[..., SimulationContext[Any, Any]], context_type)
    return constructor(usd_path, fixed_base=fixed_base)
