"""Runtime and package-entry-point registry for MLOps backends."""

from __future__ import annotations

from importlib import metadata
from typing import Any, Callable, Dict, List, cast

from anchor_bench.errors import (
    MLOpsBackendNotFoundError,
    MLOpsError,
    MLOpsPluginError,
)
from anchor_bench.mlops.base import MLOpsBackend

DEFAULT_BACKEND = "mlflow"
ENTRY_POINT_GROUP = "anchor_bench.mlops_backends"
BackendFactory = Callable[..., MLOpsBackend]

_BACKENDS: Dict[str, BackendFactory] = {}


def register_backend(
    name: str,
    factory: BackendFactory,
    *,
    replace: bool = False,
) -> None:
    """Register a backend factory for this Python process."""

    normalized_name = _normalize_name(name)
    installed_plugins = {
        _normalize_name(entry_point.name) for entry_point in _plugins()
    }
    if (
        normalized_name in _BACKENDS or normalized_name in installed_plugins
    ) and not replace:
        raise ValueError(f"MLOps backend {normalized_name!r} is already registered.")
    if not callable(factory):
        raise TypeError("An MLOps backend factory must be callable.")
    _BACKENDS[normalized_name] = factory


def available_backends() -> List[str]:
    """Return names from the runtime registry and installed plugins."""

    installed_plugins = {
        _normalize_name(entry_point.name) for entry_point in _plugins()
    }
    return sorted(set(_BACKENDS).union(installed_plugins))


def get_backend(name: str = DEFAULT_BACKEND, **options: Any) -> MLOpsBackend:
    """Construct a backend by name, defaulting to MLflow."""

    normalized_name = _normalize_name(name)
    factory = _BACKENDS.get(normalized_name)
    if factory is None:
        factory = _load_plugin(normalized_name)
    if factory is None:
        known = ", ".join(available_backends()) or "none"
        raise MLOpsBackendNotFoundError(
            f"Unknown MLOps backend {normalized_name!r}. Available backends: {known}."
        )

    try:
        backend = factory(**options)
    except MLOpsError:
        raise
    except Exception as error:
        raise MLOpsPluginError(
            f"Could not construct MLOps backend {normalized_name!r}: {error}."
        ) from error

    if not isinstance(backend, MLOpsBackend):
        raise MLOpsPluginError(
            f"MLOps backend {normalized_name!r} returned {type(backend).__name__}; "
            "plugins must return an MLOpsBackend instance."
        )
    return backend


def _load_plugin(name: str) -> BackendFactory | None:
    matches = [
        entry_point
        for entry_point in _plugins()
        if _normalize_name(entry_point.name) == name
    ]
    if not matches:
        return None
    if len(matches) > 1:
        raise MLOpsPluginError(
            f"Multiple MLOps plugins are installed with the name {name!r}."
        )

    try:
        loaded_factory = matches[0].load()
    except Exception as error:
        raise MLOpsPluginError(
            f"Could not load MLOps plugin {name!r}: {error}."
        ) from error
    if not callable(loaded_factory):
        raise MLOpsPluginError(
            f"MLOps plugin {name!r} must expose a callable backend factory."
        )
    factory = cast(BackendFactory, loaded_factory)
    _BACKENDS[name] = factory
    return factory


def _plugins() -> List[metadata.EntryPoint]:
    return list(metadata.entry_points(group=ENTRY_POINT_GROUP))


def _normalize_name(name: str) -> str:
    normalized_name = name.strip().lower()
    if not normalized_name:
        raise ValueError("An MLOps backend name cannot be empty.")
    return normalized_name
