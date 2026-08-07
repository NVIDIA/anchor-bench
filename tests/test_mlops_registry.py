from contextlib import AbstractContextManager
from typing import Mapping, Optional

import pytest

from anchor_bench.errors import MLOpsBackendNotFoundError, MLOpsPluginError
from anchor_bench.mlops import (
    DEFAULT_BACKEND,
    MLOpsBackend,
    MLOpsRun,
    available_backends,
    get_backend,
    register_backend,
    registry,
)


class FakeBackend(MLOpsBackend):
    def __init__(self, **options) -> None:
        self.options = options

    def start_run(
        self,
        experiment_name: str,
        *,
        run_name: Optional[str] = None,
        tags: Optional[Mapping[str, str]] = None,
    ) -> AbstractContextManager[MLOpsRun]:
        raise NotImplementedError


def test_bundled_backends_are_registered_as_entry_point_plugins():
    plugin_names = {entry_point.name for entry_point in registry._plugins()}

    assert {"mlflow", "wandb"} <= plugin_names


def test_mlflow_is_the_default_backend(monkeypatch):
    class FakeEntryPoint:
        name = "mlflow"

        def __init__(self) -> None:
            self.load_count = 0

        def load(self):
            self.load_count += 1
            return FakeBackend

    entry_point = FakeEntryPoint()
    monkeypatch.setattr(registry, "_BACKENDS", {})
    monkeypatch.setattr(registry, "_plugins", lambda: [entry_point])

    backend = get_backend(tracking_uri="sqlite:///runs.db")

    assert DEFAULT_BACKEND == "mlflow"
    assert isinstance(backend, FakeBackend)
    assert backend.options == {"tracking_uri": "sqlite:///runs.db"}
    assert entry_point.load_count == 1


def test_runtime_backend_registration(monkeypatch):
    monkeypatch.delitem(registry._BACKENDS, "test-runtime", raising=False)
    register_backend("test-runtime", FakeBackend)

    backend = get_backend("TEST-RUNTIME", mode="offline")

    assert isinstance(backend, FakeBackend)
    assert backend.options == {"mode": "offline"}
    assert "test-runtime" in available_backends()


def test_installed_entry_point_is_loaded_and_cached(monkeypatch):
    class FakeEntryPoint:
        name = "test-entry-point"

        def __init__(self) -> None:
            self.load_count = 0

        def load(self):
            self.load_count += 1
            return FakeBackend

    entry_point = FakeEntryPoint()
    monkeypatch.delitem(registry._BACKENDS, entry_point.name, raising=False)
    monkeypatch.setattr(registry, "_plugins", lambda: [entry_point])

    first = get_backend(entry_point.name, entity="team")
    second = get_backend(entry_point.name)

    assert isinstance(first, FakeBackend)
    assert first.options == {"entity": "team"}
    assert isinstance(second, FakeBackend)
    assert entry_point.load_count == 1


def test_unknown_backend_lists_available_backends():
    with pytest.raises(MLOpsBackendNotFoundError, match="Available backends"):
        get_backend("does-not-exist")


def test_plugin_must_return_backend_instance(monkeypatch):
    monkeypatch.setitem(registry._BACKENDS, "invalid-plugin", lambda: object())

    with pytest.raises(MLOpsPluginError, match="must return an MLOpsBackend"):
        get_backend("invalid-plugin")


def test_plugin_construction_failure_is_wrapped(monkeypatch):
    def fail_to_construct():
        raise RuntimeError("broken constructor")

    monkeypatch.setitem(registry._BACKENDS, "broken-plugin", fail_to_construct)

    with pytest.raises(
        MLOpsPluginError,
        match="Could not construct MLOps backend 'broken-plugin'",
    ):
        get_backend("broken-plugin")


def test_duplicate_runtime_backend_is_rejected(monkeypatch):
    monkeypatch.setattr(registry, "_plugins", list)
    monkeypatch.setitem(registry._BACKENDS, "duplicate", FakeBackend)

    with pytest.raises(ValueError, match="already registered"):
        register_backend("duplicate", FakeBackend)


def test_installed_plugin_cannot_be_shadowed_unintentionally(monkeypatch):
    class FakeEntryPoint:
        name = "installed-plugin"

    monkeypatch.delitem(registry._BACKENDS, "installed-plugin", raising=False)
    monkeypatch.setattr(registry, "_plugins", lambda: [FakeEntryPoint()])

    with pytest.raises(ValueError, match="already registered"):
        register_backend("installed-plugin", FakeBackend)
