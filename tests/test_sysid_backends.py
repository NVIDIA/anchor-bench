# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed plugin registration and transactional discovery boundaries."""

from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, cast

import pytest
from pydantic import StrictInt, ValidationError

from anchor_bench.core.artifacts import _hash_name
from anchor_bench.core.simulation import SimulationContext, registry
from anchor_bench.errors import SimulationBackendError
from anchor_bench.schema.actuator_sys_id import ActuatorSysIdConfig
from anchor_bench.schema.simulation import SysIdSimulationConfig
from anchor_bench.tasks.actuator_sys_id.model import IdentifiedModelArtifact


class ToyConfig(SysIdSimulationConfig):
    type: Literal["toy"] = "toy"  # pyright: ignore[reportIncompatibleVariableOverride]
    gain: StrictInt


class ToyContext(SimulationContext[object, object]):
    def _load(self) -> tuple[object, object]:
        return object(), object()


def toy_registration(name: str = "toy") -> registry.SimulationBackend:
    return registry.SimulationBackend(
        name=name,
        context_type=ToyContext,
        sysid=registry.SysIdBackend(
            config_model=ToyConfig,
            create_context=lambda config: ToyContext(
                usd_path=config.simulation.usd_path
            ),
            create_evaluator=lambda config, space, context: None,  # type: ignore[arg-type]
        ),
    )


@pytest.fixture(autouse=True)
def isolated_registry(monkeypatch: pytest.MonkeyPatch):
    registry.discover_sysid_backends()
    monkeypatch.setattr(registry, "_BACKENDS", registry._BACKENDS.copy())
    monkeypatch.setattr(registry, "_DISCOVERED", False)
    monkeypatch.setattr(registry, "entry_points", lambda **kwargs: [])
    monkeypatch.setattr(SimulationContext, "_instance", None)


@pytest.fixture
def toy_task_data(tmp_path):
    usd = tmp_path / "scene.usda"
    usd.write_text("#usda 1.0\n")
    return {
        "type": "actuator_sys_id",
        "joint_parameters": [
            {
                "names": ["joint"],
                "controller": {
                    "type": "pd",
                    "clamps": {"effort_limit": 1, "velocity_limit": 1},
                    "stiffness": {"lower": 0, "upper": 1},
                    "damping": {"lower": 0, "upper": 1},
                },
                "parameters": {
                    key: {"lower": 0, "upper": 1}
                    for key in ("dry_friction", "viscous_friction", "armature")
                },
            }
        ],
        "optimizer": {
            "type": "python-cmaes",
            "sigma": 1,
            "max_iterations": 1,
            "epsilon": 0.1,
        },
        "simulation": {
            "type": "toy",
            "usd_path": str(usd),
            "gain": 2,
            "physics_freq": 1000,
            "control_freq": 200,
            "render_freq": 60,
        },
        "dataset": {
            "type": "parquet",
            "source": "toy.parquet",
            "resampling": {"type": "uniform", "dt": 0.005},
        },
    }


def test_plugin_simulation_roundtrips_without_losing_fields(toy_task_data):
    registry.register_simulation_backend(toy_registration())
    config = ActuatorSysIdConfig.model_validate(toy_task_data)
    assert isinstance(config.simulation, ToyConfig)
    assert config.simulation.gain == 2
    assert json.loads(config.model_dump_json())["simulation"]["gain"] == 2
    restored = ActuatorSysIdConfig.model_validate_json(config.model_dump_json())
    assert restored == config
    assert _hash_name(restored) == _hash_name(config)
    changed = deepcopy(toy_task_data)
    changed["simulation"]["gain"] = 3
    assert _hash_name(ActuatorSysIdConfig.model_validate(changed)) != _hash_name(config)


@pytest.mark.parametrize("patch", [{"gain": "2"}, {"extra": 1}, {"control_freq": 300}])
def test_plugin_simulation_is_strict(toy_task_data, patch):
    registry.register_simulation_backend(toy_registration())
    toy_task_data["simulation"].update(patch)
    with pytest.raises(ValidationError):
        ActuatorSysIdConfig.model_validate(toy_task_data)


def test_discovery_is_idempotent(monkeypatch):
    loads = []
    entry = SimpleNamespace(
        name="toy", load=lambda: loads.append("load") or toy_registration
    )
    monkeypatch.setattr(registry, "entry_points", lambda **kwargs: [entry])
    registry.discover_sysid_backends()
    registry.discover_sysid_backends()
    assert loads == ["load"]
    assert registry.get_sysid_backend("toy").config_model is ToyConfig


def test_discovery_error_does_not_leave_partial_registration(monkeypatch):
    first = SimpleNamespace(name="toy", load=lambda: toy_registration)
    second = SimpleNamespace(
        name="zzz", load=lambda: lambda: (_ for _ in ()).throw(RuntimeError("broken"))
    )
    monkeypatch.setattr(registry, "entry_points", lambda **kwargs: [second, first])
    with pytest.raises(SimulationBackendError, match="broken"):
        registry.discover_sysid_backends()
    assert "toy" not in registry.SIMULATION_CONTEXTS
    assert not registry._DISCOVERED
    monkeypatch.setattr(registry, "entry_points", lambda **kwargs: [first])
    registry.discover_sysid_backends()
    assert "toy" in registry.SIMULATION_CONTEXTS


def test_duplicate_metadata_names_fail_before_loading(monkeypatch):
    loads = []
    entry = SimpleNamespace(name="toy", load=lambda: loads.append(True))
    monkeypatch.setattr(registry, "entry_points", lambda **kwargs: [entry, entry])
    with pytest.raises(SimulationBackendError, match="Duplicate"):
        registry.discover_sysid_backends()
    assert loads == []


def test_entry_point_name_must_match_registration(monkeypatch):
    entry = SimpleNamespace(name="other", load=lambda: toy_registration)
    monkeypatch.setattr(registry, "entry_points", lambda **kwargs: [entry])
    with pytest.raises(SimulationBackendError, match="name"):
        registry.discover_sysid_backends()
    assert "toy" not in registry.SIMULATION_CONTEXTS


def test_python_registration_rejects_duplicate_names():
    registry.register_simulation_backend(toy_registration())
    with pytest.raises(SimulationBackendError, match="already registered"):
        registry.register_simulation_backend(toy_registration())
    with pytest.raises(SimulationBackendError, match="already registered"):
        registry.register_simulation_backend(toy_registration("newton"))


def test_registered_backend_without_sysid_fails_early(toy_task_data):
    registry.register_simulation_backend(
        registry.SimulationBackend("toy", ToyContext, None)
    )
    with pytest.raises(SimulationBackendError, match="system identification"):
        ActuatorSysIdConfig.model_validate(toy_task_data)


def test_wrong_concrete_context_factory_is_rejected(toy_task_data):
    class OtherContext(ToyContext):
        pass

    capability = registry.SysIdBackend(
        config_model=ToyConfig,
        create_context=lambda config: OtherContext(config.simulation.usd_path),
        create_evaluator=lambda config, space, context: None,  # type: ignore[arg-type]
    )
    registry.register_simulation_backend(
        registry.SimulationBackend("toy", ToyContext, capability)
    )
    config = ActuatorSysIdConfig.model_validate(toy_task_data)
    with pytest.raises(SimulationBackendError, match="context type"):
        registry.create_sysid_context(config)


def test_artifact_is_frozen():
    artifact = IdentifiedModelArtifact(Path("model.usd"), "usd")
    with pytest.raises(FrozenInstanceError):
        artifact.format = "other"  # type: ignore[misc]


def test_neutral_registry_import_does_not_import_newton_task():
    code = "import sys; import anchor_bench.core.simulation.registry; assert 'anchor_bench.tasks.actuator_sys_id.task' not in sys.modules; assert 'newton' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_discovery_loads_providers_in_metadata_name_order(monkeypatch):
    loaded = []

    def entry(name):
        def load():
            loaded.append(name)
            return lambda: registry.SimulationBackend(name, ToyContext)

        return SimpleNamespace(name=name, load=load)

    monkeypatch.setattr(
        registry, "entry_points", lambda **kwargs: [entry("z-last"), entry("a-first")]
    )
    registry.discover_sysid_backends()
    assert loaded == ["a-first", "z-last"]


@pytest.mark.parametrize(
    "provider",
    [
        None,
        lambda: object(),
        lambda: registry.SimulationBackend("toy", cast(Any, object)),
    ],
)
def test_invalid_plugin_capabilities_fail_atomically(monkeypatch, provider):
    entry = SimpleNamespace(name="toy", load=lambda: provider)
    monkeypatch.setattr(registry, "entry_points", lambda **kwargs: [entry])
    with pytest.raises(SimulationBackendError):
        registry.discover_sysid_backends()
    assert "toy" not in registry.SIMULATION_CONTEXTS
    assert not registry._DISCOVERED


def test_entry_point_cannot_replace_builtin_before_loading(monkeypatch):
    loads = []
    entry = SimpleNamespace(name="newton", load=lambda: loads.append(True))
    monkeypatch.setattr(registry, "entry_points", lambda **kwargs: [entry])
    with pytest.raises(SimulationBackendError, match="already registered"):
        registry.discover_sysid_backends()
    assert loads == []


def test_entry_point_enumeration_errors_are_wrapped_and_retryable(monkeypatch):
    def broken(**kwargs):
        raise RuntimeError("metadata unavailable")

    monkeypatch.setattr(registry, "entry_points", broken)
    with pytest.raises(SimulationBackendError, match="metadata unavailable"):
        registry.discover_sysid_backends()
    assert not registry._DISCOVERED
    monkeypatch.setattr(registry, "entry_points", lambda **kwargs: [])
    registry.discover_sysid_backends()
    assert registry._DISCOVERED


def test_provider_registration_side_effects_are_rolled_back(monkeypatch):
    def broken():
        registry.register_simulation_backend(toy_registration())
        raise RuntimeError("provider failed")

    entry = SimpleNamespace(name="toy", load=lambda: broken)
    monkeypatch.setattr(registry, "entry_points", lambda **kwargs: [entry])
    with pytest.raises(SimulationBackendError, match="provider failed"):
        registry.discover_sysid_backends()
    assert "toy" not in registry.SIMULATION_CONTEXTS


def test_validation_discovers_before_dispatch(monkeypatch, toy_task_data):
    entry = SimpleNamespace(name="toy", load=lambda: toy_registration)
    monkeypatch.setattr(registry, "entry_points", lambda **kwargs: [entry])
    config = ActuatorSysIdConfig.model_validate(toy_task_data)
    assert isinstance(config.simulation, ToyConfig)
    assert config.simulation.gain == 2


def test_yaml_parser_discovers_before_dispatch(monkeypatch, toy_task_data, tmp_path):
    import yaml

    from anchor_bench.schema.parse import parse

    entry = SimpleNamespace(name="toy", load=lambda: toy_registration)
    monkeypatch.setattr(registry, "entry_points", lambda **kwargs: [entry])
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(toy_task_data), encoding="utf-8")
    config = parse(path)
    assert isinstance(config, ActuatorSysIdConfig)
    assert isinstance(config.simulation, ToyConfig)
    assert config.simulation.gain == 2


def test_plugin_simulation_requires_usd_input(toy_task_data):
    registry.register_simulation_backend(toy_registration())
    toy_task_data["simulation"].pop("usd_path", None)
    with pytest.raises(ValidationError, match="usd_path"):
        ActuatorSysIdConfig.model_validate(toy_task_data)
