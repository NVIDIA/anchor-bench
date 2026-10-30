# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit simulator registration and configuration boundaries."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import ClassVar, Literal

import pytest
from pydantic import StrictInt, ValidationError

from anchor_bench.core.artifacts import _hash_name
from anchor_bench.core.simulation import SimulationContext, registry
from anchor_bench.errors import SimulationBackendError
from anchor_bench.schema.actuator_sys_id import ActuatorSysIdConfig
from anchor_bench.schema.simulation import SysIdSimulationConfig


class ToyConfig(SysIdSimulationConfig):
    type: Literal["toy"] = "toy"  # pyright: ignore[reportIncompatibleVariableOverride]
    gain: StrictInt


class ResourceOnlyContext(SimulationContext[object, object]):
    def _load(self, usd_path: Path) -> tuple[object, object]:
        return object(), object()


class ToyContext(ResourceOnlyContext):
    sysid_config_model: ClassVar[type[SysIdSimulationConfig] | None] = ToyConfig


def toy_registration(name: str = "toy") -> None:
    registry.register_simulation_backend(name, ToyContext)


@pytest.fixture(autouse=True)
def isolated_registry(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        registry, "SIMULATION_CONTEXTS", registry.SIMULATION_CONTEXTS.copy()
    )
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
    toy_registration()
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
    toy_registration()
    toy_task_data["simulation"].update(patch)
    with pytest.raises(ValidationError):
        ActuatorSysIdConfig.model_validate(toy_task_data)


def test_python_registration_rejects_duplicate_names():
    toy_registration()
    with pytest.raises(SimulationBackendError, match="already registered"):
        toy_registration()
    with pytest.raises(SimulationBackendError, match="already registered"):
        toy_registration("newton")


def test_registered_backend_without_sysid_fails_early(toy_task_data):
    registry.register_simulation_backend("toy", ResourceOnlyContext)
    with pytest.raises(SimulationBackendError, match="system identification"):
        ActuatorSysIdConfig.model_validate(toy_task_data)


def test_yaml_parser_uses_explicit_registration(toy_task_data, tmp_path):
    import yaml

    from anchor_bench.schema.parse import parse

    toy_registration()
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(toy_task_data), encoding="utf-8")
    config = parse(path)
    assert isinstance(config, ActuatorSysIdConfig)
    assert isinstance(config.simulation, ToyConfig)
    assert config.simulation.gain == 2


def test_plugin_simulation_requires_usd_input(toy_task_data):
    toy_registration()
    toy_task_data["simulation"].pop("usd_path", None)
    with pytest.raises(ValidationError, match="usd_path"):
        ActuatorSysIdConfig.model_validate(toy_task_data)


def test_unknown_backend_fails_at_context_lookup():
    with pytest.raises(ValueError, match="Unknown simulation backend"):
        registry.get_simulation_context_type("missing-backend")
