# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import pytest
from pydantic import ValidationError

from anchor_bench.schema.actuator_sys_id import (
    ActuatorSysIdConfig,
    PdControllerConfig,
)
from anchor_bench.schema.dataset import ParquetDatasetConfig


@pytest.fixture
def valid_config_data():
    return {
        "type": "actuator_sys_id",
        "joint_parameters": [
            {
                "names": ["hip", "knee"],
                "controller": {
                    "type": "pd",
                    "clamps": {
                        "effort_limit": 120.0,
                        "velocity_limit": 30.0,
                    },
                    "stiffness": {"lower": 1.0, "upper": 100.0, "initial": 20.0},
                    "damping": {"lower": 0.0, "upper": 10.0},
                },
                "parameters": {
                    "dry_friction": {"lower": 0.0, "upper": 1.0},
                    "viscous_friction": {"lower": 0.0, "upper": 1.0},
                    "armature": {"lower": 0.0, "upper": 0.1},
                },
            }
        ],
        "optimizer": {
            "type": "python-cmaes",
            "sigma": 0.5,
            "max_iterations": 100,
            "epsilon": 0.001,
        },
        "simulation": {
            "type": "newton",
            "usd_path": "assets/robots/test.usd",
            "physics_freq": 1000,
            "control_freq": 200,
            "render_freq": 60,
            "solver": {
                "type": "warp_mujoco",
                "solver_iterations": 10,
                "line_search_iterations": 4,
            },
        },
        "dataset": {
            "type": "parquet",
            "source": "runs/actuator.parquet",
            "resampling": {
                "type": "uniform",
                "dt": 0.005,
            },
        },
    }


def test_model_matches_schema_and_applies_defaults(valid_config_data):
    config = ActuatorSysIdConfig.model_validate(valid_config_data)

    assert config.type == "actuator_sys_id"
    assert config.num_envs == 64
    assert config.buffer_time == 0.0
    assert isinstance(config.dataset, ParquetDatasetConfig)
    assert config.dataset.source == "runs/actuator.parquet"
    assert config.dataset.joint_name_mapping == {}
    assert config.dataset.resampling.dt == 0.005
    assert isinstance(config.joint_parameters[0].controller, PdControllerConfig)
    assert config.joint_parameters[0].controller.delay_duration_ms == 0
    assert config.simulation.solver.integrator == "implicitfast"
    assert config.mlops is None


def test_model_accepts_optional_mlops_configuration(valid_config_data):
    data = deepcopy(valid_config_data)
    data["mlops"] = {
        "backend": "custom",
        "experiment_name": "actuator-identification",
        "run_name": "so101-baseline",
        "options": {"mode": "offline"},
    }
    data["description"] = "Identify the SO-101 arm actuators."
    data["tags"] = ["so101", "baseline"]

    config = ActuatorSysIdConfig.model_validate(data)

    assert config.mlops is not None
    assert config.mlops.backend == "custom"
    assert config.mlops.experiment_name == "actuator-identification"
    assert config.mlops.run_name == "so101-baseline"
    assert config.mlops.options == {"mode": "offline"}
    assert config.description == "Identify the SO-101 arm actuators."
    assert config.tags == ["so101", "baseline"]


def test_model_accepts_parquet_to_usd_joint_name_mapping(valid_config_data):
    data = deepcopy(valid_config_data)
    data["dataset"]["joint_name_mapping"] = {
        "recorded_hip": "/robot/joints/hip",
        "recorded_knee": "/robot/joints/knee",
    }

    config = ActuatorSysIdConfig.model_validate(data)

    assert config.dataset.joint_name_mapping == {
        "recorded_hip": "/robot/joints/hip",
        "recorded_knee": "/robot/joints/knee",
    }


def test_model_rejects_extra_properties(valid_config_data):
    data = deepcopy(valid_config_data)
    data["unexpected"] = True

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ActuatorSysIdConfig.model_validate(data)


def test_model_rejects_duplicate_joint_names(valid_config_data):
    data = deepcopy(valid_config_data)
    data["joint_parameters"][0]["names"] = ["hip", "hip"]

    with pytest.raises(ValidationError, match="joint names must be unique"):
        ActuatorSysIdConfig.model_validate(data)


def test_model_rejects_initial_outside_bound(valid_config_data):
    data = deepcopy(valid_config_data)
    data["joint_parameters"][0]["parameters"]["armature"]["initial"] = 1.0

    with pytest.raises(ValidationError, match="initial must lie between"):
        ActuatorSysIdConfig.model_validate(data)


def test_model_rejects_coerced_integer(valid_config_data):
    data = deepcopy(valid_config_data)
    data["num_envs"] = "64"

    with pytest.raises(ValidationError):
        ActuatorSysIdConfig.model_validate(data)


def test_model_requires_at_least_two_cmaes_environments(valid_config_data):
    data = deepcopy(valid_config_data)
    data["num_envs"] = 1

    with pytest.raises(ValidationError, match="greater than or equal to 2"):
        ActuatorSysIdConfig.model_validate(data)


def test_model_rejects_negative_actuator_parameter_bounds(valid_config_data):
    data = deepcopy(valid_config_data)
    data["joint_parameters"][0]["parameters"]["armature"]["lower"] = -0.1

    with pytest.raises(ValidationError, match="bounds must be nonnegative: armature"):
        ActuatorSysIdConfig.model_validate(data)


def test_controller_rejects_null(valid_config_data):
    data = deepcopy(valid_config_data)
    data["joint_parameters"][0]["controller"] = None

    with pytest.raises(ValidationError):
        ActuatorSysIdConfig.model_validate(data)


def test_model_rejects_dataset_control_timestep_mismatch(valid_config_data):
    data = deepcopy(valid_config_data)
    data["dataset"]["resampling"]["dt"] = 0.01

    with pytest.raises(ValidationError, match="resampling dt must match"):
        ActuatorSysIdConfig.model_validate(data)


def test_model_rejects_non_finite_numbers(valid_config_data):
    data = deepcopy(valid_config_data)
    data["optimizer"]["sigma"] = float("nan")

    with pytest.raises(ValidationError):
        ActuatorSysIdConfig.model_validate(data)
