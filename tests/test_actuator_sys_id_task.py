from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from anchor_bench.schema.actuator_sys_id import ActuatorSysIdConfig
from anchor_bench.tasks.actuator_sys_id import task as task_module
from anchor_bench.tasks.actuator_sys_id.artifacts import write_artifacts
from anchor_bench.tasks.actuator_sys_id.dataset import load_trajectories
from anchor_bench.tasks.actuator_sys_id.model import GenerationRecord
from anchor_bench.tasks.actuator_sys_id.task import ActuatorSysIdTask


def _config(tmp_path, **overrides):
    data = {
        "type": "actuator_sys_id",
        "joint_parameters": [
            {
                "names": ["hip"],
                "controller": {
                    "type": "pd",
                    "clamps": {
                        "effort_limit": 120.0,
                        "velocity_limit": 30.0,
                    },
                    "stiffness": {
                        "lower": 10.0,
                        "upper": 30.0,
                        "initial": 15.0,
                    },
                    "damping": {"lower": 0.0, "upper": 10.0},
                },
                "parameters": {
                    "dry_friction": {"lower": 0.0, "upper": 1.0},
                    "viscous_friction": {"lower": 0.0, "upper": 2.0},
                    "armature": {"lower": 0.0, "upper": 0.1},
                },
            }
        ],
        "optimizer": {
            "type": "python-cmaes",
            "sigma": 0.5,
            "max_iterations": 2,
            "epsilon": 0.001,
        },
        "simulation": {
            "type": "newton",
            "usd_path": str(tmp_path / "robot.usd"),
            "physics_freq": 1000,
            "control_freq": 200,
            "render_freq": 60,
            "solver": {
                "type": "warp_mujoco",
                "solver_iterations": 10,
                "line_search_iterations": 4,
            },
        },
        "num_envs": 2,
        "output_dir": str(tmp_path / "output"),
        "dataset": {
            "type": "parquet",
            "commanded_state": {"path": "commanded.parquet"},
            "measured_state": {"path": "measured.parquet"},
        },
    }
    data.update(overrides)
    return ActuatorSysIdConfig.model_validate(data)


def test_load_trajectories_aligns_commanded_and_measured_data(
    tmp_path,
    monkeypatch,
):
    index = pd.MultiIndex.from_tuples(
        [("wave", 0), ("wave", 5_000_000)],
        names=["experiment", "time_ns"],
    )
    commanded = pd.DataFrame({"hip/command_q": [0.0, 1.0]}, index=index)
    measured = pd.DataFrame({"hip/actual_q": [0.1, 0.8]}, index=index)

    def fake_load(path, *, dt):
        assert dt == 0.005
        return commanded if path == "commanded.parquet" else measured

    monkeypatch.setattr(
        "anchor_bench.tasks.actuator_sys_id.dataset.DatasetLoader.load",
        fake_load,
    )

    trajectories = load_trajectories(
        _config(tmp_path).dataset,
        joint_names=["hip"],
        control_dt=0.005,
    )

    assert len(trajectories) == 1
    np.testing.assert_allclose(
        trajectories[0].commanded_position,
        [[0.0], [1.0]],
    )
    np.testing.assert_allclose(
        trajectories[0].measured_position,
        [[0.1], [0.8]],
    )


def test_task_stops_at_unimplemented_optimization(tmp_path, monkeypatch):
    config = _config(tmp_path)
    seen = {}

    def fake_load(dataset_config, *, joint_names, control_dt):
        seen["dataset_config"] = dataset_config
        seen["joint_names"] = joint_names
        seen["control_dt"] = control_dt
        return []

    monkeypatch.setattr(task_module, "load_trajectories", fake_load)

    with pytest.raises(NotImplementedError, match="optimization is not implemented"):
        ActuatorSysIdTask(config).run()

    assert seen == {
        "dataset_config": config.dataset,
        "joint_names": ["hip"],
        "control_dt": 0.005,
    }


def test_writes_fitted_parameters_and_generation_history(tmp_path):
    candidate = {
        "hip": {
            "stiffness": 20.0,
            "damping": 5.0,
            "armature": 0.05,
            "dry_friction": 0.5,
            "viscous_friction": 1.0,
        }
    }

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    result = write_artifacts(
        output_dir,
        best_parameters=candidate,
        best_mse=0.125,
        history=[GenerationRecord(0, 0.125, 0.5)],
    )

    document = yaml.safe_load(result.best_parameters_path.read_text())
    assert document["metric"] == "position_mse"
    assert document["parameters"] == candidate
    assert "0,0.125,0.5" in result.optimization_log_path.read_text()
    assert result.run_summary_path.is_file()
