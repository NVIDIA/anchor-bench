# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import yaml

from anchor_bench.errors import DatasetAlignmentError
from anchor_bench.mlops import MLOpsRun, bind_active_run
from anchor_bench.schema.actuator_sys_id import ActuatorSysIdConfig
from anchor_bench.tasks.actuator_sys_id import task as task_module
from anchor_bench.tasks.actuator_sys_id.artifacts import write_artifacts
from anchor_bench.tasks.actuator_sys_id.dataset import load_trajectories
from anchor_bench.tasks.actuator_sys_id.model import GenerationRecord, Trajectory
from anchor_bench.tasks.actuator_sys_id.objective import PopulationEvaluator
from anchor_bench.tasks.actuator_sys_id.optimizer import OptimizationStep
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
            "source": "actuator.parquet",
            "resampling": {"type": "uniform", "dt": 0.005},
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
    dataset = pd.DataFrame(
        {
            "hip/command_q": [0.0, 1.0],
            "hip/actual_q": [0.1, 0.8],
        },
        index=index,
    )

    def fake_load(
        path,
        *,
        dt,
        dataset_tracker,
        dataset_name,
        dataset_context,
        upload_dataset,
        joint_name_mapping,
        feature_name_mapping,
    ):
        assert path == "actuator.parquet"
        assert dt == 0.005
        assert dataset_tracker is None
        assert dataset_name == "system-identification-dataset"
        assert dataset_context == "system-identification"
        assert upload_dataset is False
        assert joint_name_mapping == {}
        assert feature_name_mapping == {}
        return dataset

    monkeypatch.setattr(
        "anchor_bench.tasks.actuator_sys_id.dataset.DatasetLoader.load",
        fake_load,
    )

    trajectories = load_trajectories(
        _config(tmp_path).dataset,
        joint_names=["hip"],
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


def test_load_trajectories_maps_parquet_joint_names_to_usd_names(tmp_path):
    rows = []
    for field, values in (
        ("recorded_hip/command_q", [0.0, 1.0]),
        ("recorded_hip/actual_q", [0.1, 0.8]),
    ):
        rows.extend(
            {
                "time_ns": timestamp,
                "experiment": "wave",
                "field": field,
                "value": value,
            }
            for timestamp, value in zip((0, 5_000_000), values)
        )
    source = tmp_path / "mapped.parquet"
    pd.DataFrame(rows).to_parquet(source, index=False)
    config = _config(
        tmp_path,
        dataset={
            "type": "parquet",
            "source": str(source),
            "joint_name_mapping": {"recorded_hip": "hip"},
            "resampling": {"type": "uniform", "dt": 0.005},
        },
    )

    trajectories = load_trajectories(config.dataset, joint_names=["hip"])

    assert trajectories[0].joint_names == ["hip"]
    np.testing.assert_allclose(
        trajectories[0].commanded_position,
        [[0.0], [1.0]],
    )
    np.testing.assert_allclose(
        trajectories[0].measured_position,
        [[0.1], [0.8]],
    )


def test_load_trajectories_supports_multiple_globbed_parquet_files(tmp_path):
    for name, experiment, offset in (
        ("train-a.parquet", "wave-a", 0.0),
        ("train-b.parquet", "wave-b", 1.0),
    ):
        rows = []
        for field, values in (
            ("hip/command_q", [offset, offset + 1.0]),
            ("hip/actual_q", [offset + 0.1, offset + 0.8]),
        ):
            rows.extend(
                {
                    "time_ns": timestamp,
                    "experiment": experiment,
                    "field": field,
                    "value": value,
                }
                for timestamp, value in zip((0, 5_000_000), values)
            )
        pd.DataFrame(rows).to_parquet(tmp_path / name, index=False)

    config = _config(
        tmp_path,
        dataset={
            "type": "parquet",
            "source": str(tmp_path / "train-*.parquet"),
            "resampling": {"type": "uniform", "dt": 0.005},
        },
    )

    trajectories = load_trajectories(config.dataset, joint_names=["hip"])

    assert [trajectory.experiment for trajectory in trajectories] == [
        "wave-a",
        "wave-b",
    ]
    np.testing.assert_allclose(
        trajectories[1].commanded_position,
        [[1.0], [2.0]],
    )


def test_load_trajectories_rejects_an_episode_without_a_transition(
    tmp_path,
    monkeypatch,
):
    dataset = pd.DataFrame(
        {"hip/command_q": [0.0], "hip/actual_q": [0.1]},
        index=pd.MultiIndex.from_tuples(
            [("wave", 0)],
            names=["experiment", "time_ns"],
        ),
    )
    monkeypatch.setattr(
        "anchor_bench.tasks.actuator_sys_id.dataset.DatasetLoader.load",
        lambda *args, **kwargs: dataset,
    )

    with pytest.raises(DatasetAlignmentError, match="at least two aligned samples"):
        load_trajectories(_config(tmp_path).dataset, joint_names=["hip"])


def test_task_runs_optimization_and_writes_artifacts(tmp_path, monkeypatch):
    config = _config(tmp_path)
    trajectory = Trajectory(
        experiment="wave",
        joint_names=["hip"],
        commanded_position=np.asarray([[0.0], [1.0]]),
        measured_position=np.asarray([[0.1], [0.8]]),
    )

    class FakeEvaluator(PopulationEvaluator):
        def evaluate(self, population, trajectories):
            assert len(population) == config.num_envs
            assert trajectories == (trajectory,)
            return np.asarray([1.0, 0.25])

    class FakeOptimizer:
        def __init__(self):
            self.converged = False
            self.best_score = float("inf")
            self.best_candidate = np.zeros(5)

        def step(self, objective):
            population = np.asarray(
                [
                    [-1.0, -1.0, -1.0, -1.0, -1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            )
            scores = objective.evaluate(population)
            best_index = int(np.argmin(scores))
            self.best_score = float(scores[best_index])
            self.best_candidate = population[best_index]
            self.converged = True
            return OptimizationStep(
                iteration=0,
                scores=scores,
                best_score=self.best_score,
                converged=True,
            )

    output_directory = tmp_path / "task-output"
    output_directory.mkdir()
    written = {}

    def write(output_dir, **values):
        assert output_dir == output_directory
        written.update(values)

    monkeypatch.setattr(
        task_module, "load_trajectories", lambda *args, **kwargs: [trajectory]
    )
    monkeypatch.setattr(task_module, "create_optimizer", lambda *args: FakeOptimizer())
    requested_context = {}

    class FakeContext:
        def create_sysid_evaluator(self, task_config, parameter_space):
            assert task_config is config
            return FakeEvaluator()

    def create_context(backend, usd_path, *, fixed_base):
        requested_context.update(
            backend=backend, usd_path=usd_path, fixed_base=fixed_base
        )
        return FakeContext()

    monkeypatch.setattr(task_module, "create_simulation_context", create_context)
    monkeypatch.setattr(task_module, "write_artifacts", write)

    mlops_run = Mock(spec=MLOpsRun)
    with bind_active_run(mlops_run):
        result = ActuatorSysIdTask(config).run(output_directory)

    assert result == output_directory
    assert written["best_mse"] == pytest.approx(0.25)
    assert requested_context == {
        "backend": "newton",
        "usd_path": config.simulation.usd_path,
        "fixed_base": True,
    }
    # The base evaluator exports no model unless the simulator overrides it.
    assert written["identified_usd_path"] is None
    assert written["best_parameters"]["hip"]["stiffness"] == pytest.approx(20.0)
    assert written["history"] == [
        GenerationRecord(generation=0, best_mse=0.25, population_mean_mse=0.625)
    ]
    mlops_run.log_metrics.assert_called_once_with(
        {
            "optimization/best_mse": 0.25,
            "optimization/iteration_best_mse": 0.25,
            "optimization/population_mean_mse": 0.625,
        },
        step=0,
    )
    mlops_run.log_artifact.assert_called_once_with(
        output_directory,
        collection="sysid-output",
        artifact_type="actuator-sysid",
    )


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
    identified_usd_path = output_dir / "identified.usda"
    result = write_artifacts(
        output_dir,
        best_parameters=candidate,
        best_mse=0.125,
        history=[GenerationRecord(0, 0.125, 0.5)],
        identified_usd_path=identified_usd_path,
    )

    document = yaml.safe_load(result.best_parameters_path.read_text())
    assert document["metric"] == "position_mse"
    assert document["parameters"] == candidate
    assert "0,0.125,0.5" in result.optimization_log_path.read_text()
    assert result.run_summary_path.is_file()
    assert result.identified_usd_path == identified_usd_path
