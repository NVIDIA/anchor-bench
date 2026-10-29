# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import builtins
import importlib
from pathlib import Path
from typing import Literal
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import yaml

from anchor_bench.core.simulation import SimulationContext, registry
from anchor_bench.core.task_runner import TaskRunner
from anchor_bench.errors import DatasetAlignmentError, SimulationBackendError
from anchor_bench.mlops import MLOpsRun, bind_active_run
from anchor_bench.schema.actuator_sys_id import ActuatorSysIdConfig
from anchor_bench.schema.simulation import SysIdSimulationConfig
from anchor_bench.tasks.actuator_sys_id import artifacts as artifacts_module
from anchor_bench.tasks.actuator_sys_id import task as task_module
from anchor_bench.tasks.actuator_sys_id.artifacts import write_artifacts
from anchor_bench.tasks.actuator_sys_id.dataset import load_trajectories
from anchor_bench.tasks.actuator_sys_id.model import (
    GenerationRecord,
    IdentifiedModelArtifact,
    SysIdResult,
    Trajectory,
)
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


class ToySimulationConfig(SysIdSimulationConfig):
    type: Literal["task-toy"] = "task-toy"  # pyright: ignore[reportIncompatibleVariableOverride]
    gain: float = 2.0


class ToyContext(SimulationContext[object, object]):
    def _load(self, usd_path: Path):
        return object(), object()


@pytest.fixture
def register_toy(monkeypatch, tmp_path):
    monkeypatch.setattr(SimulationContext, "_instance", None)
    monkeypatch.setattr(
        registry, "SIMULATION_CONTEXTS", registry.SIMULATION_CONTEXTS.copy()
    )
    monkeypatch.setattr(registry, "_SYSID_CALLBACKS", registry._SYSID_CALLBACKS.copy())

    def register(create_evaluator, *, export_model=None, describe_runtime=None):
        usd = tmp_path / "scene.usda"
        usd.write_text("#usda 1.0\n")
        context = ToyContext(str(usd))

        def create_context(config):
            assert isinstance(config.simulation, ToySimulationConfig)
            assert config.simulation.gain == 2.0
            return context

        registry.register_simulation_backend(
            "task-toy",
            ToyContext,
            config_model=ToySimulationConfig,
            create_context=create_context,
            create_evaluator=create_evaluator,
            export_model=export_model,
            describe_runtime=describe_runtime,
        )
        config = _config(
            tmp_path,
            simulation={
                "type": "task-toy",
                "usd_path": str(usd),
                "gain": 2.0,
                "physics_freq": 1000,
                "control_freq": 200,
                "render_freq": 60,
            },
        )
        return config, context

    return register


def test_sysid_result_preserves_eight_positional_argument_signature(tmp_path):
    usd_path = tmp_path / "identified.usda"

    result = SysIdResult(
        {"hip": {"armature": 0.05}},
        0.25,
        2,
        tmp_path,
        tmp_path / "best_parameters.yaml",
        tmp_path / "optimization_log.csv",
        tmp_path / "run_summary.yaml",
        usd_path,
    )

    assert result.identified_usd_path == usd_path
    assert result.identified_model_path is None


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


def test_task_uses_registered_evaluator_and_context(
    tmp_path, monkeypatch, register_toy
):
    trajectory = Trajectory(
        experiment="wave",
        joint_names=["hip"],
        commanded_position=np.asarray([[0.0], [1.0]]),
        measured_position=np.asarray([[0.1], [0.8]]),
    )

    class FakeEvaluator:
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
    evaluator = FakeEvaluator()
    received = {}

    def create_evaluator(selected_config, space, selected_context):
        received.update(config=selected_config, context=selected_context, space=space)
        return evaluator

    def export_model(selected_config, best, selected_context, output):
        assert selected_config is config
        assert selected_context is context
        assert output == output_directory
        assert best["hip"]["stiffness"] == pytest.approx(20.0)
        path = output / "identified.json"
        path.write_text('{"model": "toy"}')
        return IdentifiedModelArtifact(path, "json")

    def describe_runtime(selected_evaluator):
        assert selected_evaluator is evaluator
        return {"type": "task-toy", "effective_gain": 2.0}

    config, context = register_toy(
        create_evaluator,
        export_model=export_model,
        describe_runtime=describe_runtime,
    )
    monkeypatch.setattr(task_module, "write_artifacts", write)

    mlops_run = Mock(spec=MLOpsRun)
    with bind_active_run(mlops_run):
        result = ActuatorSysIdTask(config).run(output_directory)

    assert result == output_directory
    assert written["best_mse"] == pytest.approx(0.25)
    assert received["config"] is config
    assert received["context"] is context
    assert written["model_artifact"] == IdentifiedModelArtifact(
        output_directory / "identified.json",
        "json",
    )
    assert written["runtime_simulation"] == {"type": "task-toy", "effective_gain": 2.0}
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


@pytest.mark.parametrize("model_format", [None, "json", "usd"])
def test_common_artifacts_allow_missing_or_json_model_export(
    tmp_path,
    monkeypatch,
    model_format,
):
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
    model_artifact = None
    if model_format is not None:
        # Deliberately use the USD suffix for JSON: compatibility depends on format.
        path = output_dir / "identified.usda"
        path.write_text("exported")
        model_artifact = IdentifiedModelArtifact(path, model_format)

    original_import = builtins.__import__

    def guard_import(name, *args, **kwargs):
        if name.endswith("actuator_sys_id.usd"):
            raise AssertionError("common artifacts must not import USD support")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard_import)
    importlib.reload(artifacts_module)
    result = write_artifacts(
        output_dir,
        best_parameters=candidate,
        best_mse=0.125,
        history=[GenerationRecord(0, 0.125, 0.5)],
        model_artifact=model_artifact,
        runtime_simulation={"type": "toy", "solver": {"iterations": 8}},
    )

    document = yaml.safe_load(result.best_parameters_path.read_text())
    assert document["metric"] == "position_mse"
    assert document["parameters"] == candidate
    assert "0,0.125,0.5" in result.optimization_log_path.read_text()
    summary = yaml.safe_load(result.run_summary_path.read_text())
    assert summary["best_mse"] == 0.125
    assert summary["generations"] == 1
    assert summary["runtime_simulation"] == {"type": "toy", "solver": {"iterations": 8}}
    assert result.identified_model_path == (
        model_artifact.path if model_artifact else None
    )
    assert result.identified_usd_path == (
        model_artifact.path if model_artifact and model_format == "usd" else None
    )
    if model_artifact is None:
        assert not (output_dir / "identified.usda").exists()


def test_export_failure_prevents_completion(tmp_path, monkeypatch, register_toy):
    class Evaluator:
        def evaluate(self, population, trajectories):
            return np.asarray([1.0] * len(population))

    def export_model(*args):
        raise RuntimeError("toy export failed")

    config, _ = register_toy(lambda *args: Evaluator(), export_model=export_model)
    config.num_envs = 4
    monkeypatch.setattr(
        task_module,
        "load_trajectories",
        lambda *args, **kwargs: [
            Trajectory("wave", ["hip"], np.zeros((2, 1)), np.zeros((2, 1)))
        ],
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config.model_dump(mode="json")))
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="toy export failed"):
        TaskRunner().run(str(config_path), output=output)
    assert not (output / ".completed").exists()
    assert not (output / "best_params.yaml").exists()


def test_missing_sysid_capability_fails_before_optimizer(tmp_path, monkeypatch):
    config = _config(tmp_path)
    monkeypatch.setattr(
        registry, "SIMULATION_CONTEXTS", registry.SIMULATION_CONTEXTS.copy()
    )
    monkeypatch.setattr(registry, "_SYSID_CALLBACKS", registry._SYSID_CALLBACKS.copy())
    registry.SIMULATION_CONTEXTS["resource-only"] = ToyContext
    config.simulation = SysIdSimulationConfig(
        usd_path=config.simulation.usd_path,
        type="resource-only",
        physics_freq=1000,
        control_freq=200,
        render_freq=60,
    )

    def unexpected(*args, **kwargs):
        raise AssertionError("must fail before optimization or dataset loading")

    monkeypatch.setattr(task_module, "load_trajectories", unexpected)
    monkeypatch.setattr(task_module, "create_optimizer", unexpected)
    with pytest.raises(
        SimulationBackendError, match="does not support system identification"
    ):
        ActuatorSysIdTask(config).run(tmp_path)
