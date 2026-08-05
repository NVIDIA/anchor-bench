from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from anchor_bench.schema.actuator_sys_id import ActuatorSysIdConfig
from anchor_bench.tasks.actuator_sys_id import optimizer as optimizer_module
from anchor_bench.tasks.actuator_sys_id import simulation as simulation_module
from anchor_bench.tasks.actuator_sys_id import task as task_module
from anchor_bench.tasks.actuator_sys_id.artifacts import write_artifacts
from anchor_bench.tasks.actuator_sys_id.dataset import load_trajectories
from anchor_bench.tasks.actuator_sys_id.model import GenerationRecord, Trajectory
from anchor_bench.tasks.actuator_sys_id.optimizer import CmaEsOptimizer
from anchor_bench.tasks.actuator_sys_id.parameter_space import create_parameter_space
from anchor_bench.tasks.actuator_sys_id.simulation import (
    NewtonPopulationEvaluator,
    _model_joint_indices,
)
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


def test_parameter_space_uses_initials_and_midpoints(tmp_path):
    space = create_parameter_space(_config(tmp_path))

    assert space.dimension == 5
    assert space.initial.tolist() == pytest.approx([15.0, 5.0, 0.05, 0.5, 1.0])
    candidate = space.decode(space.initial_normalized)
    assert candidate["hip"]["stiffness"] == pytest.approx(15.0)
    assert candidate["hip"]["viscous_friction"] == pytest.approx(1.0)


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

    def fake_load(path, *, dt):
        assert path == "actuator.parquet"
        assert dt == 0.005
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


def test_task_orchestrates_optimizer_evaluator_and_artifacts(tmp_path, monkeypatch):
    config = _config(tmp_path)
    trajectory = Trajectory(
        experiment="wave",
        joint_names=["hip"],
        commanded_position=np.asarray([[0.0], [1.0]]),
        measured_position=np.asarray([[0.1], [0.8]]),
    )

    class FakeOptimizer:
        def __init__(self):
            self.generation = 0
            self.best_score = float("inf")
            self.best_normalized = None
            self.last_population = None

        def ask(self):
            self.last_population = np.asarray(
                [
                    [-1.0, -1.0, -1.0, -1.0, -1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            )
            return self.last_population

        def tell(self, scores):
            best = int(np.argmin(scores))
            self.best_score = float(scores[best])
            assert self.last_population is not None
            self.best_normalized = self.last_population[best]
            self.generation += 1
            return True

    class FakeEvaluator:
        def evaluate(self, population, trajectories):
            assert len(population) == 2
            assert trajectories == [trajectory]
            return np.asarray([1.0, 0.25])

    written = {}

    def write(output_dir, **values):
        written.update(values)
        return write_artifacts(output_dir, **values)

    optimizer = FakeOptimizer()
    output_directory = tmp_path / "task-output"
    output_directory.mkdir()

    monkeypatch.setattr(
        task_module,
        "load_trajectories",
        lambda *args, **kwargs: [trajectory],
    )
    monkeypatch.setattr(
        optimizer_module,
        "CmaEsOptimizer",
        lambda *args, **kwargs: optimizer,
    )
    monkeypatch.setattr(
        task_module,
        "NewtonPopulationEvaluator",
        lambda *args, **kwargs: FakeEvaluator(),
    )
    monkeypatch.setattr(task_module, "write_artifacts", write)

    result = ActuatorSysIdTask(config, output_directory).run()

    assert result == output_directory
    assert written["best_mse"] == pytest.approx(0.25)
    assert written["best_parameters"]["hip"]["stiffness"] == pytest.approx(20.0)
    assert len(written["history"]) == 1


def test_cmaes_optimizer_sampling_is_seed_deterministic(tmp_path):
    config = _config(tmp_path)
    space = create_parameter_space(config)
    first = CmaEsOptimizer(
        space,
        config.optimizer,
        population_size=4,
        seed=7,
    )
    second = CmaEsOptimizer(
        space,
        config.optimizer,
        population_size=4,
        seed=7,
    )

    first_population = first.ask()
    second_population = second.ask()
    first.tell(np.asarray([0.5, 0.25, 0.75, 1.0]))

    np.testing.assert_allclose(first_population, second_population)
    assert first.best_score == pytest.approx(0.25)


def test_newton_joint_mapping_separates_state_and_target_layouts():
    class FakeArray:
        def __init__(self, values):
            self.values = np.asarray(values)

        def numpy(self):
            return self.values

    model = SimpleNamespace(
        joint_label=["root", "hip", "root", "hip"],
        joint_world=FakeArray([0, 0, 1, 1]),
        joint_q_start=FakeArray([0, 7, 8, 15, 16]),
        joint_qd_start=FakeArray([0, 6, 7, 13, 14]),
        joint_target_q_start=FakeArray([0, 6, 7, 13, 14]),
    )

    q_indices, dof_indices, target_indices = _model_joint_indices(
        model,
        ["hip"],
        2,
    )

    np.testing.assert_array_equal(q_indices, [[7], [15]])
    np.testing.assert_array_equal(dof_indices, [[6], [13]])
    np.testing.assert_array_equal(target_indices, [[6], [13]])


def test_newton_evaluator_uses_shared_simulation_context(tmp_path, monkeypatch):
    class FakeArray:
        def __init__(self, values):
            self.values = np.asarray(values, dtype=np.float32)

        def numpy(self):
            return self.values

    class FakeContext:
        def __init__(self):
            self.runtime = SimpleNamespace(
                JointTargetMode=SimpleNamespace(POSITION=1),
            )
            self.created_world_count = None
            self.solver_options = None
            self.template = SimpleNamespace(
                joint_target_mode=[0],
                joint_target_ke=[0.0],
                joint_target_kd=[0.0],
                joint_armature=[0.0],
                joint_friction=[0.0],
                joint_damping=[0.0],
                joint_effort_limit=[0.0],
                joint_velocity_limit=[0.0],
            )
            self.control = SimpleNamespace(joint_target_pos=FakeArray([0.0, 0.0]))
            self.model = SimpleNamespace(
                joint_q=FakeArray([0.0, 0.0]),
                joint_qd=FakeArray([0.0, 0.0]),
                state=lambda: SimpleNamespace(),
                control=lambda: self.control,
                contacts=lambda: object(),
            )

        def create_replicated_model(self, *, world_count, configure_builder):
            self.created_world_count = world_count
            configure_builder(self.template)
            return self.model

        def create_mujoco_solver(self, model, **options):
            assert model is self.model
            self.solver_options = options
            return object()

    context = FakeContext()
    config = _config(tmp_path)
    monkeypatch.setattr(simulation_module, "NewtonSimulationContext", FakeContext)
    monkeypatch.setattr(
        simulation_module,
        "create_simulation_context",
        lambda backend, usd_path: context,
    )
    monkeypatch.setattr(
        simulation_module,
        "_builder_joint_indices",
        lambda builder, joint_names: {"hip": (0, 0)},
    )
    monkeypatch.setattr(
        simulation_module,
        "_model_joint_indices",
        lambda model, joint_names, num_envs: (
            np.asarray([[0], [1]]),
            np.asarray([[0], [1]]),
            np.asarray([[0], [1]]),
        ),
    )

    evaluator = NewtonPopulationEvaluator(config, create_parameter_space(config))

    assert evaluator._simulation_context is context
    assert context.created_world_count == config.num_envs
    assert context.template.joint_target_ke == [15.0]
    assert context.solver_options == {
        "solver": "newton",
        "integrator": "implicitfast",
        "iterations": 10,
        "ls_iterations": 4,
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
