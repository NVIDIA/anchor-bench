# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, cast

import newton
import numpy as np
import pytest
import warp as wp
from pxr import Usd, UsdPhysics

from anchor_bench.core.simulation.newton import NewtonSimulationContext
from anchor_bench.schema.actuator_sys_id import ActuatorSysIdConfig
from anchor_bench.tasks.actuator_sys_id import simulation as simulation_module
from anchor_bench.tasks.actuator_sys_id.model import Candidate, Trajectory
from anchor_bench.tasks.actuator_sys_id.parameter_space import create_parameter_space
from anchor_bench.tasks.actuator_sys_id.simulation import (
    NewtonPopulationEvaluator,
    _accumulate_position_error,
    _builder_body_indices,
    _builder_joint_indices,
    _model_body_indices,
    _model_joint_indices,
    _mujoco_actuator_paths,
    _set_builder_body_mass,
)

# The OpenUSD extension modules do not publish complete Python type information.


def _config(tmp_path, *, dataset_dt=0.005):
    return ActuatorSysIdConfig.model_validate(
        {
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
                "usd": {
                    "type": "local",
                    "path": str(tmp_path / "robot.usd"),
                },
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
                "resampling": {"type": "uniform", "dt": dataset_dt},
            },
        }
    )


def test_newton_joint_mapping_separates_state_and_target_layouts():
    class FakeArray:
        def __init__(self, values):
            self.values = np.asarray(values)

        def numpy(self):
            return self.values

    model = cast(
        "newton.Model",
        SimpleNamespace(
            joint_label=["root", "hip", "root", "hip"],
            joint_world=FakeArray([0, 0, 1, 1]),
            joint_q_start=FakeArray([0, 7, 8, 15, 16]),
            joint_qd_start=FakeArray([0, 6, 7, 13, 14]),
            joint_target_q_start=FakeArray([0, 5, 6, 11, 12]),
        ),
    )

    q_indices, dof_indices, target_indices = _model_joint_indices(
        model,
        ["hip"],
        2,
    )

    np.testing.assert_array_equal(q_indices, [[7], [15]])
    np.testing.assert_array_equal(dof_indices, [[6], [13]])
    np.testing.assert_array_equal(target_indices, [[5], [11]])


def test_newton_joint_mapping_rejects_non_scalar_targets():
    class FakeArray:
        def __init__(self, values):
            self.values = np.asarray(values)

        def numpy(self):
            return self.values

    model = cast(
        "newton.Model",
        SimpleNamespace(
            joint_label=["root", "hip", "root", "hip"],
            joint_world=FakeArray([0, 0, 1, 1]),
            joint_q_start=FakeArray([0, 7, 8, 15, 16]),
            joint_qd_start=FakeArray([0, 6, 7, 13, 14]),
            joint_target_q_start=FakeArray([0, 5, 7, 12, 14]),
        ),
    )

    with pytest.raises(ValueError, match="joint 'hip' is not scalar"):
        _model_joint_indices(model, ["hip"], 2)


def test_builder_joint_mapping_handles_final_joint_without_end_sentinel():
    builder = cast(
        "newton.ModelBuilder",
        SimpleNamespace(
            joint_label=["root", "hip"],
            joint_q_start=[0, 0],
            joint_qd_start=[0, 0],
            joint_coord_count=1,
            joint_dof_count=1,
        ),
    )

    assert _builder_joint_indices(builder, ["hip"]) == {"hip": (0, 0)}


def test_builder_body_mapping_and_mass_scaling():
    builder = cast(
        "newton.ModelBuilder",
        SimpleNamespace(
            body_label=["/World/base", "/World/link"],
            body_mass=[4.0, 2.0],
            body_inv_mass=[0.25, 0.5],
            body_inertia=[np.eye(3), np.eye(3) * 4.0],
            body_inv_inertia=[np.eye(3), np.eye(3) * 0.25],
        ),
    )

    indices = _builder_body_indices(builder, ["link"])
    _set_builder_body_mass(builder, indices["link"], 1.0)

    assert indices == {"link": 1}
    assert builder.body_mass[1] == pytest.approx(1.0)
    assert builder.body_inv_mass[1] == pytest.approx(1.0)
    np.testing.assert_allclose(
        np.asarray(cast(Any, builder.body_inertia[1])), np.eye(3) * 2.0
    )
    np.testing.assert_allclose(
        np.asarray(cast(Any, builder.body_inv_inertia[1])), np.eye(3) * 0.5
    )


def test_newton_body_mapping_resolves_each_replicated_world():
    class FakeArray:
        def __init__(self, values):
            self.values = np.asarray(values)

        def numpy(self):
            return self.values

    model = cast(
        "newton.Model",
        SimpleNamespace(
            body_label=["base", "link", "base", "link"],
            body_world=FakeArray([0, 0, 1, 1]),
        ),
    )

    np.testing.assert_array_equal(
        _model_body_indices(model, ["link"], 2),
        [[1], [3]],
    )


def test_mujoco_actuator_paths_select_only_identified_joints(tmp_path):
    source = tmp_path / "robot.usda"
    stage = Usd.Stage.CreateNew(str(source))
    hip = UsdPhysics.RevoluteJoint.Define(stage, "/World/hip")
    knee = UsdPhysics.RevoluteJoint.Define(stage, "/World/knee")
    hip_actuator = stage.DefinePrim("/World/hip_actuator", "MjcActuator")
    hip_actuator.CreateRelationship("mjc:target").SetTargets([hip.GetPath()])
    knee_actuator = stage.DefinePrim("/World/knee_actuator", "MjcActuator")
    knee_actuator.CreateRelationship("mjc:target").SetTargets([knee.GetPath()])
    stage.GetRootLayer().Save()

    assert _mujoco_actuator_paths(source, ["hip"]) == ("/World/hip_actuator",)


def test_actuator_config_rejects_dataset_timestep_mismatch(tmp_path):
    with pytest.raises(ValueError, match="resampling dt must match"):
        _config(tmp_path, dataset_dt=0.01)


def test_newton_evaluator_uses_shared_simulation_context(
    tmp_path,
    monkeypatch,
    caplog,
):
    class FakeArray:
        def __init__(self, values):
            self.values = np.asarray(values, dtype=np.float32)

        def numpy(self):
            return self.values

    class FakeContext:
        def __init__(self):
            self.created_world_count = None
            self.solver_options = None
            self.captured_step = None
            self.template = SimpleNamespace(
                joint_target_mode=[0],
                joint_target_ke=[0.0],
                joint_target_kd=[0.0],
                joint_armature=[0.0],
                joint_friction=[0.0],
                joint_damping=[0.0],
                custom_attributes={
                    "mujoco:dof_passive_damping": SimpleNamespace(values={0: 0.6})
                },
                joint_effort_limit=[0.0],
                joint_velocity_limit=[0.0],
            )
            self.control = SimpleNamespace(joint_target_q=FakeArray([8.0, 9.0]))
            self.model = SimpleNamespace(
                device="cpu",
                joint_q=FakeArray([0.0, 0.0]),
                joint_qd=FakeArray([0.0, 0.0]),
                joint_target_q=FakeArray([1.0, 2.0]),
                joint_target_ke=FakeArray([0.0, 0.0]),
                joint_target_kd=FakeArray([0.0, 0.0]),
                joint_armature=FakeArray([0.0, 0.0]),
                joint_friction=FakeArray([0.0, 0.0]),
                joint_damping=FakeArray([0.0, 0.0]),
            )
            self.solver = object()

        def initialize(
            self,
            *,
            world_count,
            configure_builder,
            configuration_key,
            solver_options,
            usd_ignore_paths,
        ):
            self.created_world_count = world_count
            self.configuration_key = configuration_key
            self.solver_options = solver_options
            self.usd_ignore_paths = usd_ignore_paths
            configure_builder(self.template)

        def capture_step_graph(self, *, dt, substeps):
            self.captured_step = (dt, substeps)
            return True

    context = FakeContext()
    config = _config(tmp_path)
    monkeypatch.setattr(simulation_module, "NewtonSimulationContext", FakeContext)
    requested_context = {}

    def create_context(backend, usd_path, *, fixed_base):
        requested_context.update(
            backend=backend,
            usd_path=usd_path,
            fixed_base=fixed_base,
        )
        return context

    monkeypatch.setattr(
        simulation_module,
        "create_simulation_context",
        create_context,
    )
    monkeypatch.setattr(
        simulation_module,
        "_mujoco_actuator_paths",
        lambda source_usd_path, joint_names: ("/World/hip_actuator",),
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

    with caplog.at_level(logging.DEBUG, logger=simulation_module.__name__):
        evaluator = NewtonPopulationEvaluator(config, create_parameter_space(config))

    assert evaluator._simulation_context is context
    assert evaluator._simulation_context.control is context.control
    np.testing.assert_allclose(evaluator._base_targets, [1.0, 2.0])
    assert context.created_world_count == config.num_envs
    assert requested_context["fixed_base"] is True
    assert context.usd_ignore_paths == ("/World/hip_actuator",)
    assert context.template.joint_target_ke == [15.0]
    assert (
        context.template.custom_attributes["mujoco:dof_passive_damping"].values[0]
        == context.template.joint_damping[0]
    )
    assert context.captured_step == (pytest.approx(0.005), 5)
    assert context.solver_options == {
        "solver": "newton",
        "integrator": "implicitfast",
        "iterations": 10,
        "ls_iterations": 4,
        "njmax": 512,
    }
    assert (
        "Initialized device score accumulation device=cpu worlds=2 joints=1"
        in caplog.text
    )


def test_newton_evaluator_weights_episode_mse_by_transition_count(monkeypatch):
    class FakeScoreArray:
        def __init__(self):
            self.values = np.zeros(2, dtype=np.float64)

        def zero_(self):
            self.values.fill(0.0)

        def numpy(self):
            return self.values

    evaluator = NewtonPopulationEvaluator.__new__(NewtonPopulationEvaluator)
    evaluator.num_envs = 2
    evaluator.joint_names = ["hip"]
    score_sums = FakeScoreArray()
    evaluator._score_sums = cast(Any, score_sums)
    short = Trajectory(
        experiment="short",
        joint_names=["hip"],
        commanded_position=np.zeros((2, 1)),
        measured_position=np.zeros((2, 1)),
    )
    long = Trajectory(
        experiment="long",
        joint_names=["hip"],
        commanded_position=np.zeros((3, 1)),
        measured_position=np.zeros((3, 1)),
    )
    episode_mse = {
        "short": np.asarray([3.0, 5.0]),
        "long": np.asarray([0.0, 2.0]),
    }
    monkeypatch.setattr(evaluator, "_apply_population", lambda population: None)

    def evaluate_trajectory(trajectory):
        transitions = trajectory.measured_position.shape[0] - 1
        score_sums.values += episode_mse[trajectory.experiment] * transitions
        return transitions

    monkeypatch.setattr(evaluator, "_evaluate_trajectory", evaluate_trajectory)

    scores = evaluator.evaluate([{}, {}], [short, long])

    np.testing.assert_allclose(scores, [1.0, 3.0])


def test_newton_evaluator_penalizes_only_nonfinite_worlds(monkeypatch, caplog):
    class FakeScoreArray:
        def __init__(self):
            self.values = np.asarray([2.0, np.nan, 4.0], dtype=np.float64)

        def zero_(self):
            pass

        def numpy(self):
            return self.values

    evaluator = NewtonPopulationEvaluator.__new__(NewtonPopulationEvaluator)
    evaluator.num_envs = 3
    evaluator.joint_names = ["hip"]
    evaluator._score_sums = cast(Any, FakeScoreArray())
    trajectory = Trajectory(
        experiment="unstable",
        joint_names=["hip"],
        commanded_position=np.zeros((2, 1)),
        measured_position=np.zeros((2, 1)),
    )
    monkeypatch.setattr(evaluator, "_apply_population", lambda population: None)
    monkeypatch.setattr(evaluator, "_evaluate_trajectory", lambda trajectory: 1)

    with caplog.at_level(logging.WARNING, logger=simulation_module.__name__):
        scores = evaluator.evaluate([{}, {}, {}], [trajectory])

    np.testing.assert_allclose(scores, [2.0, 8.0, 4.0])
    assert "worlds=[1] omitted_worlds=0 penalty=8 invalid=1/3" in caplog.text


def test_newton_evaluator_rejects_fully_nonfinite_population(monkeypatch):
    class FakeScoreArray:
        def zero_(self):
            pass

        def numpy(self):
            return np.asarray([np.nan, np.inf], dtype=np.float64)

    evaluator = NewtonPopulationEvaluator.__new__(NewtonPopulationEvaluator)
    evaluator.num_envs = 2
    evaluator.joint_names = ["hip"]
    evaluator._score_sums = cast(Any, FakeScoreArray())
    trajectory = Trajectory(
        experiment="unstable",
        joint_names=["hip"],
        commanded_position=np.zeros((2, 1)),
        measured_position=np.zeros((2, 1)),
    )
    monkeypatch.setattr(evaluator, "_apply_population", lambda population: None)
    monkeypatch.setattr(evaluator, "_evaluate_trajectory", lambda trajectory: 1)

    with pytest.raises(RuntimeError, match=r"every candidate \(2 worlds\)"):
        evaluator.evaluate([{}, {}], [trajectory])


def test_newton_evaluator_rejects_a_buffer_without_scored_transitions():
    evaluator = NewtonPopulationEvaluator.__new__(NewtonPopulationEvaluator)
    evaluator.joint_names = ["hip"]
    evaluator.buffer_steps = 1
    trajectory = Trajectory(
        experiment="too-short",
        joint_names=["hip"],
        commanded_position=np.zeros((2, 1)),
        measured_position=np.zeros((2, 1)),
    )

    with pytest.raises(ValueError, match="at least one scored transition"):
        evaluator._evaluate_trajectory(trajectory)


def test_position_error_kernel_accumulates_all_worlds_and_samples():
    joint_q = wp.array([2.0, 3.0, 3.0, 4.0], dtype=wp.float32, device="cpu")
    joint_q_indices = wp.array([0, 1, 2, 3], dtype=wp.int32, device="cpu")
    measured_positions = wp.array(
        [2.0, 3.0, 3.0, 4.0],
        dtype=wp.float64,
        device="cpu",
    )
    score_sums = wp.zeros(2, dtype=wp.float64, device="cpu")

    wp.launch(
        _accumulate_position_error,
        dim=2,
        inputs=[joint_q, joint_q_indices, measured_positions, 0, 2],
        outputs=[score_sums],
        device="cpu",
    )
    joint_q.assign([3.0, 4.0, 4.0, 5.0])
    wp.launch(
        _accumulate_position_error,
        dim=2,
        inputs=[joint_q, joint_q_indices, measured_positions, 1, 2],
        outputs=[score_sums],
        device="cpu",
    )

    np.testing.assert_allclose(score_sums.numpy(), [0.0, 4.0])


def test_newton_evaluator_restores_measured_pose_after_solver_reset(monkeypatch):
    class FakeArray:
        def __init__(self, values):
            self.values = np.asarray(values, dtype=np.float32)

        def assign(self, values):
            self.values = np.asarray(values, dtype=np.float32)

    class FakeSolver:
        def __init__(self):
            self.reset_states = []

        def reset(self, state):
            self.reset_states.append(state)
            state.joint_q.assign([-1.0, -1.0])
            state.joint_qd.assign([1.0, 1.0])

    state = SimpleNamespace(
        joint_q=FakeArray([0.0, 0.0]),
        joint_qd=FakeArray([0.0, 0.0]),
    )
    state_next = SimpleNamespace(
        joint_q=FakeArray([0.0, 0.0]),
        joint_qd=FakeArray([0.0, 0.0]),
    )
    solver = FakeSolver()
    context = SimpleNamespace(
        model=object(),
        solver=solver,
        state=state,
        state_next=state_next,
    )
    evaluator = NewtonPopulationEvaluator.__new__(NewtonPopulationEvaluator)
    evaluator._simulation_context = cast(NewtonSimulationContext, context)
    evaluator._base_q = np.asarray([0.0, 0.0], dtype=np.float32)
    evaluator._base_qd = np.asarray([0.0, 0.0], dtype=np.float32)
    evaluator._joint_q_indices = np.asarray([[0], [1]])
    evaluator._joint_dof_indices = np.asarray([[0], [1]])
    fk_calls = []
    monkeypatch.setattr(
        simulation_module.newton,
        "eval_fk",
        lambda *args: fk_calls.append(args),
    )

    evaluator._reset(np.asarray([0.75], dtype=np.float32))

    assert solver.reset_states == [state]
    assert fk_calls == [
        (context.model, state.joint_q, state.joint_qd, state),
        (context.model, state_next.joint_q, state_next.joint_qd, state_next),
    ]
    np.testing.assert_allclose(state.joint_q.values, [0.75, 0.75])
    np.testing.assert_allclose(state_next.joint_q.values, [0.75, 0.75])
    np.testing.assert_allclose(state.joint_qd.values, [0.0, 0.0])
    np.testing.assert_allclose(state_next.joint_qd.values, [0.0, 0.0])


def test_newton_evaluator_applies_candidates_and_scores_delayed_trajectories(
    monkeypatch,
):
    class FakeArray:
        def __init__(self, values):
            self.values = np.asarray(values, dtype=np.float32)
            self.numpy_calls = 0

        def numpy(self):
            self.numpy_calls += 1
            return self.values

        def assign(self, values):
            self.values = np.asarray(values, dtype=np.float32)

    class FakeScoreArray(FakeArray):
        def zero_(self):
            self.values.fill(0.0)

    class FakeSolver:
        def __init__(self):
            self.reset_states = []
            self.changed_flags = []

        def reset(self, state):
            self.reset_states.append(state)

        def notify_model_changed(self, flags):
            self.changed_flags.append(flags)

    class FakeContext:
        def __init__(self):
            self.model = SimpleNamespace(
                joint_q=FakeArray([0.0, 0.0, 0.0, 0.0]),
                joint_qd=FakeArray([0.0, 0.0, 0.0, 0.0]),
                joint_target_q=FakeArray([-9.0] * 6),
                joint_target_ke=FakeArray([0.0, 0.0, 0.0, 0.0]),
                joint_target_kd=FakeArray([0.0, 0.0, 0.0, 0.0]),
                joint_armature=FakeArray([0.0, 0.0, 0.0, 0.0]),
                joint_friction=FakeArray([0.0, 0.0, 0.0, 0.0]),
                joint_damping=FakeArray([0.0, 0.0, 0.0, 0.0]),
                body_mass=FakeArray([2.0, 2.0]),
                body_inv_mass=FakeArray([0.5, 0.5]),
                body_inertia=FakeArray(np.stack([np.eye(3) * 4.0] * 2)),
                body_inv_inertia=FakeArray(np.stack([np.eye(3) * 0.25] * 2)),
            )
            self.state = SimpleNamespace(
                joint_q=FakeArray([0.0, 0.0, 0.0, 0.0]),
                joint_qd=FakeArray([0.0, 0.0, 0.0, 0.0]),
            )
            self.state_next = SimpleNamespace(
                joint_q=FakeArray([0.0, 0.0, 0.0, 0.0]),
                joint_qd=FakeArray([0.0, 0.0, 0.0, 0.0]),
            )
            self.control = SimpleNamespace(
                joint_target_q=FakeArray([-9.0] * 6),
            )
            self.solver = FakeSolver()
            self.step_dts = []
            self.target_history = []
            self._simulated_positions = iter(
                (
                    [2.0, 3.0, 3.0, 4.0],
                    [3.0, 4.0, 4.0, 5.0],
                )
            )

        def step(self, *, dt, substeps=1):
            physics_dt = dt / substeps
            for _ in range(substeps):
                self.step_dts.append(physics_dt)
                self.target_history.append(
                    np.array(self.control.joint_target_q.numpy(), copy=True)
                )
                if len(self.step_dts) % 2 == 0:
                    self.state.joint_q.assign(next(self._simulated_positions))

    context = FakeContext()
    evaluator = NewtonPopulationEvaluator.__new__(NewtonPopulationEvaluator)
    evaluator.joint_names = ["hip", "knee"]
    evaluator.body_names = ["link"]
    evaluator.num_envs = 2
    evaluator.physics_dt = 0.001
    evaluator.control_substeps = 2
    evaluator.buffer_steps = 1
    evaluator.delay_steps = np.asarray([0, 1], dtype=np.int64)
    evaluator._simulation_context = cast(NewtonSimulationContext, context)
    evaluator._joint_q_indices = np.asarray([[0, 1], [2, 3]])
    evaluator._joint_dof_indices = np.asarray([[0, 1], [2, 3]])
    evaluator._joint_target_indices = np.asarray([[1, 2], [4, 5]])
    evaluator._body_indices = np.asarray([[0], [1]])
    evaluator._base_q = np.zeros(4, dtype=np.float32)
    evaluator._base_qd = np.zeros(4, dtype=np.float32)
    evaluator._base_targets = np.full(6, -9.0, dtype=np.float32)
    evaluator._base_model_arrays = {
        model_array_name: np.zeros(4, dtype=np.float32)
        for model_array_name in evaluator._MODEL_ARRAYS.values()
    }
    evaluator._base_body_arrays = {
        "body_mass": np.asarray([2.0, 2.0], dtype=np.float32),
        "body_inv_mass": np.asarray([0.5, 0.5], dtype=np.float32),
        "body_inertia": np.stack([np.eye(3, dtype=np.float32) * 4.0] * 2),
        "body_inv_inertia": np.stack([np.eye(3, dtype=np.float32) * 0.25] * 2),
    }
    evaluator._device = cast(Any, "cpu")
    evaluator._joint_q_indices_device = cast(Any, object())
    score_sums = FakeScoreArray([0.0, 0.0])
    evaluator._score_sums = cast(Any, score_sums)
    monkeypatch.setattr(simulation_module.newton, "eval_fk", lambda *args: None)
    monkeypatch.setattr(
        simulation_module.wp,
        "array",
        lambda values, **kwargs: FakeArray(values),
    )

    def launch_position_error(measured_positions, sample_index):
        simulated = context.state.joint_q.values[evaluator._joint_q_indices]
        measured = measured_positions.values.reshape(-1, 2)[sample_index]
        error = simulated - measured[np.newaxis, :]
        score_sums.values += np.sum(np.square(error), axis=1)

    monkeypatch.setattr(evaluator, "_launch_position_error", launch_position_error)

    population: list[Candidate] = [
        {
            "hip": {
                "stiffness": 1.0,
                "damping": 2.0,
                "armature": 3.0,
                "dry_friction": 4.0,
                "viscous_friction": 5.0,
            },
            "knee": {
                "stiffness": 11.0,
                "damping": 12.0,
                "armature": 13.0,
                "dry_friction": 14.0,
                "viscous_friction": 15.0,
            },
            "link": {"mass": 1.0},
        },
        {
            "hip": {
                "stiffness": 21.0,
                "damping": 22.0,
                "armature": 23.0,
                "dry_friction": 24.0,
                "viscous_friction": 25.0,
            },
            "knee": {
                "stiffness": 31.0,
                "damping": 32.0,
                "armature": 33.0,
                "dry_friction": 34.0,
                "viscous_friction": 35.0,
            },
            "link": {"mass": 4.0},
        },
    ]
    trajectory = Trajectory(
        experiment=object(),
        joint_names=["hip", "knee"],
        commanded_position=np.asarray([[10.0, 100.0], [20.0, 200.0], [30.0, 300.0]]),
        measured_position=np.asarray([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]),
    )

    scores = evaluator.evaluate(population, [trajectory])

    np.testing.assert_allclose(scores, [0.0, 1.0])
    np.testing.assert_allclose(context.model.joint_target_ke.values, [1, 11, 21, 31])
    np.testing.assert_allclose(context.model.joint_target_kd.values, [2, 12, 22, 32])
    np.testing.assert_allclose(context.model.joint_armature.values, [3, 13, 23, 33])
    np.testing.assert_allclose(context.model.joint_friction.values, [4, 14, 24, 34])
    np.testing.assert_allclose(context.model.joint_damping.values, [5, 15, 25, 35])
    np.testing.assert_allclose(context.model.body_mass.values, [1.0, 4.0])
    np.testing.assert_allclose(context.model.body_inv_mass.values, [1.0, 0.25])
    np.testing.assert_allclose(
        context.model.body_inertia.values,
        np.stack([np.eye(3) * 2.0, np.eye(3) * 8.0]),
    )
    np.testing.assert_allclose(
        context.model.body_inv_inertia.values,
        np.stack([np.eye(3) * 0.5, np.eye(3) * 0.125]),
    )
    expected_targets = np.asarray(
        [
            [-9.0, 10.0, 100.0, -9.0, 10.0, 100.0],
            [-9.0, 20.0, 100.0, -9.0, 20.0, 100.0],
        ]
    )
    np.testing.assert_allclose(context.target_history[::2], expected_targets)
    np.testing.assert_allclose(context.target_history[1::2], expected_targets)
    np.testing.assert_allclose(context.step_dts, [0.001] * 4)
    assert context.state.joint_q.numpy_calls == 0
    assert score_sums.numpy_calls == 1
    assert context.solver.reset_states == [context.state]
    assert context.solver.changed_flags == [
        newton.ModelFlags.JOINT_DOF_PROPERTIES
        | newton.ModelFlags.ACTUATOR_PROPERTIES
        | newton.ModelFlags.BODY_INERTIAL_PROPERTIES
    ]
