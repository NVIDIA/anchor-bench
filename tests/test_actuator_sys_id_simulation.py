from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import newton
import numpy as np

from anchor_bench.schema.actuator_sys_id import ActuatorSysIdConfig
from anchor_bench.tasks.actuator_sys_id import simulation as simulation_module
from anchor_bench.tasks.actuator_sys_id.parameter_space import create_parameter_space
from anchor_bench.tasks.actuator_sys_id.simulation import (
    NewtonPopulationEvaluator,
    _model_joint_indices,
)


def _config(tmp_path):
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
            joint_target_q_start=FakeArray([0, 6, 7, 13, 14]),
        ),
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
            )
            self.solver = object()

        def initialize(
            self,
            *,
            world_count,
            configure_builder,
            solver_options,
        ):
            self.created_world_count = world_count
            self.solver_options = solver_options
            configure_builder(self.template)

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
    assert not hasattr(evaluator, "_model")
    assert not hasattr(evaluator, "_solver")
    assert not hasattr(evaluator, "_state_in")
    assert not hasattr(evaluator, "_state_out")
    assert not hasattr(evaluator, "_control")
    assert not hasattr(evaluator, "_contacts")
    assert context.created_world_count == config.num_envs
    assert context.template.joint_target_ke == [15.0]
    assert context.solver_options == {
        "solver": "newton",
        "integrator": "implicitfast",
        "iterations": 10,
        "ls_iterations": 4,
    }
