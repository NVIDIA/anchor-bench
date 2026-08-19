# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pytest

from anchor_bench.schema.actuator_sys_id import ActuatorSysIdConfig
from anchor_bench.tasks.actuator_sys_id.model import Candidate, Trajectory
from anchor_bench.tasks.actuator_sys_id.objective import ActuatorSysIdObjective
from anchor_bench.tasks.actuator_sys_id.optimizer import (
    CmaEsOptimizer,
    FloatArray,
    create_optimizer,
)
from anchor_bench.tasks.actuator_sys_id.parameter_space import create_parameter_space


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


def test_parameter_space_uses_deterministic_order_and_initials(tmp_path):
    space = create_parameter_space(_config(tmp_path))

    assert [parameter.name for parameter in space.parameters] == [
        "stiffness",
        "damping",
        "armature",
        "dry_friction",
        "viscous_friction",
    ]
    assert space.initial.tolist() == pytest.approx([15.0, 5.0, 0.05, 0.5, 1.0])
    assert space.initial_normalized.tolist() == pytest.approx(
        [-0.5, 0.0, 0.0, 0.0, 0.0]
    )

    candidate = space.decode(space.initial_normalized)
    assert candidate["hip"] == pytest.approx(
        {
            "stiffness": 15.0,
            "damping": 5.0,
            "armature": 0.05,
            "dry_friction": 0.5,
            "viscous_friction": 1.0,
        }
    )

    with pytest.raises(ValueError, match="at least one dimension"):
        space.normalize(np.asarray(0.0))
    with pytest.raises(ValueError, match="within their bounds"):
        space.normalize(np.asarray([31.0, 5.0, 0.05, 0.5, 1.0]))


def test_trajectory_requires_a_state_transition():
    with pytest.raises(ValueError, match="at least two samples"):
        Trajectory(
            experiment="single-sample",
            joint_names=["hip"],
            commanded_position=np.asarray([[0.0]]),
            measured_position=np.asarray([[0.0]]),
        )


def test_objective_decodes_population_before_evaluation(tmp_path):
    config = _config(tmp_path)
    space = create_parameter_space(config)
    trajectory = Trajectory(
        experiment="wave",
        joint_names=["hip"],
        commanded_position=np.asarray([[0.0], [1.0]]),
        measured_position=np.asarray([[0.1], [0.8]]),
    )

    class RecordingEvaluator:
        def __init__(self) -> None:
            self.population: Optional[Sequence[Candidate]] = None
            self.trajectories: Optional[Sequence[Trajectory]] = None

        def evaluate(
            self,
            population: Sequence[Candidate],
            trajectories: Sequence[Trajectory],
        ) -> FloatArray:
            self.population = population
            self.trajectories = trajectories
            return np.asarray([1.0, 0.25], dtype=np.float64)

    evaluator = RecordingEvaluator()
    objective = ActuatorSysIdObjective(space, evaluator, [trajectory])
    scores = objective.evaluate(
        np.asarray(
            [space.initial_normalized, np.zeros(space.dimension)],
            dtype=np.float64,
        )
    )

    population = evaluator.population
    trajectories = evaluator.trajectories
    assert population is not None
    assert trajectories is not None
    assert population[0]["hip"]["stiffness"] == pytest.approx(15.0)
    assert population[1]["hip"]["stiffness"] == pytest.approx(20.0)
    assert trajectories == (trajectory,)
    np.testing.assert_allclose(scores, [1.0, 0.25])


def test_cmaes_step_is_deterministic_and_enforces_iteration_limit(tmp_path):
    config = _config(tmp_path, num_envs=4)
    space = create_parameter_space(config)
    first = create_optimizer(config, space)
    second = create_optimizer(config, space)
    assert isinstance(first, CmaEsOptimizer)
    assert isinstance(second, CmaEsOptimizer)

    class RecordingObjective:
        def __init__(self) -> None:
            self.population: Optional[FloatArray] = None

        def evaluate(self, population: FloatArray) -> FloatArray:
            self.population = np.array(population, copy=True)
            return np.asarray([0.5, 0.25, 0.75, 1.0], dtype=np.float64)

    first_objective = RecordingObjective()
    second_objective = RecordingObjective()
    step = first.step(first_objective)
    second.step(second_objective)

    first_population = first_objective.population
    second_population = second_objective.population
    assert first_population is not None
    assert second_population is not None
    np.testing.assert_allclose(first_population, second_population)
    assert step.iteration == 0
    assert step.iteration_best_score == pytest.approx(0.25)
    assert step.mean_score == pytest.approx(0.625)
    assert not step.converged
    np.testing.assert_allclose(first.best_candidate, first_population[1])

    final_step = first.step(RecordingObjective())
    assert final_step.converged
    assert first.converged
    with pytest.raises(RuntimeError, match="already converged"):
        first.step(RecordingObjective())
