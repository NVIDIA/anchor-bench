"""Actuator system-identification task orchestration."""

from __future__ import annotations

from pathlib import Path

from anchor_bench.schema.actuator_sys_id import ActuatorSysIdConfig
from anchor_bench.tasks.actuator_sys_id.artifacts import write_artifacts
from anchor_bench.tasks.actuator_sys_id.dataset import load_trajectories
from anchor_bench.tasks.actuator_sys_id.model import GenerationRecord
from anchor_bench.tasks.actuator_sys_id.objective import ActuatorSysIdObjective
from anchor_bench.tasks.actuator_sys_id.optimizer import create_optimizer
from anchor_bench.tasks.actuator_sys_id.parameter_space import create_parameter_space
from anchor_bench.tasks.actuator_sys_id.simulation import NewtonPopulationEvaluator
from anchor_bench.tasks.base_task import BaseTask


class ActuatorSysIdTask(BaseTask[ActuatorSysIdConfig]):
    """Fit actuator parameters by replaying recorded commands in Newton."""

    def run(self) -> Path:
        joint_names = [
            joint for group in self.config.joint_parameters for joint in group.names
        ]
        trajectories = load_trajectories(
            self.config.dataset,
            joint_names=joint_names,
        )
        parameter_space = create_parameter_space(self.config)
        optimizer = create_optimizer(self.config, parameter_space)
        evaluator = NewtonPopulationEvaluator(self.config, parameter_space)
        objective = ActuatorSysIdObjective(
            parameter_space,
            evaluator,
            trajectories,
        )
        history: list[GenerationRecord] = []

        while not optimizer.converged:
            step = optimizer.step(objective)
            history.append(
                GenerationRecord(
                    generation=step.iteration,
                    best_mse=step.best_score,
                    population_mean_mse=step.mean_score,
                )
            )

        write_artifacts(
            self.output_directory,
            best_parameters=parameter_space.decode(optimizer.best_candidate),
            best_mse=optimizer.best_score,
            history=history,
        )
        return self.output_directory
