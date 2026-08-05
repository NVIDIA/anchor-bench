"""Actuator system-identification task orchestration."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from anchor_bench.schema.actuator_sys_id import ActuatorSysIdConfig
from anchor_bench.tasks.actuator_sys_id.artifacts import write_artifacts
from anchor_bench.tasks.actuator_sys_id.dataset import load_trajectories
from anchor_bench.tasks.actuator_sys_id.model import GenerationRecord, SysIdResult
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
        history = []

        for generation in range(self.config.optimizer.max_iterations):
            normalized = optimizer.ask()
            population = parameter_space.decode_population(normalized)
            scores = evaluator.evaluate(population, trajectories)
            converged = optimizer.tell(scores)
            history.append(
                GenerationRecord(
                    generation=generation,
                    best_mse=optimizer.best_score,
                    population_mean_mse=float(np.mean(scores)),
                )
            )
            if converged:
                break

        if optimizer.best_normalized is None:
            raise RuntimeError("CMA-ES did not produce a valid actuator candidate")
        best_parameters = parameter_space.decode(optimizer.best_normalized)
        sys_id_result = write_artifacts(
            self.output_directory,
            best_parameters=best_parameters,
            best_mse=optimizer.best_score,
            history=history,
        )
        self.print_summary(sys_id_result)

        return self.output_directory

    def print_summary(self, sys_id_result: SysIdResult) -> None:
        """Print the best fit and generated artifact locations."""

        print(f"Best MSE: {sys_id_result.best_mse}")
        print(f"Generations: {sys_id_result.generations}")
        print(f"Output directory: {sys_id_result.output_dir}")
        print(f"Best parameters: {sys_id_result.best_parameters}")
        print(f"Best parameters path: {sys_id_result.best_parameters_path}")
        print(f"Optimization log path: {sys_id_result.optimization_log_path}")
        print(f"Run summary path: {sys_id_result.run_summary_path}")
