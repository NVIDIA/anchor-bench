"""Actuator system-identification task orchestration."""

from __future__ import annotations

from pathlib import Path

from anchor_bench.mlops import get_active_run
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

    def run(self, output_directory: Path) -> Path:
        mlops_run = get_active_run()
        joint_names = [
            joint for group in self.config.joint_parameters for joint in group.names
        ]
        trajectories = load_trajectories(
            self.config.dataset,
            joint_names=joint_names,
            dataset_tracker=mlops_run,
            upload_dataset=(
                self.config.mlops.upload_dataset if self.config.mlops else False
            ),
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
            # The optimizer iteration is the shared monotonic step for all
            # convergence curves reported by this task.
            if mlops_run is not None:
                mlops_run.log_metrics(
                    {
                        "optimization/best_mse": step.best_score,
                        "optimization/iteration_best_mse": (step.iteration_best_score),
                        "optimization/population_mean_mse": step.mean_score,
                    },
                    step=step.iteration,
                )

        best_parameters = parameter_space.decode(optimizer.best_candidate)
        write_artifacts(
            output_directory,
            best_parameters=best_parameters,
            best_mse=optimizer.best_score,
            history=history,
            source_usd_path=self.config.simulation.usd_path,
            config=self.config,
        )
        # Upload only after every SysID output file is finalized. TaskRunner
        # separately owns the completion marker and tracked-run status.
        if mlops_run is not None:
            mlops_run.log_artifact(
                output_directory,
                collection="sysid-output",
                artifact_type="actuator-sysid",
            )
        return output_directory
