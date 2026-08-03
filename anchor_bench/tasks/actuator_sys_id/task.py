"""Actuator system-identification task orchestration."""

from __future__ import annotations

from pathlib import Path

from anchor_bench.schema.actuator_sys_id import ActuatorSysIdConfig
from anchor_bench.tasks.actuator_sys_id.dataset import load_trajectories
from anchor_bench.tasks.base_task import BaseTask


class ActuatorSysIdTask(BaseTask[ActuatorSysIdConfig]):
    """Fit actuator parameters by replaying recorded commands in Newton."""

    def run(self) -> Path:
        joint_names = [
            joint for group in self.config.joint_parameters for joint in group.names
        ]
        control_dt = 1.0 / self.config.simulation.control_freq
        load_trajectories(
            self.config.dataset,
            joint_names=joint_names,
            control_dt=control_dt,
        )
        raise NotImplementedError(
            "Actuator system-identification optimization is not implemented."
        )
