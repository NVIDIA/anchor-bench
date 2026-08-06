"""Trajectory-generation task orchestration."""

from __future__ import annotations

from pathlib import Path

from anchor_bench.schema.trajectory import TrajectoryConfig
from anchor_bench.tasks.base_task import BaseTask
from anchor_bench.tasks.trajectory.generator import generate_dataset

TRAJECTORY_ARTIFACT = "trajectories.parquet"


class TrajectoryTask(BaseTask[TrajectoryConfig]):
    """Generate commanded actuator positions as a long-form Parquet dataset."""

    def run(self, output_directory: Path) -> Path:
        dataset = generate_dataset(self.config)
        dataset.to_parquet(output_directory / TRAJECTORY_ARTIFACT, index=False)
        return output_directory
