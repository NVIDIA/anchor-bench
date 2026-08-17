"""Trajectory-generation task orchestration."""

from __future__ import annotations

import logging
from pathlib import Path

from anchor_bench.schema.trajectory import TrajectoryConfig
from anchor_bench.tasks.base_task import BaseTask
from anchor_bench.tasks.trajectory.generator import generate_dataset

TRAJECTORY_ARTIFACT = "trajectories.parquet"

logger = logging.getLogger(__name__)


class TrajectoryTask(BaseTask[TrajectoryConfig]):
    """Generate commanded actuator positions as a long-form Parquet dataset."""

    def run(self, output_directory: Path) -> Path:
        dataset = generate_dataset(self.config)
        artifact = output_directory / TRAJECTORY_ARTIFACT
        dataset.to_parquet(artifact, index=False)
        logger.info(
            "Wrote trajectory dataset path=%s rows=%d",
            artifact,
            len(dataset.index),
        )
        return output_directory
