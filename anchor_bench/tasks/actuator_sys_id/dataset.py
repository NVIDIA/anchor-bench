"""Load aligned command and measurement trajectories for system identification."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import pandas as pd

from anchor_bench.data import DatasetLoader
from anchor_bench.errors import DatasetAlignmentError
from anchor_bench.schema.dataset import ParquetDatasetConfig
from anchor_bench.tasks.actuator_sys_id.model import Trajectory


def load_trajectories(
    dataset_config: ParquetDatasetConfig,
    *,
    joint_names: Sequence[str],
) -> List[Trajectory]:
    """Load command and measurement trajectories from one aligned dataset."""

    dataset = DatasetLoader.load(
        dataset_config.source,
        dt=float(dataset_config.resampling.dt),
    )
    _validate_index(dataset)

    command_columns = [f"{joint}/command_q" for joint in joint_names]
    measured_columns = [f"{joint}/actual_q" for joint in joint_names]
    _require_columns(dataset, command_columns, "commanded")
    _require_columns(dataset, measured_columns, "measured")

    trajectories: List[Trajectory] = []
    experiments = list(
        dict.fromkeys(dataset.index.get_level_values("experiment").tolist())
    )
    for experiment in experiments:
        episode = dataset.xs(experiment, level="experiment")
        trajectories.append(
            Trajectory(
                experiment=experiment,
                joint_names=list(joint_names),
                commanded_position=episode.loc[:, command_columns].to_numpy(
                    dtype=np.float64
                ),
                measured_position=episode.loc[:, measured_columns].to_numpy(
                    dtype=np.float64
                ),
            )
        )
    if not trajectories:
        raise DatasetAlignmentError(
            "Commanded and measured datasets have no shared experiments."
        )
    return trajectories


def _validate_index(dataset: pd.DataFrame) -> None:
    if dataset.index.names != ["experiment", "time_ns"]:
        raise DatasetAlignmentError(
            "The actuator dataset must use experiment/time_ns indexing."
        )


def _require_columns(
    dataset: pd.DataFrame,
    expected: Sequence[str],
    label: str,
) -> None:
    missing = [column for column in expected if column not in dataset.columns]
    if missing:
        raise DatasetAlignmentError(
            f"The {label} dataset is missing fields: {', '.join(missing)}."
        )
