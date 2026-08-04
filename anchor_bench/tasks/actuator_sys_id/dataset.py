"""Load aligned command and measurement trajectories for system identification."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import pandas as pd

from anchor_bench.data import DatasetLoader
from anchor_bench.errors import DatasetAlignmentError
from anchor_bench.schema.actuator_sys_id import ParquetDatasetConfig
from anchor_bench.tasks.actuator_sys_id.model import Trajectory


def load_trajectories(
    dataset_config: ParquetDatasetConfig,
    *,
    joint_names: Sequence[str],
    control_dt: float,
) -> List[Trajectory]:
    """Load both datasets and align every shared experiment and timestamp."""

    commanded = DatasetLoader.load(
        dataset_config.commanded_state.path,
        dt=control_dt,
    )
    measured = DatasetLoader.load(
        dataset_config.measured_state.path,
        dt=control_dt,
    )
    _validate_index(commanded, "commanded")
    _validate_index(measured, "measured")

    command_columns = [f"{joint}/command_q" for joint in joint_names]
    measured_columns = [f"{joint}/actual_q" for joint in joint_names]
    _require_columns(commanded, command_columns, "commanded")
    _require_columns(measured, measured_columns, "measured")

    shared_index = commanded.index.intersection(measured.index, sort=False)
    if shared_index.empty:
        raise DatasetAlignmentError(
            "Commanded and measured datasets do not share timestamps."
        )
    commanded = commanded.loc[shared_index, command_columns]
    measured = measured.loc[shared_index, measured_columns]

    trajectories: List[Trajectory] = []
    experiments = list(
        dict.fromkeys(shared_index.get_level_values("experiment").tolist())
    )
    for experiment in experiments:
        command_episode = commanded.xs(experiment, level="experiment")
        measured_episode = measured.xs(experiment, level="experiment")
        episode_index = command_episode.index.intersection(
            measured_episode.index,
            sort=False,
        )
        if episode_index.empty:
            continue
        trajectories.append(
            Trajectory(
                experiment=experiment,
                joint_names=list(joint_names),
                commanded_position=command_episode.loc[
                    episode_index, command_columns
                ].to_numpy(dtype=np.float64),
                measured_position=measured_episode.loc[
                    episode_index, measured_columns
                ].to_numpy(dtype=np.float64),
            )
        )
    if not trajectories:
        raise DatasetAlignmentError(
            "Commanded and measured datasets have no shared experiments."
        )
    return trajectories


def _validate_index(dataset: pd.DataFrame, label: str) -> None:
    if dataset.index.names != ["experiment", "time_ns"]:
        raise DatasetAlignmentError(
            f"The {label} dataset must use experiment/time_ns indexing."
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
