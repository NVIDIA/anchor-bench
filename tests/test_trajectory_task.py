from __future__ import annotations

import pandas as pd
import yaml

from anchor_bench.core.task_runner import TaskRunner
from anchor_bench.data import DatasetLoader
from anchor_bench.schema.parse import parse
from anchor_bench.schema.trajectory import TrajectoryConfig
from anchor_bench.tasks.trajectory.task import TRAJECTORY_ARTIFACT, TrajectoryTask


def _config_data():
    return {
        "type": "trajectory_v1",
        "name": "test_pack",
        "sample_rate_hz": 10,
        "actuators": [
            {"name": "shoulder", "unit": "rad"},
            {"name": "jaw", "unit": "normalized"},
        ],
        "poses": {
            "home": [0.0, 0.5],
            "target": [1.0, 0.25],
        },
        "trajectories": [
            {
                "name": "step",
                "initial_pose": "home",
                "pieces": [
                    {"type": "hold", "duration_s": 0.2},
                    {"type": "transition", "duration_s": 0.4, "to": "target"},
                    {"type": "hold", "duration_s": 0.2},
                ],
            }
        ],
    }


def test_task_writes_loader_compatible_command_only_parquet(tmp_path):
    config = TrajectoryConfig.model_validate(_config_data())

    result = TrajectoryTask(config).run(tmp_path)

    path = result / TRAJECTORY_ARTIFACT
    written = pd.read_parquet(path)
    assert written.columns.tolist() == [
        "time_ns",
        "time_utc",
        "experiment",
        "field",
        "value",
    ]
    assert written["experiment"].unique().tolist() == ["step"]
    assert set(written["field"]) == {"shoulder/command_q", "jaw/command_q"}

    loaded = DatasetLoader.load(path, dt=0.1)
    assert loaded.index.names == ["experiment", "time_ns"]
    assert loaded.columns.tolist() == ["shoulder/command_q", "jaw/command_q"]


def test_parser_and_default_runner_register_trajectory_task(tmp_path):
    config_path = tmp_path / "trajectory.yaml"
    config_path.write_text(yaml.safe_dump(_config_data()), encoding="utf-8")

    parsed = parse(config_path)
    output = TaskRunner().run(config_path, output=tmp_path / "output")

    assert isinstance(parsed, TrajectoryConfig)
    assert TrajectoryTask.name() == "trajectory_v1"
    assert (output / TRAJECTORY_ARTIFACT).is_file()
    assert (output / ".completed").is_file()
