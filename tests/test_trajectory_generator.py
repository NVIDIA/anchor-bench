from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from anchor_bench.schema.trajectory import TrajectoryConfig
from anchor_bench.tasks.trajectory.generator import generate_dataset, generate_positions


@pytest.fixture
def config_data():
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
                    {"type": "hold", "duration_s": 0.25},
                    {
                        "type": "transition",
                        "duration_s": 0.4,
                        "to": "target",
                        "interpolation": "linear",
                    },
                    {"type": "hold", "duration_s": 0.2},
                ],
            }
        ],
    }


def test_piece_sampling_uses_half_even_counts_and_excluded_transition_endpoint(
    config_data,
):
    config = TrajectoryConfig.model_validate(config_data)

    positions = generate_positions(config, config.trajectories[0])

    # round(0.25 * 10) is 2, followed by four transition and two hold samples.
    assert positions.shape == (8, 2)
    np.testing.assert_allclose(
        positions,
        [
            [0.0, 0.5],
            [0.0, 0.5],
            [0.0, 0.5],
            [0.25, 0.4375],
            [0.5, 0.375],
            [0.75, 0.3125],
            [1.0, 0.25],
            [1.0, 0.25],
        ],
    )


def test_sine_and_chirp_modify_only_selected_actuators(config_data):
    data = deepcopy(config_data)
    data["sample_rate_hz"] = 4
    data["trajectories"][0]["pieces"] = [
        {
            "type": "sine",
            "duration_s": 1.0,
            "ramp_s": 0.0,
            "components": [
                {
                    "actuator": "shoulder",
                    "amplitude": 1.0,
                    "frequency_hz": 1.0,
                }
            ],
        },
        {
            "type": "chirp",
            "duration_s": 1.0,
            "ramp_s": 0.0,
            "center": "home",
            "components": [
                {
                    "actuator": "shoulder",
                    "amplitude": 1.0,
                    "start_frequency_hz": 0.0,
                    "end_frequency_hz": 0.0,
                    "phase_rad": np.pi / 2.0,
                }
            ],
        },
    ]
    config = TrajectoryConfig.model_validate(data)

    positions = generate_positions(config, config.trajectories[0])

    np.testing.assert_allclose(positions[:4, 0], [0.0, 1.0, 0.0, -1.0], atol=1e-12)
    np.testing.assert_allclose(positions[4:, 0], 1.0)
    np.testing.assert_allclose(positions[:, 1], 0.5)


@pytest.mark.parametrize(
    ("piece", "expected"),
    [
        (
            {"type": "hold", "duration_s": 0.04, "pose": "target"},
            [[1.0, 0.25], [1.0, 0.25]],
        ),
        (
            {
                "type": "transition",
                "duration_s": 0.04,
                "to": "target",
                "interpolation": "linear",
            },
            [[0.0, 0.5], [1.0, 0.25]],
        ),
        (
            {
                "type": "sine",
                "duration_s": 0.04,
                "ramp_s": 0.0,
                "components": [
                    {
                        "actuator": "shoulder",
                        "amplitude": 1.0,
                        "frequency_hz": 0.0,
                        "phase_rad": np.pi / 2.0,
                    }
                ],
            },
            [[1.0, 0.5], [1.0, 0.5]],
        ),
        (
            {
                "type": "chirp",
                "duration_s": 0.04,
                "ramp_s": 0.0,
                "components": [
                    {
                        "actuator": "shoulder",
                        "amplitude": 1.0,
                        "start_frequency_hz": 0.0,
                        "end_frequency_hz": 0.0,
                        "phase_rad": np.pi / 2.0,
                    }
                ],
            },
            [[1.0, 0.5], [1.0, 0.5]],
        ),
    ],
)
def test_short_piece_emits_one_sample_and_updates_current_pose(
    config_data,
    piece,
    expected,
):
    data = deepcopy(config_data)
    data["trajectories"][0]["pieces"] = [
        piece,
        {"type": "hold", "duration_s": 0.1},
    ]
    config = TrajectoryConfig.model_validate(data)

    positions = generate_positions(config, config.trajectories[0])

    np.testing.assert_allclose(positions, expected)


def test_dataset_uses_command_only_actuator_shape(config_data):
    dataset = generate_dataset(TrajectoryConfig.model_validate(config_data))

    assert dataset.columns.tolist() == [
        "time_ns",
        "time_utc",
        "experiment",
        "field",
        "value",
    ]
    assert dataset["experiment"].unique().tolist() == ["step"]
    assert set(dataset["field"]) == {"shoulder/command_q", "jaw/command_q"}
    assert dataset["time_ns"].dtype == np.dtype("int64")
    assert str(dataset["time_utc"].dtype) == "datetime64[ns, UTC]"
    assert dataset["value"].dtype == np.dtype("float64")
    assert not dataset["field"].str.contains("actual|temperature").any()
