from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from anchor_bench.schema.trajectory import (
    ChirpPiece,
    SinePiece,
    TrajectoryConfig,
    TransitionPiece,
)


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


def test_config_applies_piece_defaults(config_data):
    data = deepcopy(config_data)
    data["trajectories"][0]["pieces"] = [
        {
            "type": "sine",
            "duration_s": 1.0,
            "components": [
                {"actuator": "shoulder", "amplitude": 0.1, "frequency_hz": 1.0}
            ],
        },
        {
            "type": "chirp",
            "duration_s": 1.0,
            "components": [
                {
                    "actuator": "shoulder",
                    "amplitude": 0.1,
                    "start_frequency_hz": 1.0,
                    "end_frequency_hz": 2.0,
                }
            ],
        },
        {"type": "transition", "duration_s": 1.0, "to": "home"},
    ]

    config = TrajectoryConfig.model_validate(data)
    sine, chirp, transition = config.trajectories[0].pieces

    assert isinstance(sine, SinePiece)
    assert sine.ramp_s == 1.0
    assert sine.components[0].phase_rad == 0.0
    assert isinstance(chirp, ChirpPiece)
    assert chirp.ramp_s == 1.0
    assert isinstance(transition, TransitionPiece)
    assert transition.interpolation == "quintic_smoothstep"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda data: data["poses"].update(home=[0.0]), "exactly 2"),
        (
            lambda data: data["trajectories"][0].update(initial_pose="missing"),
            "unknown pose",
        ),
        (
            lambda data: data["actuators"].append({"name": "shoulder", "unit": "rad"}),
            "actuator names must be unique",
        ),
        (
            lambda data: data["trajectories"][0]["pieces"].append(
                {
                    "type": "sine",
                    "duration_s": 1.0,
                    "components": [
                        {
                            "actuator": "missing",
                            "amplitude": 1.0,
                            "frequency_hz": 1.0,
                        }
                    ],
                }
            ),
            "unknown actuators",
        ),
    ],
)
def test_config_rejects_cross_reference_errors(config_data, change, message):
    data = deepcopy(config_data)
    change(data)

    with pytest.raises(ValidationError, match=message):
        TrajectoryConfig.model_validate(data)
