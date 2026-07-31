from __future__ import annotations

from typing import Any, Optional, Tuple

import pandas as pd
import pytest

from anchor_bench.core import SimulationContext
from anchor_bench.train import (
    BasePreprocessor,
    PositionErrorPreprocessor,
    ResidualTorquePreprocessor,
    SimPreprocessor,
    SolverPDPreprocessor,
)


class DifferencePreprocessor(BasePreprocessor):
    def process(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"position_error": self.dataset["command"] - self.dataset["position"]},
            index=self.dataset.index,
        )


class RecordingSimulationContext(SimulationContext):
    def __init__(self) -> None:
        self.loaded_path: Optional[str] = None

    def load(self, usd_path: str) -> Tuple[Any, Any]:
        self.loaded_path = usd_path
        return "model", "state"


class SimulatedFeaturePreprocessor(SimPreprocessor):
    def process(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"simulated": self.dataset["command"]},
            index=self.dataset.index,
        )


@pytest.fixture
def dataset() -> pd.DataFrame:
    index = pd.MultiIndex.from_tuples(
        [("wave", 0), ("wave", 20_000_000)],
        names=["experiment", "time_ns"],
    )
    return pd.DataFrame(
        {
            "position": [1.0, 2.0],
            "command": [1.5, 2.25],
        },
        index=index,
    )


def test_base_preprocessor_generates_aligned_features(
    dataset: pd.DataFrame,
) -> None:
    result = DifferencePreprocessor(dataset).process()

    assert result.index.equals(dataset.index)
    assert result["position_error"].tolist() == [0.5, 0.25]


def test_base_preprocessor_cannot_be_instantiated(dataset: pd.DataFrame) -> None:
    with pytest.raises(TypeError):
        BasePreprocessor(dataset)  # pyright: ignore[reportAbstractUsage]


def test_simulation_context_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        SimulationContext()  # pyright: ignore[reportAbstractUsage]


def test_sim_preprocessor_loads_usd_asset(dataset: pd.DataFrame) -> None:
    context = RecordingSimulationContext()

    preprocessor = SimulatedFeaturePreprocessor(
        dataset,
        "assets/robots/so101.usd",
        context,
    )

    assert context.loaded_path == "assets/robots/so101.usd"
    assert preprocessor.model == "model"
    assert preprocessor.state == "state"


def test_position_error_preprocessor_handles_multiple_entities() -> None:
    dataset = pd.DataFrame(
        {
            "shoulder/command_q": [2.0, 1.0],
            "shoulder/actual_q": [1.0, 1.5],
            "elbow/command_q": [3.0, 4.0],
            "elbow/actual_q": [2.5, 3.0],
        }
    )

    result = PositionErrorPreprocessor(dataset).process()

    assert result.columns.tolist() == [
        "shoulder/position_error",
        "elbow/position_error",
    ]
    assert result["shoulder/position_error"].tolist() == [1.0, -0.5]
    assert result["elbow/position_error"].tolist() == [0.5, 1.0]
    assert result.index.equals(dataset.index)


def test_solver_pd_preprocessor_uses_entity_gains() -> None:
    dataset = pd.DataFrame(
        {
            "shoulder/position_error": [1.0, -0.5],
            "shoulder/actual_dq": [2.0, -2.0],
            "elbow/position_error": [0.5, 1.0],
            "elbow/actual_dq": [1.0, 0.0],
        }
    )

    result = SolverPDPreprocessor(
        dataset,
        kp={"shoulder": 2.0, "elbow": 4.0},
        kd={"shoulder": 0.5, "elbow": 1.0},
    ).process()

    assert result["shoulder/solver_pd"].tolist() == [1.0, 0.0]
    assert result["elbow/solver_pd"].tolist() == [1.0, 4.0]


def test_residual_torque_preprocessor_subtracts_baselines() -> None:
    dataset = pd.DataFrame(
        {
            "elbow/position_error": [0.5, 1.0, 0.0],
            "elbow/actual_dq": [1.0, -2.0, 0.0],
            "elbow/actual_effort": [10.0, 10.0, 10.0],
        }
    )

    result = ResidualTorquePreprocessor(
        dataset,
        kp={"elbow": 4.0},
        kd={"elbow": 1.0},
        dry_friction={"elbow": 0.25},
        viscous_friction={"elbow": 0.1},
    ).process()

    assert result["elbow/torque_residual"].tolist() == pytest.approx([8.65, 4.45, 10.0])


def test_preprocessor_parameters_require_matching_entities() -> None:
    with pytest.raises(ValueError, match="kd entities must match kp entities"):
        SolverPDPreprocessor(
            pd.DataFrame(),
            kp={"shoulder": 2.0},
            kd={"elbow": 1.0},
        )
