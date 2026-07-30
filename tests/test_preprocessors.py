from __future__ import annotations

from typing import Any, Optional, Tuple

import pandas as pd
import pytest

from anchor_bench.core import SimulationContext
from anchor_bench.train import BasePreprocessor, SimPreprocessor


class DifferencePreprocessor(BasePreprocessor):
    def process(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"position_error": self.dataset["command"] - self.dataset["position"]},
            index=self.dataset.index,
        )


class RecordingSimulationContext:
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


def test_sim_preprocessor_loads_usd_asset(dataset: pd.DataFrame) -> None:
    context: SimulationContext = RecordingSimulationContext()

    preprocessor = SimulatedFeaturePreprocessor(
        dataset,
        "assets/robots/so101.usd",
        context,
    )

    assert context.loaded_path == "assets/robots/so101.usd"
    assert preprocessor.model == "model"
    assert preprocessor.state == "state"
