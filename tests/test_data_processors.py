# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
import pytest

from anchor_bench.core import SimulationContext
from anchor_bench.train import BaseDataProcessor
from anchor_bench.train.data_processors.gru import (
    PositionErrorDataProcessor,
    PreviousTorqueDataProcessor,
    ResidualTorqueDataProcessor,
    SolverPDDataProcessor,
)


class FakeNewtonArray:
    """Expose Newton-style host conversion for test values."""

    def __init__(self, values: list[float]) -> None:
        """Store values returned by the fake device array."""

        self.values = values

    def numpy(self) -> np.ndarray:
        """Return values as a NumPy array."""

        return np.asarray(self.values)


class DifferenceDataProcessor(BaseDataProcessor):
    def process(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"position_error": self.dataset["command"] - self.dataset["position"]},
            index=self.dataset.index,
        )


class RecordingSimulationContext(SimulationContext):
    loaded_path: Optional[str] = None

    def _load(self, usd_path: Path) -> Tuple[Any, Any]:
        """Record the USD selected by the current context."""

        self.loaded_path = str(usd_path)
        return SimpleNamespace(
            joint_label=["shoulder", "elbow"],
            joint_qd_start=FakeNewtonArray([0, 1, 2]),
            joint_target_ke=FakeNewtonArray([2.0, 4.0]),
            joint_target_kd=FakeNewtonArray([0.5, 1.0]),
            joint_friction=FakeNewtonArray([0.2, 0.25]),
            joint_damping=FakeNewtonArray([0.05, 0.1]),
        ), "state"


@pytest.fixture(autouse=True)
def simulation_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> RecordingSimulationContext:
    """Create one process-wide context for every data processor test."""

    monkeypatch.setattr(
        SimulationContext,
        "_instance",
        None,
    )
    source = tmp_path / "robot.usda"
    source.write_text("#usda 1.0", encoding="utf-8")
    return RecordingSimulationContext(str(source))


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


def test_base_data_processor_generates_aligned_features(
    dataset: pd.DataFrame,
) -> None:
    result = DifferenceDataProcessor(dataset).process()

    assert result.index.equals(dataset.index)
    assert result["position_error"].tolist() == [0.5, 0.25]


def test_base_data_processor_cannot_be_instantiated(dataset: pd.DataFrame) -> None:
    with pytest.raises(TypeError):
        BaseDataProcessor(dataset)  # pyright: ignore[reportAbstractUsage]


def test_simulation_context_cannot_be_instantiated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "robot.usda"
    source.write_text("#usda 1.0", encoding="utf-8")
    monkeypatch.setattr(SimulationContext, "_instance", None)

    with pytest.raises(TypeError):
        SimulationContext(str(source))  # pyright: ignore[reportAbstractUsage]


def test_base_data_processor_uses_current_context(
    dataset: pd.DataFrame,
    simulation_context: RecordingSimulationContext,
) -> None:
    data_processor = DifferenceDataProcessor(dataset)

    assert data_processor.simulation_context is simulation_context
    assert data_processor.simulation_context.model.joint_label == [
        "shoulder",
        "elbow",
    ]
    assert data_processor.simulation_context.state == "state"


def test_position_error_data_processor_handles_multiple_entities() -> None:
    dataset = pd.DataFrame(
        {
            "shoulder/command_q": [2.0, 1.0],
            "shoulder/actual_q": [1.0, 1.5],
            "elbow/command_q": [3.0, 4.0],
            "elbow/actual_q": [2.5, 3.0],
        },
        index=pd.Index(["sample-a", "sample-b"]),
    )

    result = PositionErrorDataProcessor(dataset).process()

    assert result.columns.tolist() == [
        "shoulder/position_error",
        "elbow/position_error",
    ]
    assert result["shoulder/position_error"].tolist() == [1.0, -0.5]
    assert result["elbow/position_error"].tolist() == [0.5, 1.0]
    assert result.index.equals(dataset.index)


def test_solver_pd_data_processor_uses_entity_gains() -> None:
    dataset = pd.DataFrame(
        {
            "shoulder/position_error": [1.0, -0.5],
            "shoulder/actual_dq": [2.0, -2.0],
            "elbow/position_error": [0.5, 1.0],
            "elbow/actual_dq": [1.0, 0.0],
        },
        index=pd.Index(["sample-a", "sample-b"]),
    )

    result = SolverPDDataProcessor(dataset).process()

    assert result["shoulder/solver_pd"].tolist() == [1.0, 0.0]
    assert result["elbow/solver_pd"].tolist() == [1.0, 4.0]
    assert result.index.equals(dataset.index)


def test_residual_torque_data_processor_subtracts_baselines() -> None:
    dataset = pd.DataFrame(
        {
            "elbow/position_error": [0.5, 1.0, 0.0],
            "elbow/actual_dq": [1.0, -2.0, 0.0],
            "elbow/actual_effort": [10.0, 10.0, 10.0],
        },
        index=pd.Index(["sample-a", "sample-b", "sample-c"]),
    )

    result = ResidualTorqueDataProcessor(dataset).process()

    assert result["elbow/torque_residual"].tolist() == pytest.approx([8.65, 4.45, 10.0])
    assert result.index.equals(dataset.index)


def test_data_processor_entity_must_match_usd_joint() -> None:
    """Reject dataset entities that do not identify a USD joint."""

    dataset = pd.DataFrame(
        {
            "wrist/position_error": [0.5],
            "wrist/actual_dq": [1.0],
        }
    )

    with pytest.raises(ValueError, match="wrist"):
        SolverPDDataProcessor(dataset).process()


def test_residual_torque_requires_actual_effort_column() -> None:
    dataset = pd.DataFrame(
        {
            "elbow/position_error": [0.5],
            "elbow/actual_dq": [1.0],
        }
    )

    data_processor = ResidualTorqueDataProcessor(dataset)

    with pytest.raises(ValueError, match="elbow/actual_effort"):
        data_processor.process()


def test_previous_torque_shifts_within_each_experiment() -> None:
    """Reset shifted normalization data at every experiment boundary."""

    index = pd.MultiIndex.from_tuples(
        [("first", 0), ("first", 1), ("second", 0), ("second", 1)],
        names=["experiment", "time_ns"],
    )
    dataset = pd.DataFrame(
        {
            "shoulder/actual_effort": [1.0, 2.0, 10.0, 20.0],
            "elbow/actual_effort": [3.0, 4.0, 30.0, 40.0],
        },
        index=index,
    )

    result = PreviousTorqueDataProcessor(dataset).process()

    assert result["shoulder/previous_torque"].tolist() == [0.0, 1.0, 0.0, 10.0]
    assert result["elbow/previous_torque"].tolist() == [0.0, 3.0, 0.0, 30.0]
    assert result.index.equals(dataset.index)
