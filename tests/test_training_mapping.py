import pandas as pd
import pytest

from anchor_bench.train.task import TrainingTask


@pytest.fixture
def dataset() -> pd.DataFrame:
    """Create aligned entity features for mapping tests."""

    index = pd.MultiIndex.from_tuples(
        [("wave", 0), ("wave", 20_000_000)],
        names=["experiment", "time_ns"],
    )
    return pd.DataFrame(
        {
            "shoulder/actual_q": [1.0, 1.1],
            "elbow/actual_q": [2.0, 2.1],
            "shoulder/position_error": [0.1, 0.2],
            "elbow/position_error": [0.3, 0.4],
            "shoulder/actual_effort": [3.0, 3.1],
            "elbow/actual_effort": [4.0, 4.1],
        },
        index=index,
    )


def test_maps_features_across_robot_entities(
    dataset: pd.DataFrame,
) -> None:
    """Map inputs feature-first and targets in output-entity order."""

    inputs, targets = TrainingTask._map_features(
        dataset,
        input_features=["actual_q", "position_error"],
        output_feature="actual_effort",
        input_entities=["shoulder", "elbow"],
        output_entities=["elbow", "shoulder"],
    )

    assert inputs.columns.tolist() == [
        "shoulder/actual_q",
        "elbow/actual_q",
        "shoulder/position_error",
        "elbow/position_error",
    ]
    assert targets.columns.tolist() == [
        "elbow/actual_effort",
        "shoulder/actual_effort",
    ]
    assert inputs.index.equals(dataset.index)
    assert targets.index.equals(dataset.index)


def test_rejects_missing_mapped_columns(dataset: pd.DataFrame) -> None:
    """Report every mapped column absent from the dataset."""

    with pytest.raises(
        ValueError,
        match=(
            "wrist/actual_q, shoulder/command_q, wrist/command_q, wrist/actual_effort"
        ),
    ):
        TrainingTask._map_features(
            dataset,
            input_features=["actual_q", "command_q"],
            output_feature="actual_effort",
            input_entities=["shoulder", "wrist"],
            output_entities=["wrist"],
        )
