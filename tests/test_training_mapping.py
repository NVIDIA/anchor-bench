# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

from anchor_bench.tasks.train import EntityMapping, TrainingTask


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
            "shoulder/torque_residual": [0.5, 0.6],
            "elbow/torque_residual": [0.7, 0.8],
            "shoulder/actual_effort": [3.0, 3.1],
            "elbow/actual_effort": [4.0, 4.1],
        },
        index=index,
    )


def test_prepares_siso_mapping(dataset: pd.DataFrame) -> None:
    """Prepare one scalar input/output sequence."""

    sequences = TrainingTask._map_features(
        dataset,
        input_features=["actual_q", "position_error"],
        output_features=["actual_effort"],
        mappings={
            "shoulder": EntityMapping(
                inputs=["shoulder"],
                outputs=["shoulder"],
            )
        },
    )

    assert [(sequence.experiment, sequence.name) for sequence in sequences] == [
        ("wave", "shoulder")
    ]
    assert sequences[0].inputs.columns.tolist() == [
        "shoulder/actual_q",
        "shoulder/position_error",
    ]
    assert sequences[0].targets.columns.tolist() == ["shoulder/actual_effort"]
    assert sequences[0].inputs.index.equals(dataset.index)
    assert sequences[0].targets.index.equals(dataset.index)

    sequences[0].inputs.iloc[0, 0] = -1.0
    sequences[0].targets.iloc[0, 0] = -1.0

    assert dataset.loc[("wave", 0), "shoulder/actual_q"] == 1.0
    assert dataset.loc[("wave", 0), "shoulder/actual_effort"] == 3.0


def test_prepares_mimo_mapping_in_feature_major_order(
    dataset: pd.DataFrame,
) -> None:
    """Prepare one multi-entity sequence without changing configured order."""

    sequences = TrainingTask._map_features(
        dataset,
        input_features=["actual_q", "position_error"],
        output_features=["actual_effort", "torque_residual"],
        mappings={
            "arm": EntityMapping(
                inputs=["shoulder", "elbow"],
                outputs=["elbow", "shoulder"],
            )
        },
    )

    assert sequences[0].inputs.columns.tolist() == [
        "shoulder/actual_q",
        "elbow/actual_q",
        "shoulder/position_error",
        "elbow/position_error",
    ]
    assert sequences[0].targets.columns.tolist() == [
        "elbow/actual_effort",
        "shoulder/actual_effort",
        "elbow/torque_residual",
        "shoulder/torque_residual",
    ]


def test_prepares_shared_siso_as_separate_ordered_sequences(
    dataset: pd.DataFrame,
) -> None:
    """Keep shared SISO mappings separate and in YAML declaration order."""

    sequences = TrainingTask._map_features(
        dataset,
        input_features=["actual_q", "position_error"],
        output_features=["actual_effort"],
        mappings={
            "elbow": EntityMapping(inputs=["elbow"], outputs=["elbow"]),
            "shoulder": EntityMapping(
                inputs=["shoulder"],
                outputs=["shoulder"],
            ),
        },
    )

    assert [sequence.name for sequence in sequences] == ["elbow", "shoulder"]
    assert [sequence.inputs.columns.tolist() for sequence in sequences] == [
        ["elbow/actual_q", "elbow/position_error"],
        ["shoulder/actual_q", "shoulder/position_error"],
    ]
    assert [sequence.targets.columns.tolist() for sequence in sequences] == [
        ["elbow/actual_effort"],
        ["shoulder/actual_effort"],
    ]


def test_orders_sequences_by_experiment_then_mapping(
    dataset: pd.DataFrame,
) -> None:
    """Preserve recurrent boundaries and deterministic shared mapping order."""

    second_experiment = dataset.copy()
    second_experiment.index = pd.MultiIndex.from_tuples(
        [("step", 0), ("step", 20_000_000)],
        names=["experiment", "time_ns"],
    )

    sequences = TrainingTask._map_features(
        pd.concat([dataset, second_experiment]),
        input_features=["actual_q"],
        output_features=["actual_effort"],
        mappings={
            "elbow": EntityMapping(inputs=["elbow"], outputs=["elbow"]),
            "shoulder": EntityMapping(
                inputs=["shoulder"],
                outputs=["shoulder"],
            ),
        },
    )

    assert [(sequence.experiment, sequence.name) for sequence in sequences] == [
        ("wave", "elbow"),
        ("wave", "shoulder"),
        ("step", "elbow"),
        ("step", "shoulder"),
    ]
    assert [
        sequence.inputs.index.get_level_values("experiment").unique().item()
        for sequence in sequences
    ] == ["wave", "wave", "step", "step"]


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
            output_features=["actual_effort"],
            mappings={
                "arm": EntityMapping(
                    inputs=["shoulder", "wrist"],
                    outputs=["wrist"],
                )
            },
        )


def test_reports_shared_missing_column_once(dataset: pd.DataFrame) -> None:
    """Deduplicate a column used as both input and target."""

    with pytest.raises(ValueError) as error:
        TrainingTask._map_features(
            dataset,
            input_features=["actual_effort"],
            output_features=["actual_effort"],
            mappings={
                "wrist": EntityMapping(
                    inputs=["wrist"],
                    outputs=["wrist"],
                )
            },
        )

    assert str(error.value).count("wrist/actual_effort") == 1


def test_rejects_different_shared_mapping_widths(
    dataset: pd.DataFrame,
) -> None:
    """Require every shared-weight sequence to have the same tensor shape."""

    with pytest.raises(
        ValueError,
        match="mapping .arm. has shape 2->1, expected 1->1",
    ):
        TrainingTask._map_features(
            dataset,
            input_features=["actual_q"],
            output_features=["actual_effort"],
            mappings={
                "shoulder": EntityMapping(
                    inputs=["shoulder"],
                    outputs=["shoulder"],
                ),
                "arm": EntityMapping(
                    inputs=["shoulder", "elbow"],
                    outputs=["elbow"],
                ),
            },
        )
