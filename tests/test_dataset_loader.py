# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

import anchor_bench.data.loader as loader_module
from anchor_bench.data import DatasetLoader
from anchor_bench.data.datasets import HuggingFaceDatasetFile
from anchor_bench.errors import DatasetAlignmentError, DatasetLoadError
from anchor_bench.mlops import MLOpsRun
from anchor_bench.schema.dataset import HuggingFaceDatasetSource


def _rows(field, timestamps, values):
    """Build long-form signal rows for a test dataset."""

    return [
        {
            "time_ns": timestamp,
            "experiment": "episode",
            "field": field,
            "value": value,
        }
        for timestamp, value in zip(timestamps, values)
    ]


def _write_dataset(tmp_path: Path, dataset: pd.DataFrame) -> Path:
    """Write a test dataset to Parquet and return its path."""

    source = tmp_path / "dataset.parquet"
    dataset.to_parquet(source, index=False)
    return source


def test_loads_and_resamples_local_parquet_file(tmp_path: Path) -> None:
    dataset = pd.DataFrame(
        _rows("actual_q", [0, 20_000_000], [1.0, 2.0])
        + _rows("command_q", [0, 20_000_000], [2.0, 4.0])
    )
    source = _write_dataset(tmp_path, dataset)

    result = DatasetLoader.load(source, dt=0.02)

    assert result.columns.tolist() == ["actual_q", "command_q"]
    assert result.index.get_level_values("time_ns").tolist() == [0, 20_000_000]


def test_loads_multiple_local_parquet_files_from_glob(tmp_path: Path) -> None:
    for name, experiment, value in (
        ("b.parquet", "episode-b", 2.0),
        ("a.parquet", "episode-a", 1.0),
    ):
        dataset = pd.DataFrame(_rows("actual_q", [0], [value]))
        dataset["experiment"] = experiment
        dataset.to_parquet(tmp_path / name, index=False)

    source_pattern = str(tmp_path / "*.parquet")
    tracker = Mock(spec=MLOpsRun)

    result = DatasetLoader.load(
        source_pattern,
        dt=0.02,
        dataset_tracker=tracker,
    )

    assert result.index.get_level_values("experiment").unique().tolist() == [
        "episode-a",
        "episode-b",
    ]
    assert result["actual_q"].tolist() == [1.0, 2.0]
    assert [call.kwargs["source"] for call in tracker.log_dataset.call_args_list] == [
        str(tmp_path / "a.parquet"),
        str(tmp_path / "b.parquet"),
    ]
    assert [call.kwargs["name"] for call in tracker.log_dataset.call_args_list] == [
        "dataset-a",
        "dataset-b",
    ]
    assert [
        call.kwargs["metadata"]["source_pattern"]
        for call in tracker.log_dataset.call_args_list
    ] == [source_pattern, source_pattern]


def test_rejects_local_glob_without_file_matches(tmp_path: Path) -> None:
    with pytest.raises(DatasetLoadError, match="glob did not match any files"):
        DatasetLoader.load(str(tmp_path / "*.parquet"), dt=0.02)


def test_logs_dataset_lineage_and_opt_in_upload(tmp_path: Path) -> None:
    dataset = pd.DataFrame(
        _rows("actual_q", [0, 20_000_000], [1.0, 2.0])
        + _rows("command_q", [0, 20_000_000], [2.0, 4.0])
    )
    source = _write_dataset(tmp_path, dataset)
    tracker = Mock(spec=MLOpsRun)

    DatasetLoader.load(
        source,
        dt=0.02,
        dataset_tracker=tracker,
        dataset_name="training-dataset",
        dataset_context="training",
        upload_dataset=True,
    )

    tracker.log_dataset.assert_called_once()
    reported_dataset = tracker.log_dataset.call_args.args[0]
    options = tracker.log_dataset.call_args.kwargs
    pd.testing.assert_frame_equal(reported_dataset, dataset)
    assert options == {
        "source": str(source),
        "digest": hashlib.sha256(source.read_bytes()).hexdigest(),
        "name": "training-dataset",
        "context": "training",
        "path": source,
        "metadata": {
            "digest_algorithm": "sha256",
            "format": "parquet",
            "size_bytes": source.stat().st_size,
        },
        "upload": True,
    }


def test_downloads_huggingface_url_before_loading(monkeypatch) -> None:
    dataset = pd.DataFrame(_rows("actual_q", [0], [1.0]))
    seen = []
    tracker = Mock(spec=MLOpsRun)

    def download(source, output_folder, *, http_timeout_seconds):
        """Write remote Parquet data and record the downloader arguments."""

        output_folder = Path(output_folder)
        destination = output_folder / "dataset.parquet"
        dataset.to_parquet(destination, index=False)
        seen.append((source, output_folder, http_timeout_seconds))
        return destination

    monkeypatch.setattr(loader_module, "download_from_huggingface", download)

    source = "https://huggingface.co/datasets/example/resolve/main/dataset.parquet"
    result = DatasetLoader.load(
        source,
        dt=0.02,
        http_timeout_seconds=2.5,
        dataset_tracker=tracker,
    )

    assert result["actual_q"].tolist() == [1.0]
    assert seen[0][0] == (
        "https://huggingface.co/datasets/example/resolve/main/dataset.parquet"
    )
    assert seen[0][2] == 2.5
    assert not seen[0][1].exists()
    assert tracker.log_dataset.call_args.kwargs["source"] == source


def test_downloads_structured_huggingface_source_and_logs_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "b.parquet"
    second = tmp_path / "a.parquet"
    for path, experiment, value in (
        (first, "episode-b", 2.0),
        (second, "episode-a", 1.0),
    ):
        dataset = pd.DataFrame(_rows("actual_q", [0], [value]))
        dataset["experiment"] = experiment
        dataset.to_parquet(path, index=False)
    source = HuggingFaceDatasetSource(
        provider="huggingface",
        repo_id="nvidia/Anchor-Lab",
        revision="abc123",
        paths=["data/selected/*.parquet"],
    )
    tracker = Mock(spec=MLOpsRun)

    monkeypatch.setattr(
        loader_module,
        "download_huggingface_dataset",
        lambda _: [
            HuggingFaceDatasetFile(
                path=second,
                repo_path="data/selected/a.parquet",
            ),
            HuggingFaceDatasetFile(
                path=first,
                repo_path="data/selected/b.parquet",
            ),
        ],
    )

    result = DatasetLoader.load(source, dt=0.02, dataset_tracker=tracker)

    assert result.index.get_level_values("experiment").unique().tolist() == [
        "episode-a",
        "episode-b",
    ]
    calls = tracker.log_dataset.call_args_list
    assert [call.kwargs["source"] for call in calls] == [
        "hf://datasets/nvidia/Anchor-Lab@abc123/data/selected/a.parquet",
        "hf://datasets/nvidia/Anchor-Lab@abc123/data/selected/b.parquet",
    ]
    assert [call.kwargs["metadata"]["repo_path"] for call in calls] == [
        "data/selected/a.parquet",
        "data/selected/b.parquet",
    ]
    assert all(call.kwargs["metadata"]["provider"] == "huggingface" for call in calls)


def test_rejects_globbing_huggingface_urls() -> None:
    with pytest.raises(DatasetLoadError, match="URL.*do not support globbing"):
        DatasetLoader.load(
            "https://huggingface.co/datasets/example/resolve/main/*.parquet",
            dt=0.02,
        )


@pytest.mark.parametrize(
    "source",
    [
        r"C:\Users\alice\dataset.parquet",
        "C:/Users/alice/dataset.parquet",
    ],
)
def test_treats_windows_drive_paths_as_local(monkeypatch, source: str) -> None:
    dataset = pd.DataFrame(_rows("actual_q", [0], [1.0]))
    seen = []

    def local_path(candidate):
        seen.append(candidate)
        return Path("dataset.parquet")

    monkeypatch.setattr(DatasetLoader, "_local_path", local_path)
    monkeypatch.setattr(pd, "read_parquet", lambda _: dataset)

    result = DatasetLoader.load(source, dt=0.02)

    assert result["actual_q"].tolist() == [1.0]
    assert seen == [source]


def test_rejects_missing_local_file(tmp_path: Path) -> None:
    source = tmp_path / "missing.parquet"

    with pytest.raises(DatasetLoadError, match="does not exist or is not a file"):
        DatasetLoader.load(str(source), dt=0.02)


@pytest.mark.parametrize(
    "source",
    [
        "s3://bucket/dataset.parquet",
        "x://host/dataset.parquet",
    ],
)
def test_non_http_source_is_treated_as_local_path(source: str) -> None:
    with pytest.raises(DatasetLoadError, match="does not exist or is not a file"):
        DatasetLoader.load(
            source,
            dt=0.02,
        )


def test_wraps_parquet_read_errors(tmp_path: Path) -> None:
    source = tmp_path / "invalid.parquet"
    source.write_text("not parquet", encoding="utf-8")

    with pytest.raises(DatasetLoadError, match="Could not load Parquet dataset"):
        DatasetLoader.load(str(source), dt=0.02)


@pytest.mark.parametrize("dt", [0.0, -0.001, float("nan"), 0.0000000001])
def test_rejects_invalid_resampling_interval(tmp_path: Path, dt: float) -> None:
    dataset = pd.DataFrame(_rows("actual_q", [0], [1.0]))
    source = _write_dataset(tmp_path, dataset)

    with pytest.raises(DatasetAlignmentError, match="dt must"):
        DatasetLoader.load(str(source), dt=dt)


def test_resamples_every_signal_to_uniform_timestamps(
    tmp_path: Path,
) -> None:
    dataset = pd.DataFrame(
        _rows("actual_q", [0, 10_100_000, 20_000_000], [1.0, 1.5, 2.0])
        + _rows("dq", [0, 20_000_000], [0.0, 1.0])
        + _rows("command_q", [0, 20_000_000], [2.0, 4.0])
    )
    source = _write_dataset(tmp_path, dataset)

    result = DatasetLoader.load(str(source), dt=0.01)

    assert result.index.get_level_values("time_ns").tolist() == [
        0,
        10_000_000,
        20_000_000,
    ]
    assert result["actual_q"].tolist() == pytest.approx([1.0, 1.49504950495, 2.0])
    assert result["dq"].tolist() == pytest.approx([0.0, 0.5, 1.0])
    assert result["command_q"].tolist() == pytest.approx([2.0, 3.0, 4.0])


def test_resamples_joint_signals_with_independent_timestamps(tmp_path: Path) -> None:
    dataset = pd.DataFrame(
        _rows("elbow/actual_q", [0, 10_000_000], [1.0, 1.5])
        + _rows("shoulder/actual_q", [0, 11_000_000], [2.0, 2.5])
    )
    source = _write_dataset(tmp_path, dataset)

    result = DatasetLoader.load(str(source), dt=0.01)

    assert result.columns.tolist() == [
        "elbow/actual_q",
        "shoulder/actual_q",
    ]
    assert result.index.get_level_values("time_ns").tolist() == [0, 10_000_000]
    assert result["shoulder/actual_q"].tolist() == pytest.approx([2.0, 2.45454545455])


def test_maps_parquet_joint_names_before_resampling(tmp_path: Path) -> None:
    dataset = pd.DataFrame(
        _rows("recorded_hip/actual_q", [0, 10_000_000], [1.0, 1.5])
        + _rows("knee/actual_q", [0, 10_000_000], [2.0, 2.5])
        + _rows("recorded_hip/command_q", [0, 10_000_000], [1.1, 1.6])
    )
    source = _write_dataset(tmp_path, dataset)

    result = DatasetLoader.load(
        source,
        dt=0.01,
        joint_name_mapping={"recorded_hip": "/robot/joints/hip"},
    )

    assert result.columns.tolist() == [
        "/robot/joints/hip/actual_q",
        "knee/actual_q",
        "/robot/joints/hip/command_q",
    ]


def test_maps_parquet_feature_names_before_resampling(tmp_path: Path) -> None:
    dataset = pd.DataFrame(
        _rows("recorded_hip/dq", [0, 10_000_000], [1.0, 1.5])
        + _rows("recorded_hip/tau_est", [0, 10_000_000], [2.0, 2.5])
    )
    source = _write_dataset(tmp_path, dataset)

    result = DatasetLoader.load(
        source,
        dt=0.01,
        joint_name_mapping={"recorded_hip": "/robot/joints/hip"},
        feature_name_mapping={"dq": "actual_dq", "tau_est": "actual_effort"},
    )

    assert result.columns.tolist() == [
        "/robot/joints/hip/actual_dq",
        "/robot/joints/hip/actual_effort",
    ]


def test_maps_each_distinct_field_name_once() -> None:
    dataset = pd.DataFrame(
        _rows("recorded_hip/dq", [0, 10_000_000], [1.0, 1.5])
        + _rows("recorded_hip/tau_est", [0, 10_000_000], [2.0, 2.5])
    )
    joint_name_mapping = Mock(wraps={"recorded_hip": "hip"})
    feature_name_mapping = Mock(wraps={"dq": "actual_dq", "tau_est": "actual_effort"})

    result = DatasetLoader._map_field_names(
        dataset,
        joint_name_mapping,
        feature_name_mapping,
    )

    assert result["field"].tolist() == [
        "hip/actual_dq",
        "hip/actual_dq",
        "hip/actual_effort",
        "hip/actual_effort",
    ]
    assert joint_name_mapping.get.call_count == 2
    assert feature_name_mapping.get.call_count == 2


def test_resamples_experiments_with_different_fields(tmp_path: Path) -> None:
    first = pd.DataFrame(
        _rows("actual_q", [10_000_000, 0], [2.0, 1.0])
        + _rows("command_q", [10_000_000, 0], [4.0, 2.0])
    )
    second = pd.DataFrame(_rows("actual_q", [10_000_000, 0], [6.0, 5.0]))
    second["experiment"] = "episode-2"
    source = _write_dataset(tmp_path, pd.concat([first, second], ignore_index=True))

    result = DatasetLoader.load(source, dt=0.01)

    assert result.index.tolist() == [
        ("episode", 0),
        ("episode", 10_000_000),
        ("episode-2", 0),
        ("episode-2", 10_000_000),
    ]
    assert result.columns.tolist() == ["actual_q", "command_q"]
    assert result["actual_q"].tolist() == [1.0, 2.0, 5.0, 6.0]
    assert result["command_q"].iloc[:2].tolist() == [2.0, 4.0]
    assert result["command_q"].iloc[2:].isna().all()


def test_rejects_joint_name_mapping_field_collisions(tmp_path: Path) -> None:
    dataset = pd.DataFrame(
        _rows("recorded_hip/actual_q", [0, 10_000_000], [1.0, 1.5])
        + _rows("hip/actual_q", [0, 10_000_000], [2.0, 2.5])
    )
    source = _write_dataset(tmp_path, dataset)

    with pytest.raises(
        DatasetAlignmentError,
        match="mapping produces conflicting fields: hip/actual_q",
    ):
        DatasetLoader.load(
            source,
            dt=0.01,
            joint_name_mapping={"recorded_hip": "hip"},
        )


def test_allows_mapped_field_names_in_different_experiments(tmp_path: Path) -> None:
    first = pd.DataFrame(_rows("recorded_hip/actual_q", [0], [1.0]))
    second = pd.DataFrame(_rows("hip/actual_q", [0], [2.0]))
    second["experiment"] = "episode-2"
    source = _write_dataset(tmp_path, pd.concat([first, second], ignore_index=True))

    result = DatasetLoader.load(
        source,
        dt=0.01,
        joint_name_mapping={"recorded_hip": "hip"},
    )

    assert result.index.tolist() == [("episode", 0), ("episode-2", 0)]
    assert result.columns.tolist() == ["hip/actual_q"]
    assert result["hip/actual_q"].tolist() == [1.0, 2.0]


def test_rejects_feature_name_mapping_field_collisions(tmp_path: Path) -> None:
    dataset = pd.DataFrame(
        _rows("hip/dq", [0, 10_000_000], [1.0, 1.5])
        + _rows("hip/actual_dq", [0, 10_000_000], [2.0, 2.5])
    )
    source = _write_dataset(tmp_path, dataset)

    with pytest.raises(
        DatasetAlignmentError,
        match="mapping produces conflicting fields: hip/actual_dq",
    ):
        DatasetLoader.load(
            source,
            dt=0.01,
            feature_name_mapping={"dq": "actual_dq"},
        )


def test_starts_uniform_grid_at_signal_intersection(tmp_path: Path) -> None:
    dataset = pd.DataFrame(
        _rows("elbow/actual_q", [0, 10_000_000], [1.0, 1.5])
        + _rows("shoulder/actual_q", [1_000_000, 11_000_000], [2.0, 2.5])
    )
    source = _write_dataset(tmp_path, dataset)

    result = DatasetLoader.load(str(source), dt=0.005)

    assert result.index.get_level_values("time_ns").tolist() == [
        1_000_000,
        6_000_000,
    ]


def test_crops_uniform_grid_to_signal_intersection(
    tmp_path: Path,
) -> None:
    dataset = pd.DataFrame(
        _rows(
            "actual_q",
            [0, 10_000_000, 20_000_000, 30_000_000],
            [1.0, 1.5, 2.0, 2.5],
        )
        + _rows("command_q", [10_000_000, 20_000_000], [3.0, 4.0])
    )
    source = _write_dataset(tmp_path, dataset)

    result = DatasetLoader.load(str(source), dt=0.01)

    assert result.index.get_level_values("time_ns").tolist() == [
        10_000_000,
        20_000_000,
    ]
    assert result["command_q"].tolist() == [3.0, 4.0]
