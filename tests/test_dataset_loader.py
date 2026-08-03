from pathlib import Path

import pandas as pd
import pytest

import anchor_bench.data.loader as loader_module
from anchor_bench.data import DatasetLoader
from anchor_bench.errors import DatasetAlignmentError, DatasetLoadError


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


def test_downloads_huggingface_url_before_loading(monkeypatch) -> None:
    dataset = pd.DataFrame(_rows("actual_q", [0], [1.0]))
    seen = []

    def download(source, output_folder, *, http_timeout_seconds):
        """Write remote Parquet data and record the downloader arguments."""

        output_folder = Path(output_folder)
        destination = output_folder / "dataset.parquet"
        dataset.to_parquet(destination, index=False)
        seen.append((source, output_folder, http_timeout_seconds))
        return destination

    monkeypatch.setattr(loader_module, "download_from_huggingface", download)

    result = DatasetLoader.load(
        "https://huggingface.co/datasets/example/resolve/main/dataset.parquet",
        dt=0.02,
        http_timeout_seconds=2.5,
    )

    assert result["actual_q"].tolist() == [1.0]
    assert seen[0][0] == (
        "https://huggingface.co/datasets/example/resolve/main/dataset.parquet"
    )
    assert seen[0][2] == 2.5
    assert not seen[0][1].exists()


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
