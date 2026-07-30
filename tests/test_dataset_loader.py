from io import BytesIO
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


def test_loads_and_aligns_local_parquet_file(tmp_path: Path) -> None:
    dataset = pd.DataFrame(
        _rows("actual_q", [0, 20_000_000], [1.0, 2.0])
        + _rows("command_q", [0, 20_000_000], [2.0, 4.0])
    )
    source = _write_dataset(tmp_path, dataset)

    result = DatasetLoader.load(str(source), reference_field="actual_q")

    assert result.columns.tolist() == ["actual_q", "command_q"]
    assert result.index.get_level_values("time_ns").tolist() == [0, 20_000_000]


def test_loads_direct_http_url(monkeypatch) -> None:
    dataset = pd.DataFrame(_rows("actual_q", [0], [1.0]))
    payload = BytesIO()
    dataset.to_parquet(payload, index=False)
    seen = []

    def open_url(source, *, timeout):
        """Return remote Parquet bytes and record the configured timeout."""

        seen.append((source, timeout))
        return BytesIO(payload.getvalue())

    monkeypatch.setattr(loader_module, "urlopen", open_url)

    result = DatasetLoader.load(
        "https://example.com/dataset.parquet",
        reference_field="actual_q",
        http_timeout_seconds=2.5,
    )

    assert result["actual_q"].tolist() == [1.0]
    assert seen == [("https://example.com/dataset.parquet", 2.5)]


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

    result = DatasetLoader.load(source, reference_field="actual_q")

    assert result["actual_q"].tolist() == [1.0]
    assert seen == [source]


def test_rejects_missing_local_file(tmp_path: Path) -> None:
    source = tmp_path / "missing.parquet"

    with pytest.raises(DatasetLoadError, match="does not exist or is not a file"):
        DatasetLoader.load(str(source), reference_field="actual_q")


@pytest.mark.parametrize(
    "source",
    [
        "s3://bucket/dataset.parquet",
        "x://host/dataset.parquet",
    ],
)
def test_rejects_unsupported_url_scheme(source: str) -> None:
    with pytest.raises(DatasetLoadError, match="Unsupported dataset URL scheme"):
        DatasetLoader.load(
            source,
            reference_field="actual_q",
        )


def test_wraps_parquet_read_errors(tmp_path: Path) -> None:
    source = tmp_path / "invalid.parquet"
    source.write_text("not parquet", encoding="utf-8")

    with pytest.raises(DatasetLoadError, match="Could not load Parquet dataset"):
        DatasetLoader.load(str(source), reference_field="actual_q")


def test_aligns_every_signal_to_recorded_position_timestamps(
    tmp_path: Path,
) -> None:
    dataset = pd.DataFrame(
        _rows("actual_q", [0, 10_100_000, 20_000_000], [1.0, 1.5, 2.0])
        + _rows("dq", [0, 20_000_000], [0.0, 1.0])
        + _rows("command_q", [0, 20_000_000], [2.0, 4.0])
    )
    source = _write_dataset(tmp_path, dataset)

    result = DatasetLoader.load(str(source), reference_field="actual_q")

    assert result.index.get_level_values("time_ns").tolist() == [
        0,
        10_100_000,
        20_000_000,
    ]
    assert result["actual_q"].tolist() == [1.0, 1.5, 2.0]
    assert result["dq"].tolist() == pytest.approx([0.0, 0.505, 1.0])
    assert result["command_q"].tolist() == pytest.approx([2.0, 3.01, 4.0])


def test_accepts_matching_joint_reference_clocks(tmp_path: Path) -> None:
    dataset = pd.DataFrame(
        _rows("elbow/actual_q", [0, 10_000_000], [1.0, 1.5])
        + _rows("shoulder/actual_q", [0, 10_000_000], [2.0, 2.5])
    )
    source = _write_dataset(tmp_path, dataset)

    result = DatasetLoader.load(str(source), reference_field="actual_q")

    assert result.columns.tolist() == [
        "elbow/actual_q",
        "shoulder/actual_q",
    ]


def test_rejects_mismatched_joint_reference_clocks(tmp_path: Path) -> None:
    dataset = pd.DataFrame(
        _rows("elbow/actual_q", [0, 10_000_000], [1.0, 1.5])
        + _rows("shoulder/actual_q", [0, 11_000_000], [2.0, 2.5])
    )
    source = _write_dataset(tmp_path, dataset)

    with pytest.raises(DatasetAlignmentError, match="do not share"):
        DatasetLoader.load(str(source), reference_field="actual_q")


def test_crops_reference_timestamps_to_signal_intersection(
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

    result = DatasetLoader.load(str(source), reference_field="actual_q")

    assert result.index.get_level_values("time_ns").tolist() == [
        10_000_000,
        20_000_000,
    ]
    assert result["command_q"].tolist() == [3.0, 4.0]
