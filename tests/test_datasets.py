# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from io import BytesIO
from pathlib import Path
from unittest.mock import Mock

import pytest

import anchor_bench.data.datasets as datasets_module
from anchor_bench.data import download_from_huggingface
from anchor_bench.data.datasets import download_huggingface_dataset
from anchor_bench.errors import DatasetLoadError
from anchor_bench.schema.dataset import HuggingFaceDatasetSource


def test_download_huggingface_dataset_uses_snapshot_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Select immediate files before downloading, ignoring extra cached files."""

    snapshot = tmp_path / "snapshot"
    first = snapshot / "data" / "selected" / "b.parquet"
    second = snapshot / "data" / "selected" / "a.parquet"
    for path in (first, second):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(path.name, encoding="utf-8")
    seen = []

    def download(**kwargs):
        """Record the exact download selection."""

        seen.append(kwargs)
        return str(snapshot)

    list_files = Mock(
        return_value=[
            "data/selected/b.parquet",
            "data/selected/nested/large.parquet",
            "data/selected/a.parquet",
            "other/a.parquet",
        ]
    )
    (first.parent / "cached.parquet").write_text("cached", encoding="utf-8")
    monkeypatch.setattr(datasets_module, "list_repo_files", list_files)
    monkeypatch.setattr(datasets_module, "snapshot_download", download)
    source = HuggingFaceDatasetSource(
        provider="huggingface",
        repo_id="nvidia/Anchor-Lab",
        revision="abc123",
        paths=["data/selected/*.parquet", "data/selected/a.parquet"],
    )

    files = download_huggingface_dataset(source)

    assert [file.repo_path for file in files] == [
        "data/selected/a.parquet",
        "data/selected/b.parquet",
    ]
    assert [file.path for file in files] == [second, first]
    assert seen == [
        {
            "repo_id": "nvidia/Anchor-Lab",
            "repo_type": "dataset",
            "revision": "abc123",
            "allow_patterns": [
                "data/selected/a.parquet",
                "data/selected/b.parquet",
            ],
        }
    ]
    list_files.assert_called_once_with(
        repo_id="nvidia/Anchor-Lab", repo_type="dataset", revision="abc123"
    )


@pytest.mark.parametrize(
    ("pattern", "expected"),
    [
        ("data/*.parquet", ["data/a.parquet"]),
        ("data/?.parquet", ["data/a.parquet"]),
        ("data/[ab].parquet", ["data/a.parquet"]),
        (
            "data/**/*.parquet",
            ["data/a.parquet", "data/nested/b.parquet"],
        ),
        ("data/nested/**", ["data/nested/b.parquet"]),
        (
            "**/*.parquet",
            ["data/a.parquet", "data/nested/b.parquet", "root.parquet"],
        ),
    ],
)
def test_selects_repository_globs_before_downloading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pattern: str,
    expected: list[str],
) -> None:
    """Apply rooted POSIX glob semantics consistently to selection and download."""

    list_files = Mock(
        return_value=[
            "root.parquet",
            "data/nested/b.parquet",
            "data/a.parquet",
            "data/other.txt",
        ]
    )
    download = Mock(return_value=str(tmp_path))
    monkeypatch.setattr(datasets_module, "list_repo_files", list_files)
    monkeypatch.setattr(datasets_module, "snapshot_download", download)
    source = HuggingFaceDatasetSource(
        provider="huggingface",
        repo_id="nvidia/Anchor-Lab",
        revision="abc123",
        paths=[pattern],
    )

    files = download_huggingface_dataset(source)

    assert [file.repo_path for file in files] == expected
    assert [file.path for file in files] == [tmp_path / path for path in expected]
    assert download.call_args.kwargs["allow_patterns"] == expected


def test_download_huggingface_dataset_rejects_unmatched_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject an unmatched selection without starting a snapshot download."""

    download = Mock(return_value=str(tmp_path))
    monkeypatch.setattr(
        datasets_module,
        "list_repo_files",
        Mock(return_value=["other/data/missing/a.parquet"]),
    )
    monkeypatch.setattr(
        datasets_module,
        "snapshot_download",
        download,
    )
    source = HuggingFaceDatasetSource(
        provider="huggingface",
        repo_id="nvidia/Anchor-Lab",
        revision="abc123",
        paths=["data/missing/*.parquet"],
    )

    with pytest.raises(DatasetLoadError, match="did not match any files"):
        download_huggingface_dataset(source)

    download.assert_not_called()


@pytest.mark.parametrize("operation", ["list_repo_files", "snapshot_download"])
def test_wraps_huggingface_repository_errors(
    monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    """Report repository listing and download failures as dataset errors."""

    monkeypatch.setattr(
        datasets_module, "list_repo_files", Mock(return_value=["data/a.parquet"])
    )
    monkeypatch.setattr(
        datasets_module, operation, Mock(side_effect=OSError("Hub unavailable"))
    )
    source = HuggingFaceDatasetSource(
        provider="huggingface",
        repo_id="nvidia/Anchor-Lab",
        revision="abc123",
        paths=["data/*.parquet"],
    )

    with pytest.raises(DatasetLoadError, match="Could not download.*Hub unavailable"):
        download_huggingface_dataset(source)


def test_download_from_huggingface_writes_file(tmp_path: Path, monkeypatch):
    payload = b"parquet bytes"
    seen = []

    def open_url(url, *, timeout):
        seen.append((url, timeout))
        return BytesIO(payload)

    monkeypatch.setattr(datasets_module, "urlopen", open_url)
    output_folder = tmp_path / "downloads"
    url = "https://huggingface.co/datasets/example/data/resolve/main/data.parquet"

    downloaded = download_from_huggingface(url, output_folder)

    assert downloaded == output_folder / "data.parquet"
    assert downloaded.read_bytes() == payload
    assert seen == [(url, datasets_module.HUGGING_FACE_HTTP_TIMEOUT_SECONDS)]
    assert list(output_folder.glob("*.part")) == []


def test_download_from_huggingface_requires_huggingface_url(tmp_path: Path):
    with pytest.raises(DatasetLoadError, match="Expected a Hugging Face"):
        download_from_huggingface(
            "https://example.com/data.parquet",
            tmp_path / "downloads",
        )

    assert not (tmp_path / "downloads").exists()
