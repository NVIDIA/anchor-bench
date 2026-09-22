# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from io import BytesIO
from pathlib import Path

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
    snapshot = tmp_path / "snapshot"
    first = snapshot / "data" / "selected" / "b.parquet"
    second = snapshot / "data" / "selected" / "a.parquet"
    for path in (first, second):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(path.name, encoding="utf-8")
    seen = []

    def download(**kwargs):
        seen.append(kwargs)
        return str(snapshot)

    monkeypatch.setattr(datasets_module, "snapshot_download", download)
    source = HuggingFaceDatasetSource(
        provider="huggingface",
        repo_id="nvidia/Anchor-Lab",
        revision="abc123",
        paths=["data/selected/*.parquet"],
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
            "allow_patterns": ["data/selected/*.parquet"],
        }
    ]


def test_download_huggingface_dataset_rejects_unmatched_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        datasets_module,
        "snapshot_download",
        lambda **_: str(tmp_path),
    )
    source = HuggingFaceDatasetSource(
        provider="huggingface",
        repo_id="nvidia/Anchor-Lab",
        revision="abc123",
        paths=["data/missing/*.parquet"],
    )

    with pytest.raises(DatasetLoadError, match="did not match any files"):
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
