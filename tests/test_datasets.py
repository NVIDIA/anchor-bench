from io import BytesIO
from pathlib import Path

import pytest

import anchor_bench.data.datasets as datasets_module
from anchor_bench.data import download_from_huggingface
from anchor_bench.errors import DatasetLoadError


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
