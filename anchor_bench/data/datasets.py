# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Download dataset files from Hugging Face."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from shutil import copyfileobj
from tempfile import NamedTemporaryFile
from typing import Union
from urllib.parse import unquote, urlparse
from urllib.request import urlopen

from huggingface_hub import snapshot_download

from anchor_bench.errors import DatasetLoadError
from anchor_bench.schema.dataset import HuggingFaceDatasetSource

HUGGING_FACE_HTTP_TIMEOUT_SECONDS = 60.0

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HuggingFaceDatasetFile:
    """One cached dataset file and its path inside the Hub repository."""

    path: Path
    repo_path: str


def download_huggingface_dataset(
    source: HuggingFaceDatasetSource,
) -> list[HuggingFaceDatasetFile]:
    """Download and return the files selected by a structured Hub source."""

    try:
        snapshot = Path(
            snapshot_download(
                repo_id=source.repo_id,
                repo_type="dataset",
                revision=source.revision,
                allow_patterns=source.paths,
            )
        )
        matches = {
            path.relative_to(snapshot).as_posix(): path
            for pattern in source.paths
            for path in snapshot.glob(pattern)
            if path.is_file()
        }
    except Exception as error:
        raise DatasetLoadError(
            "Could not download Hugging Face dataset "
            f"{source.repo_id}@{source.revision}: {error}."
        ) from error
    if not matches:
        patterns = ", ".join(source.paths)
        raise DatasetLoadError(
            "Hugging Face dataset paths did not match any files in "
            f"{source.repo_id}@{source.revision}: {patterns}."
        )
    files = [
        HuggingFaceDatasetFile(path=matches[repo_path], repo_path=repo_path)
        for repo_path in sorted(matches)
    ]
    logger.info(
        "Downloaded Hugging Face dataset repo_id=%s revision=%s files=%d",
        source.repo_id,
        source.revision,
        len(files),
    )
    return files


def download_from_huggingface(
    url: str,
    output_folder: Union[str, Path],
    *,
    http_timeout_seconds: float = HUGGING_FACE_HTTP_TIMEOUT_SECONDS,
) -> Path:
    """Download one Hugging Face dataset file into ``output_folder``."""

    parsed_url = urlparse(url)
    if parsed_url.scheme not in ("http", "https") or parsed_url.hostname not in (
        "huggingface.co",
        "www.huggingface.co",
    ):
        raise DatasetLoadError(f"Expected a Hugging Face HTTP(S) URL: {url}.")

    filename = Path(unquote(parsed_url.path)).name
    if not filename:
        raise DatasetLoadError(f"Hugging Face URL does not identify a file: {url}.")

    folder = Path(output_folder).expanduser()
    destination = folder / filename
    temporary_path = None
    try:
        logger.info("Downloading dataset url=%s destination=%s", url, destination)
        folder.mkdir(parents=True, exist_ok=True)
        with (
            urlopen(
                url,
                timeout=http_timeout_seconds,
            ) as response,
            NamedTemporaryFile(
                dir=folder,
                prefix=f".{filename}.",
                suffix=".part",
                delete=False,
            ) as stream,
        ):
            temporary_path = Path(stream.name)
            copyfileobj(response, stream)
        temporary_path.replace(destination)
    except Exception as error:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise DatasetLoadError(
            f"Could not download Hugging Face dataset from {url}: {error}."
        ) from error
    logger.info(
        "Downloaded dataset destination=%s size_bytes=%d",
        destination,
        destination.stat().st_size,
    )
    return destination
