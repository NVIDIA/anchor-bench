# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Download dataset files from Hugging Face."""

from __future__ import annotations

import logging
from pathlib import Path
from shutil import copyfileobj
from tempfile import NamedTemporaryFile
from typing import Union
from urllib.parse import unquote, urlparse
from urllib.request import urlopen

from anchor_bench.errors import DatasetLoadError

HUGGING_FACE_HTTP_TIMEOUT_SECONDS = 60.0

logger = logging.getLogger(__name__)


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
