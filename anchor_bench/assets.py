# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve declarative asset references into local filesystem paths."""

from __future__ import annotations

import logging
from pathlib import Path

from newton.utils import download_asset

from anchor_bench.errors import AssetResolutionError
from anchor_bench.schema.usd import LocalUsdSource, UsdSource

logger = logging.getLogger(__name__)


def resolve_usd_path(source: UsdSource) -> Path:
    """Resolve one composed USD source into a local filesystem path."""

    if isinstance(source, LocalUsdSource):
        return Path(source.path).expanduser().resolve()

    logger.info(
        "Resolving Newton asset folder=%s path=%s ref=%s",
        source.folder,
        source.path,
        source.ref,
    )
    try:
        asset_root = download_asset(source.folder, ref=source.ref)
    except (ImportError, OSError, RuntimeError) as error:
        raise AssetResolutionError(
            f"Could not download Newton asset folder {source.folder!r} at "
            f"revision {source.ref}: {error}"
        ) from error

    root = asset_root.resolve()
    resolved = (root / source.path).resolve()
    if not resolved.is_relative_to(root):
        raise AssetResolutionError(
            f"Newton asset path escapes downloaded folder: {source.path!r}"
        )
    if not resolved.is_file():
        raise AssetResolutionError(
            f"Newton asset file does not exist: {source.folder}/{source.path} "
            f"at revision {source.ref}"
        )
    logger.debug("Resolved Newton asset path=%s", resolved)
    return resolved
