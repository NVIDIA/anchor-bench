# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared output-folder creation and config artifact writing."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import yaml
from pydantic import BaseModel

COMPLETION_MARKER = ".completed"


def get_or_create_output_folder(
    config: BaseModel,
    *,
    output_root: Optional[Union[str, Path]] = None,
    output_folder: Optional[Union[str, Path]] = None,
    initialize_config: bool = True,
) -> Tuple[Path, bool]:
    """Get or initialize an output folder and report whether it is complete."""

    if output_folder is None:
        output_root = output_root or resolve_repo_root() / "output"
        output_folder = Path(output_root).expanduser() / _hash_name(config)
    else:
        output_folder = Path(output_folder).expanduser()
    try:
        output_folder.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        if output_folder.is_dir():
            return output_folder, (output_folder / COMPLETION_MARKER).is_file()
        raise
    if initialize_config:
        write_yaml(
            output_folder / "config.yaml",
            config.model_dump(mode="json"),
        )
    return output_folder, False


def resolve_repo_root() -> Path:
    """Return the root directory of the Anchor Bench repository."""

    return Path(__file__).resolve().parents[2]


def _hash_name(config: BaseModel) -> str:
    content = config.model_dump_json().encode()
    return hashlib.sha256(content).hexdigest()[:8]


def write_yaml(path: Path, document: Any) -> None:
    """Write a YAML artifact."""

    path.write_text(
        yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
    )
