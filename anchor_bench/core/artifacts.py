"""Shared output-folder creation and input artifact writing."""

from __future__ import annotations

import hashlib
from collections.abc import Hashable, Mapping
from pathlib import Path
from typing import Any, Optional, Union

import yaml
from pydantic import BaseModel


def create_output_folder(
    hash_input: Hashable,
    *,
    config: BaseModel,
    input_data: Mapping[str, Any],
    output_root: Optional[Union[str, Path]] = None,
) -> Path:
    """Create a content-addressed output folder and record its run inputs."""

    output_root = output_root or resolve_repo_root() / "output"
    folder_name = _hash_name(hash_input)
    output_folder = Path(output_root).expanduser() / folder_name
    output_folder.mkdir(parents=True, exist_ok=False)
    _write_yaml(
        output_folder / "config.yaml",
        config.model_dump(mode="json"),
    )
    _write_yaml(output_folder / "input_data.yaml", dict(input_data))
    return output_folder


def resolve_repo_root() -> Path:
    """Return the root directory of the Anchor Bench repository."""

    return Path(__file__).resolve().parents[2]


def _hash_name(value: Hashable) -> str:
    hash(value)
    type_name = f"{type(value).__module__}.{type(value).__qualname__}"
    content = f"{type_name}:{value!r}".encode()
    return hashlib.sha256(content).hexdigest()[:8]


def _write_yaml(path: Path, document: Any) -> None:
    path.write_text(
        yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
    )
