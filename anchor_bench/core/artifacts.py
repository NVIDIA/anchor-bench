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
) -> Tuple[Path, bool]:
    """Get or initialize an output folder and report whether it is complete."""

    output_root = output_root or resolve_repo_root() / "output"
    folder_name = _hash_name(config)
    output_folder = Path(output_root).expanduser() / folder_name
    try:
        output_folder.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        if output_folder.is_dir():
            return output_folder, (output_folder / COMPLETION_MARKER).is_file()
        raise
    _write_yaml(
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


def _write_yaml(path: Path, document: Any) -> None:
    path.write_text(
        yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
    )
