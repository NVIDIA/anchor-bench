"""Host-safe inspection of schema-v3 TorchScript metadata archives."""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from anchor_bench.core.hashing import sha256_hex
from anchor_bench.training.metadata import RuntimeGRUMetadata


@dataclass(frozen=True)
class EmbeddedRuntimeMetadata:
    """Validated metadata plus the exact bytes authenticated by promotion."""

    metadata: RuntimeGRUMetadata
    raw_bytes: bytes
    sha256: str


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate object keys while decoding embedded JSON."""

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate metadata JSON key: {key!r}")
        result[key] = value
    return result


def _metadata_member(names: list[str]) -> str:
    """Find the one TorchScript extra-file metadata member."""

    matches = [
        name
        for name in names
        if PurePosixPath(name).parts[-2:] == ("extra", "metadata.json")
    ]
    if len(matches) != 1:
        raise ValueError(
            "TorchScript archive must contain exactly one extra/metadata.json"
        )
    return matches[0]


def read_runtime_metadata_archive(path: str | Path) -> EmbeddedRuntimeMetadata:
    """Read embedded metadata without importing Torch."""

    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"network archive is not a regular file: {source}")
    try:
        with zipfile.ZipFile(source) as archive:
            member = _metadata_member(archive.namelist())
            raw = archive.read(member)
    except (OSError, zipfile.BadZipFile, KeyError) as error:
        raise ValueError(f"invalid TorchScript archive {source}: {error}") from error
    try:
        document = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite metadata number: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid embedded metadata.json in {source}: {error}") from error
    if not isinstance(document, dict):
        raise ValueError(f"embedded metadata.json in {source} must be an object")
    try:
        metadata = RuntimeGRUMetadata.from_dict(document, f"{source}:metadata.json")
    except ValueError as error:
        raise ValueError(f"invalid embedded metadata.json in {source}: {error}") from error
    return EmbeddedRuntimeMetadata(metadata, raw, sha256_hex(raw))


__all__ = ["EmbeddedRuntimeMetadata", "read_runtime_metadata_archive"]
