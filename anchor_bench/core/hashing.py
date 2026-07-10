"""Deterministic structured, file, tree, and path hashing primitives.

Semantic identities use compact canonical JSON.  Raw artifacts use SHA-256 of
their bytes.  These two domains are deliberately separate: reformatting YAML
does not change a semantic identity, while changing any generated artifact byte
does change its recorded artifact hash.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path, PurePath, PurePosixPath
from typing import Any


_CHUNK_SIZE = 1 << 20


class CanonicalizationError(ValueError):
    """A value cannot participate in a canonical semantic hash."""


class UnsafePathError(ValueError):
    """A path is not normalized or escapes its allowed container."""


def canonical_object(value: Any) -> Any:
    """Return the JSON-compatible canonical form of ``value``.

    Mapping keys must be strings, sequences retain their order, mapping order
    is normalized by :func:`canonical_bytes`, and paths become POSIX strings.
    Non-finite floats and implicit/coercive representations are rejected.
    """

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalizationError(f"non-finite float: {value!r}")
        return value
    if isinstance(value, PurePath):
        return value.as_posix()
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalizationError(f"non-string mapping key: {key!r}")
            result[key] = canonical_object(item)
        return result
    if isinstance(value, (list, tuple)):
        return [canonical_object(item) for item in value]
    raise CanonicalizationError(f"unsupported canonical type: {type(value).__name__}")


def canonical_bytes(value: Any) -> bytes:
    """Serialize ``value`` as deterministic, finite, compact UTF-8 JSON."""

    try:
        text = json.dumps(
            canonical_object(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        return text.encode("utf-8")
    except (UnicodeEncodeError, ValueError) as error:
        if isinstance(error, CanonicalizationError):
            raise
        raise CanonicalizationError(str(error)) from error


def sha256_hex(data: bytes) -> str:
    """Return the lowercase SHA-256 digest of raw ``data``."""

    return hashlib.sha256(data).hexdigest()


def canonical_sha256(value: Any) -> str:
    """Hash the canonical structured representation of ``value``."""

    return sha256_hex(canonical_bytes(value))


def sha256_file(path: str | Path) -> str:
    """Hash the raw bytes of one regular file.

    This primitive follows a caller-selected symlink.  Security boundaries
    must first resolve the path with :func:`contained_path`; run verification
    additionally rejects symlinks entirely.
    """

    source = Path(path)
    if not source.is_file():
        raise ValueError(f"not a regular file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        while chunk := stream.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


# Compatibility with the implementation plan's original spelling.
file_sha256 = sha256_file


def require_relative_posix_path(value: str | PurePosixPath) -> str:
    """Validate and return an already-normalized relative POSIX path."""

    text = value.as_posix() if isinstance(value, PurePosixPath) else value
    if not isinstance(text, str) or not text or "\x00" in text or "\\" in text:
        raise UnsafePathError(f"invalid relative POSIX path: {value!r}")
    raw_parts = text.split("/")
    if text.startswith("/") or any(part in {"", ".", ".."} for part in raw_parts):
        raise UnsafePathError(f"invalid relative POSIX path: {value!r}")
    normalized = PurePosixPath(text)
    if normalized.is_absolute() or normalized.as_posix() != text:
        raise UnsafePathError(f"path is not normalized relative POSIX: {value!r}")
    return text


def contained_path(
    root: str | Path,
    candidate: str | Path,
    *,
    must_exist: bool = True,
    allow_root: bool = False,
) -> Path:
    """Resolve ``candidate`` under ``root`` and reject traversal/symlink escape.

    Existing path components are resolved even when ``must_exist`` is false,
    so a symlinked parent cannot redirect a future output outside ``root``.
    """

    base_path = Path(root)
    try:
        base = base_path.resolve(strict=True)
    except OSError as error:
        raise UnsafePathError(f"containment root is not resolvable: {base_path}") from error
    raw = Path(candidate)
    target = raw if raw.is_absolute() else base / raw
    try:
        resolved = target.resolve(strict=must_exist)
    except OSError as error:
        raise UnsafePathError(f"path is not resolvable under {base}: {candidate}") from error
    try:
        relative = resolved.relative_to(base)
    except ValueError as error:
        raise UnsafePathError(f"path escapes {base}: {candidate}") from error
    if not allow_root and relative == Path("."):
        raise UnsafePathError(f"path must select a child of {base}: {candidate}")
    return resolved


def repo_relative_posix(
    path: str | Path,
    *,
    repository_root: str | Path,
    must_exist: bool = True,
) -> str:
    """Return a containment-checked repository-relative POSIX path."""

    root = Path(repository_root).resolve(strict=True)
    resolved = contained_path(root, path, must_exist=must_exist)
    return resolved.relative_to(root).as_posix()


def _walk_tree(root: Path) -> list[Path]:
    files: list[Path] = []
    for current, directory_names, file_names in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in directory_names:
            child = current_path / name
            if child.is_symlink():
                raise UnsafePathError(f"tree contains a symlink: {child}")
        for name in file_names:
            child = current_path / name
            if child.is_symlink():
                raise UnsafePathError(f"tree contains a symlink: {child}")
            if not child.is_file():
                raise UnsafePathError(f"tree contains a non-regular file: {child}")
            files.append(child)
    return sorted(files, key=lambda item: item.relative_to(root).as_posix())


def tree_manifest(root: str | Path) -> dict[str, str]:
    """Return sorted relative paths mapped to raw hashes for a directory tree."""

    base = Path(root)
    if base.is_symlink():
        raise UnsafePathError(f"tree root must not be a symlink: {base}")
    if not base.is_dir():
        raise ValueError(f"not a directory tree: {base}")
    resolved = base.resolve(strict=True)
    return {
        item.relative_to(resolved).as_posix(): sha256_file(item)
        for item in _walk_tree(resolved)
    }


def sha256_tree(root: str | Path) -> str:
    """Hash a directory's relative-path/raw-file-hash manifest."""

    return canonical_sha256(tree_manifest(root))


def tree_fingerprint(root: str | Path) -> dict[str, Any]:
    """Return the conventional file map plus its deterministic tree hash."""

    files = tree_manifest(root)
    return {"files": files, "tree_sha256": canonical_sha256(files)}


def validate_sha256(value: str, *, label: str = "sha256") -> str:
    """Validate a lowercase 64-character SHA-256 string."""

    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


__all__ = [
    "CanonicalizationError",
    "UnsafePathError",
    "canonical_bytes",
    "canonical_object",
    "canonical_sha256",
    "contained_path",
    "file_sha256",
    "repo_relative_posix",
    "require_relative_posix_path",
    "sha256_file",
    "sha256_hex",
    "sha256_tree",
    "tree_fingerprint",
    "tree_manifest",
    "validate_sha256",
]
