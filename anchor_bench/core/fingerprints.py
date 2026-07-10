"""Deterministic source and dependency fingerprints for semantic run identity."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from anchor_bench.core.hashing import canonical_sha256, sha256_file, sha256_hex


DEFAULT_SOURCE_PATHS = ("anchor_bench", "pyproject.toml", "versions")


def _git(repository: Path, *arguments: str) -> bytes:
    """Run one non-mutating Git query and return its exact stdout bytes."""

    try:
        result = subprocess.run(
            ("git", *arguments),
            cwd=repository,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", b"").decode("utf-8", errors="replace").strip()
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(f"unable to fingerprint Anchor source{suffix}") from error
    return result.stdout


def source_fingerprint(
    repository: str | Path,
    *,
    source_paths: Sequence[str] = DEFAULT_SOURCE_PATHS,
) -> dict[str, Any]:
    """Fingerprint HEAD plus tracked and untracked implementation changes.

    Datasets and outputs are intentionally outside the default source paths;
    phase identities fingerprint their contents independently.
    """

    root = Path(repository).resolve(strict=True)
    requested = tuple(str(path) for path in source_paths)
    if not requested or any(not path or Path(path).is_absolute() or ".." in Path(path).parts for path in requested):
        raise ValueError("source_paths must contain safe repository-relative paths")

    commit = _git(root, "rev-parse", "HEAD").decode("ascii").strip()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise RuntimeError("git rev-parse HEAD did not return a full commit hash")

    tracked_diff = _git(root, "diff", "--binary", "HEAD", "--", *requested)
    untracked_output = _git(
        root,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
        "--",
        *requested,
    )
    untracked_names = sorted(
        value.decode("utf-8")
        for value in untracked_output.split(b"\0")
        if value
    )
    untracked: dict[str, str] = {}
    for name in untracked_names:
        candidate = root / name
        if candidate.is_symlink():
            raise ValueError(f"untracked source path must not be a symlink: {name}")
        path = candidate.resolve(strict=True)
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"untracked source path escapes repository: {name}") from error
        if not path.is_file():
            raise ValueError(f"untracked source path must be a regular file: {name}")
        untracked[name] = sha256_file(path)

    content = {
        "git_commit": commit,
        "source_paths": list(requested),
        "tracked_diff_sha256": sha256_hex(tracked_diff),
        "untracked_files": untracked,
    }
    return {**content, "source_sha256": canonical_sha256(content)}


__all__ = ["DEFAULT_SOURCE_PATHS", "source_fingerprint"]
