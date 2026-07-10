"""Host-safe immutable-run persistence and verification.

All phase runners share this protocol: take a stable sibling lock, validate a
completed target as a no-op, otherwise work in a temporary sibling, write the
manifest last, verify every byte, and atomically rename the directory into
place.  Existing incomplete, corrupt, or identity-mismatched targets are never
silently replaced.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import shutil
import stat
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from anchor_bench.core.hashing import (
    UnsafePathError,
    contained_path,
    require_relative_posix_path,
    sha256_file,
    validate_sha256,
)
from anchor_bench.core.manifest_schema import (
    MANIFEST_NAME,
    GeneratedFile,
    RunManifest,
    finalize_manifest,
    manifest_sha256,
)
from anchor_bench.core.profiles import load_yaml_mapping


class RunVerificationError(RuntimeError):
    """A run directory is incomplete, corrupt, unsafe, or mismatched."""


# A useful generic spelling for callers that are not specifically phase runs.
ArtifactError = RunVerificationError


def sibling_lock_path(target: str | Path) -> Path:
    """Return the stable lock file beside ``target``."""

    path = Path(target)
    return path.with_name(f"{path.name}.lock")


@contextmanager
def sibling_lock(target: str | Path) -> Iterator[None]:
    """Hold an exclusive ``flock`` on a stable sibling file."""

    lock_path = sibling_lock_path(target)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as error:
        raise RunVerificationError(f"cannot open stable lock {lock_path}: {error}") from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise RunVerificationError(f"stable lock is not a regular file: {lock_path}")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


@contextmanager
def ordered_locks(*targets: str | Path) -> Iterator[None]:
    """Acquire stable sibling locks in caller-specified order."""

    normalized = [Path(target).absolute() for target in targets]
    if len(normalized) != len(set(normalized)):
        raise ValueError("ordered_locks targets must be unique")
    with contextlib.ExitStack() as stack:
        for target in targets:
            stack.enter_context(sibling_lock(target))
        yield


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _tree_entries(root: Path) -> tuple[dict[str, Path], list[Path]]:
    """Inventory regular files and directories, rejecting every symlink."""

    if root.is_symlink() or not root.is_dir():
        raise RunVerificationError(f"run path is not a real directory: {root}")
    resolved_root = root.resolve(strict=True)
    files: dict[str, Path] = {}
    directories: list[Path] = [resolved_root]
    for current, directory_names, file_names in os.walk(resolved_root, followlinks=False):
        current_path = Path(current)
        directory_names.sort()
        file_names.sort()
        for name in directory_names:
            child = current_path / name
            if child.is_symlink():
                raise RunVerificationError(f"run directory contains a symlink: {child}")
            if not child.is_dir():
                raise RunVerificationError(f"run directory contains a non-directory: {child}")
            directories.append(child)
        for name in file_names:
            child = current_path / name
            if child.is_symlink():
                raise RunVerificationError(f"run directory contains a symlink: {child}")
            try:
                mode = child.lstat().st_mode
            except OSError as error:
                raise RunVerificationError(f"cannot inspect generated file {child}: {error}") from error
            if not stat.S_ISREG(mode):
                raise RunVerificationError(f"run directory contains a non-regular file: {child}")
            relative = child.relative_to(resolved_root).as_posix()
            files[relative] = child
    return files, directories


def generated_file(run_dir: str | Path, path: str | Path, role: str) -> GeneratedFile:
    """Build one generated-file record from raw bytes under ``run_dir``."""

    root = Path(run_dir)
    relative = require_relative_posix_path(Path(path).as_posix())
    try:
        payload = contained_path(root, relative, must_exist=True)
    except UnsafePathError as error:
        raise RunVerificationError(str(error)) from error
    if payload.is_symlink() or not payload.is_file():
        raise RunVerificationError(f"generated artifact is not a regular file: {payload}")
    return GeneratedFile(
        path=relative,
        role=role,
        sha256=sha256_file(payload),
        size_bytes=payload.stat().st_size,
    )


def generated_files(
    run_dir: str | Path,
    roles: Mapping[str | Path, str] | Sequence[tuple[str | Path, str]],
) -> tuple[GeneratedFile, ...]:
    """Build a path-sorted set of generated-file records."""

    items = roles.items() if isinstance(roles, Mapping) else roles
    records = [generated_file(run_dir, path, role) for path, role in items]
    records.sort(key=lambda entry: entry.path)
    if len({entry.path for entry in records}) != len(records):
        raise ValueError("duplicate generated-file path")
    return tuple(records)


def _verify_generated_files(
    run_dir: Path,
    manifest: RunManifest,
    *,
    require_manifest_file: bool,
) -> None:
    actual_files, _ = _tree_entries(run_dir)
    manifest_path = actual_files.pop(MANIFEST_NAME, None)
    if require_manifest_file and manifest_path is None:
        raise RunVerificationError(f"missing {MANIFEST_NAME} in {run_dir}")
    if not require_manifest_file and manifest_path is not None:
        raise RunVerificationError(f"{MANIFEST_NAME} already exists in unfinished run {run_dir}")

    expected_paths = {entry.path for entry in manifest.generated_files}
    actual_paths = set(actual_files)
    missing = sorted(expected_paths - actual_paths)
    extras = sorted(actual_paths - expected_paths)
    if missing:
        raise RunVerificationError(f"missing generated files in {run_dir}: {', '.join(missing)}")
    if extras:
        raise RunVerificationError(f"unrecorded generated files in {run_dir}: {', '.join(extras)}")

    for entry in manifest.generated_files:
        payload = actual_files[entry.path]
        actual_hash = sha256_file(payload)
        if actual_hash != entry.sha256:
            raise RunVerificationError(
                f"hash mismatch for {payload}: expected {entry.sha256}, got {actual_hash}"
            )
        if entry.size_bytes is not None:
            actual_size = payload.stat().st_size
            if actual_size != entry.size_bytes:
                raise RunVerificationError(
                    f"size mismatch for {payload}: expected {entry.size_bytes}, got {actual_size}"
                )


def manifest_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / MANIFEST_NAME


def load_manifest(run_dir: str | Path) -> tuple[RunManifest, str]:
    """Load a finalized manifest and return it with its canonical self-hash."""

    root = Path(run_dir)
    path = manifest_path(root)
    if root.is_symlink() or not root.is_dir():
        raise RunVerificationError(f"run path is not a real directory: {root}")
    if path.is_symlink() or not path.is_file():
        raise RunVerificationError(f"missing regular {MANIFEST_NAME} in {root}")
    try:
        data = load_yaml_mapping(path)
        manifest = RunManifest.from_dict(data, path)
    except (OSError, ValueError) as error:
        raise RunVerificationError(f"invalid {MANIFEST_NAME} in {root}: {error}") from error
    if manifest.run_manifest_sha256 is None:
        raise RunVerificationError(f"{MANIFEST_NAME} has no run_manifest_sha256")
    actual_hash = manifest_sha256(manifest)
    if actual_hash != manifest.run_manifest_sha256:
        raise RunVerificationError(
            f"run manifest hash mismatch in {root}: expected {manifest.run_manifest_sha256}, got {actual_hash}"
        )
    return manifest, actual_hash


def verify_run_directory(
    run_dir: str | Path,
    expected_run_input_sha256: str | None = None,
    *,
    expected_run_manifest_sha256: str | None = None,
) -> RunManifest:
    """Authenticate one completed immutable run directory."""

    root = Path(run_dir)
    manifest, actual_manifest_hash = load_manifest(root)
    if expected_run_input_sha256 is not None:
        validate_sha256(expected_run_input_sha256, label="expected_run_input_sha256")
        if manifest.run_input_sha256 != expected_run_input_sha256:
            raise RunVerificationError(
                "run input identity mismatch for "
                f"{root}: expected {expected_run_input_sha256}, got {manifest.run_input_sha256}"
            )
    if expected_run_manifest_sha256 is not None:
        validate_sha256(expected_run_manifest_sha256, label="expected_run_manifest_sha256")
        if actual_manifest_hash != expected_run_manifest_sha256:
            raise RunVerificationError(
                "run manifest identity mismatch for "
                f"{root}: expected {expected_run_manifest_sha256}, got {actual_manifest_hash}"
            )
    _verify_generated_files(root, manifest, require_manifest_file=True)
    return manifest


# More explicit spelling used by some phase runners.
validate_completed_output = verify_run_directory


def write_manifest(run_dir: str | Path, manifest: RunManifest) -> str:
    """Verify payloads and create ``manifest.yaml`` as the final run file."""

    root = Path(run_dir)
    if root.is_symlink() or not root.is_dir():
        raise RunVerificationError(f"run path is not a real directory: {root}")
    finalized = finalize_manifest(manifest)
    _verify_generated_files(root, finalized, require_manifest_file=False)
    path = manifest_path(root)
    document = finalized.model_dump(mode="json", exclude_none=True)
    try:
        serialized = yaml.safe_dump(document, sort_keys=False, allow_unicode=True)
    except yaml.YAMLError as error:
        raise RunVerificationError(f"cannot serialize {MANIFEST_NAME}: {error}") from error

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags, 0o644)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = None
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(root)
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        path.unlink(missing_ok=True)
        raise
    return finalized.run_manifest_sha256 or manifest_sha256(finalized)


# Emphasize the protocol ordering at call sites.
write_manifest_last = write_manifest


def atomic_write_yaml(path: str | Path, data: Mapping[str, Any]) -> None:
    """Atomically replace a YAML mapping using a flushed temporary sibling."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        serialized = yaml.safe_dump(dict(data), sort_keys=False, allow_unicode=True)
    except yaml.YAMLError as error:
        raise ValueError(f"cannot serialize YAML for {target}: {error}") from error
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.tmp-",
        dir=target.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_install_directory(temporary: str | Path, final: str | Path) -> None:
    """Atomically install a completed temporary sibling without replacement."""

    source = Path(temporary)
    target = Path(final)
    if source.parent.resolve(strict=True) != target.parent.resolve(strict=True):
        raise ValueError("temporary and final run directories must be siblings")
    if _lexists(target):
        raise FileExistsError(f"refusing to replace existing output: {target}")
    if source.is_symlink() or not source.is_dir():
        raise ValueError(f"temporary run is not a real directory: {source}")
    os.rename(source, target)
    _fsync_directory(target.parent)


# Compatibility with the implementation plan's original spelling.
atomic_install_dir = atomic_install_directory


def _remove_temporary(path: Path) -> None:
    if path.is_symlink():
        path.unlink(missing_ok=True)
    elif path.exists():
        shutil.rmtree(path)


@contextmanager
def temporary_sibling(target: str | Path) -> Iterator[Path]:
    """Create a unique temporary directory beside ``target`` and clean it up."""

    final = Path(target)
    final.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{final.name}.tmp-", dir=final.parent))
    try:
        yield temporary
    finally:
        _remove_temporary(temporary)


@dataclass(frozen=True)
class RunSlot:
    """One locked immutable output decision."""

    noop: bool
    run_dir: Path
    tmp_dir: Path | None
    manifest: RunManifest | None = None


@contextmanager
def run_slot(
    run_dir: str | Path,
    expected_run_input_sha256: str | None = None,
) -> Iterator[RunSlot]:
    """Yield either a validated no-op or a temporary sibling to complete."""

    target = Path(run_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    with sibling_lock(target):
        if _lexists(target):
            manifest = verify_run_directory(target, expected_run_input_sha256)
            yield RunSlot(noop=True, run_dir=target, tmp_dir=None, manifest=manifest)
            return

        temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
        try:
            yield RunSlot(noop=False, run_dir=target, tmp_dir=temporary)
            verify_run_directory(temporary, expected_run_input_sha256)
            files, directories = _tree_entries(temporary)
            for payload in files.values():
                _fsync_file(payload)
            for directory in reversed(directories):
                _fsync_directory(directory)
            atomic_install_directory(temporary, target)
        except BaseException:
            _remove_temporary(temporary)
            raise


__all__ = [
    "ArtifactError",
    "GeneratedFile",
    "MANIFEST_NAME",
    "RunManifest",
    "RunSlot",
    "RunVerificationError",
    "atomic_install_dir",
    "atomic_install_directory",
    "atomic_write_yaml",
    "generated_file",
    "generated_files",
    "load_manifest",
    "manifest_path",
    "ordered_locks",
    "run_slot",
    "sibling_lock",
    "sibling_lock_path",
    "temporary_sibling",
    "validate_completed_output",
    "verify_run_directory",
    "write_manifest",
    "write_manifest_last",
]
