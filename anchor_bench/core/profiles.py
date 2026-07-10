"""Shared profile selection and loading.

A selected profile is a complete recipe.  ``default.yaml`` is selected only
when its logical name is requested; it is never recursively merged into another
profile.  Typed CLI override application intentionally belongs to each phase's
resolver, after this loader returns a validated model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

import yaml

from anchor_bench.core.config_model import ConfigModel
from anchor_bench.core.hashing import (
    UnsafePathError,
    canonical_sha256,
    contained_path,
    repo_relative_posix,
    sha256_hex,
)


_LOGICAL_NAME = re.compile(r"^[a-z0-9_][a-z0-9_-]*$")
_MAX_LOGICAL_NAME_LENGTH = 128
ProfileModelT = TypeVar("ProfileModelT", bound=ConfigModel)


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that refuses silent duplicate-key replacement."""


def _construct_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as error:
            raise ValueError("YAML mapping keys must be hashable") from error
        if duplicate:
            raise ValueError(f"duplicate YAML key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)


def require_logical_name(value: str, *, label: str = "logical name") -> str:
    """Validate a filename-safe lowercase logical name."""

    if (
        not isinstance(value, str)
        or len(value) > _MAX_LOGICAL_NAME_LENGTH
        or _LOGICAL_NAME.fullmatch(value) is None
    ):
        raise UnsafePathError(f"invalid {label}: {value!r}")
    return value


def profile_path(
    profiles_directory: str | Path,
    name: str = "default",
    *,
    must_exist: bool = True,
) -> Path:
    """Resolve one logical profile name inside ``profiles_directory``."""

    logical_name = require_logical_name(name, label="profile name")
    return contained_path(
        Path(profiles_directory),
        f"{logical_name}.yaml",
        must_exist=must_exist,
    )


def load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    """Load one UTF-8 YAML mapping while rejecting duplicate keys."""

    source = Path(path)
    try:
        raw = source.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"cannot read UTF-8 YAML mapping {source}: {error}") from error
    try:
        data = yaml.load(text, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as error:
        raise ValueError(f"invalid YAML in {source}: {error}") from error
    if not isinstance(data, dict):
        raise ValueError(f"{source} must contain a YAML mapping")
    if any(not isinstance(key, str) for key in data):
        raise ValueError(f"{source} must contain only string mapping keys")
    return data


@dataclass(frozen=True)
class LoadedProfile(Generic[ProfileModelT]):
    """A validated profile plus exact source and semantic fingerprints."""

    name: str
    path: Path
    raw_bytes: bytes
    raw_sha256: str
    content: dict[str, Any]
    content_sha256: str
    model: ProfileModelT

    @property
    def raw_yaml(self) -> str:
        return self.raw_bytes.decode("utf-8")

    def manifest_data(self, *, repository_root: str | Path) -> dict[str, Any]:
        """Return the common manifest profile snapshot envelope."""

        return {
            "name": self.name,
            "path": repo_relative_posix(
                self.path,
                repository_root=repository_root,
                must_exist=True,
            ),
            "raw_yaml": self.raw_yaml,
            "raw_sha256": self.raw_sha256,
            "content": self.content,
            "content_sha256": self.content_sha256,
        }


def load_profile(
    profiles_directory: str | Path,
    name: str,
    model_type: type[ProfileModelT],
    *,
    context: str | Path | None = None,
) -> LoadedProfile[ProfileModelT]:
    """Load exactly one selected profile as ``model_type``.

    The loader performs no default-profile merge and accepts no override map.
    Phase code applies already-typed CLI fields to the returned model as whole
    field replacements and then validates its resolved phase configuration.
    """

    logical_name = require_logical_name(name, label="profile name")
    source = profile_path(profiles_directory, logical_name)
    try:
        raw_bytes = source.read_bytes()
        text = raw_bytes.decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"cannot read UTF-8 profile {source}: {error}") from error
    try:
        document = yaml.load(text, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as error:
        raise ValueError(f"invalid YAML in profile {source}: {error}") from error
    if not isinstance(document, dict):
        raise ValueError(f"{source} must contain a YAML mapping")
    if any(not isinstance(key, str) for key in document):
        raise ValueError(f"{source} must contain only string mapping keys")

    model = model_type.from_dict(document, context or source)
    content = model.to_dict()
    return LoadedProfile(
        name=logical_name,
        path=source,
        raw_bytes=raw_bytes,
        raw_sha256=sha256_hex(raw_bytes),
        content=content,
        content_sha256=canonical_sha256(content),
        model=model,
    )


__all__ = [
    "LoadedProfile",
    "UnsafePathError",
    "load_profile",
    "load_yaml_mapping",
    "profile_path",
    "require_logical_name",
]
