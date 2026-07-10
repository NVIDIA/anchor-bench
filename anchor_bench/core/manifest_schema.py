"""Strict common schema for immutable phase-run manifests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, Self

from pydantic import StrictInt, field_validator, model_validator

from anchor_bench.core.config_model import ConfigModel
from anchor_bench.core.hashing import (
    canonical_object,
    canonical_sha256,
    require_relative_posix_path,
    sha256_hex,
    validate_sha256,
)
from anchor_bench.core.profiles import require_logical_name


MANIFEST_NAME = "manifest.yaml"
Phase = Literal["sysid", "train", "benchmark", "deploy"]


class ProfileSnapshot(ConfigModel):
    """Exact selected profile bytes plus its validated semantic content."""

    name: str
    path: str
    raw_yaml: str
    raw_sha256: str
    content: dict[str, Any]
    content_sha256: str

    @field_validator("name")
    @classmethod
    def _valid_name(cls, value: str) -> str:
        """Validate the profile's logical name."""

        return require_logical_name(value, label="profile name")

    @field_validator("path")
    @classmethod
    def _valid_path(cls, value: str) -> str:
        """Normalize the profile source as a relative POSIX path."""

        return require_relative_posix_path(value)

    @field_validator("raw_sha256", "content_sha256")
    @classmethod
    def _valid_hash(cls, value: str, info: Any) -> str:
        """Validate profile source and semantic hashes as SHA-256 digests."""

        return validate_sha256(value, label=info.field_name)

    @model_validator(mode="after")
    def _matching_hashes(self) -> Self:
        """Require stored profile hashes to match the embedded raw and semantic content."""

        canonical_object(self.content)
        actual_raw = sha256_hex(self.raw_yaml.encode("utf-8"))
        if actual_raw != self.raw_sha256:
            raise ValueError(f"raw_sha256 mismatch: expected {actual_raw}")
        actual_content = canonical_sha256(self.content)
        if actual_content != self.content_sha256:
            raise ValueError(f"content_sha256 mismatch: expected {actual_content}")
        return self


class GeneratedFile(ConfigModel):
    """One authenticated regular file generated inside a completed run."""

    path: str
    role: str
    sha256: str
    size_bytes: StrictInt | None = None

    @field_validator("path")
    @classmethod
    def _valid_path(cls, value: str) -> str:
        """Normalize a generated payload path and exclude the manifest itself."""

        normalized = require_relative_posix_path(value)
        if normalized == MANIFEST_NAME:
            raise ValueError(f"{MANIFEST_NAME} cannot list itself as generated content")
        return normalized

    @field_validator("role")
    @classmethod
    def _valid_role(cls, value: str) -> str:
        """Validate the generated file's logical role name."""

        return require_logical_name(value, label="generated-file role")

    @field_validator("sha256")
    @classmethod
    def _valid_hash(cls, value: str) -> str:
        """Validate the generated file's content digest."""

        return validate_sha256(value)

    @field_validator("size_bytes")
    @classmethod
    def _valid_size(cls, value: int | None) -> int | None:
        """Require an optional generated file size to be non-negative."""

        if value is not None and value < 0:
            raise ValueError("must be >= 0")
        return value


class RunManifest(ConfigModel):
    """Common, phase-independent immutable-run manifest envelope."""

    schema_version: Literal[1] = 1
    phase: Phase
    robot: str
    profile: ProfileSnapshot
    resolved_config: dict[str, Any]
    resolved_config_sha256: str
    cli_overrides: dict[str, Any] = {}
    seed: StrictInt | None = None
    replicate: str | None = None
    run_input: dict[str, Any]
    run_input_sha256: str
    fingerprints: dict[str, Any] = {}
    upstream: dict[str, Any] = {}
    generated_files: tuple[GeneratedFile, ...]
    run_manifest_sha256: str | None = None

    @field_validator("robot")
    @classmethod
    def _valid_robot(cls, value: str) -> str:
        """Validate the manifest robot's logical name."""

        return require_logical_name(value, label="robot name")

    @field_validator("replicate")
    @classmethod
    def _valid_replicate(cls, value: str | None) -> str | None:
        """Validate an optional printable replicate label."""

        if value is None:
            return None
        if not value or len(value) > 128 or any(ord(character) < 32 for character in value):
            raise ValueError("replicate must be a non-empty printable label of at most 128 characters")
        return value

    @field_validator(
        "resolved_config_sha256",
        "run_input_sha256",
        "run_manifest_sha256",
    )
    @classmethod
    def _valid_hash(cls, value: str | None, info: Any) -> str | None:
        """Validate optional and required manifest identity digests."""

        if value is None:
            return None
        return validate_sha256(value, label=info.field_name)

    @model_validator(mode="after")
    def _validate_identity(self) -> Self:
        """Verify canonical identities and the generated-file index."""

        for value in (
            self.resolved_config,
            self.cli_overrides,
            self.run_input,
            self.fingerprints,
            self.upstream,
        ):
            canonical_object(value)

        resolved_hash = canonical_sha256(self.resolved_config)
        if resolved_hash != self.resolved_config_sha256:
            raise ValueError(f"resolved_config_sha256 mismatch: expected {resolved_hash}")

        input_hash = canonical_sha256(self.run_input)
        if input_hash != self.run_input_sha256:
            raise ValueError(f"run_input_sha256 mismatch: expected {input_hash}")

        paths = [entry.path for entry in self.generated_files]
        if len(paths) != len(set(paths)):
            raise ValueError("generated_files contains duplicate paths")
        if paths != sorted(paths):
            raise ValueError("generated_files must be sorted by path")

        if self.run_manifest_sha256 is not None:
            actual = manifest_sha256(self)
            if actual != self.run_manifest_sha256:
                raise ValueError(f"run_manifest_sha256 mismatch: expected {actual}")
        return self

    @classmethod
    def build(
        cls,
        *,
        phase: Phase,
        robot: str,
        profile: ProfileSnapshot | Mapping[str, Any],
        resolved_config: Mapping[str, Any],
        run_input: Mapping[str, Any],
        generated_files: Sequence[GeneratedFile | Mapping[str, Any]],
        cli_overrides: Mapping[str, Any] | None = None,
        seed: int | None = None,
        replicate: str | None = None,
        fingerprints: Mapping[str, Any] | None = None,
        upstream: Mapping[str, Any] | None = None,
        run_input_sha256: str | None = None,
    ) -> Self:
        """Build and finalize a manifest, computing both identity hashes."""

        return build_run_manifest(
            phase=phase,
            robot=robot,
            profile=profile,
            resolved_config=resolved_config,
            run_input=run_input,
            generated_files=generated_files,
            cli_overrides=cli_overrides,
            seed=seed,
            replicate=replicate,
            fingerprints=fingerprints,
            upstream=upstream,
            run_input_sha256=run_input_sha256,
        )


def manifest_sha256(manifest: RunManifest) -> str:
    """Hash a manifest's canonical contents, excluding its self-hash field."""

    data = manifest.model_dump(
        mode="json",
        exclude={"run_manifest_sha256"},
        exclude_none=True,
    )
    return canonical_sha256(data)


def finalize_manifest(manifest: RunManifest) -> RunManifest:
    """Return ``manifest`` with a verified canonical self-hash."""

    expected = manifest_sha256(manifest)
    if manifest.run_manifest_sha256 is not None:
        if manifest.run_manifest_sha256 != expected:
            raise ValueError(f"run_manifest_sha256 mismatch: expected {expected}")
        return manifest
    return manifest.model_copy(update={"run_manifest_sha256": expected})


def build_profile_snapshot(
    *,
    name: str,
    path: str,
    raw_yaml: str,
    content: Mapping[str, Any],
) -> ProfileSnapshot:
    """Build a validated profile snapshot and its two distinct hashes."""

    semantic_content = dict(content)
    return ProfileSnapshot(
        name=name,
        path=path,
        raw_yaml=raw_yaml,
        raw_sha256=sha256_hex(raw_yaml.encode("utf-8")),
        content=semantic_content,
        content_sha256=canonical_sha256(semantic_content),
    )


def build_run_manifest(
    *,
    phase: Phase,
    robot: str,
    profile: ProfileSnapshot | Mapping[str, Any],
    resolved_config: Mapping[str, Any],
    run_input: Mapping[str, Any],
    generated_files: Sequence[GeneratedFile | Mapping[str, Any]],
    cli_overrides: Mapping[str, Any] | None = None,
    seed: int | None = None,
    replicate: str | None = None,
    fingerprints: Mapping[str, Any] | None = None,
    upstream: Mapping[str, Any] | None = None,
    run_input_sha256: str | None = None,
) -> RunManifest:
    """Build the strict common envelope and compute its self-hash."""

    resolved = dict(resolved_config)
    semantic_input = dict(run_input)
    computed_input_hash = canonical_sha256(semantic_input)
    if run_input_sha256 is not None and run_input_sha256 != computed_input_hash:
        raise ValueError(f"run_input_sha256 mismatch: expected {computed_input_hash}")
    file_models = [
        item if isinstance(item, GeneratedFile) else GeneratedFile.from_dict(item, "generated_files")
        for item in generated_files
    ]
    file_models.sort(key=lambda item: item.path)
    profile_model = (
        profile
        if isinstance(profile, ProfileSnapshot)
        else ProfileSnapshot.from_dict(profile, "profile")
    )
    draft = RunManifest(
        phase=phase,
        robot=robot,
        profile=profile_model,
        resolved_config=resolved,
        resolved_config_sha256=canonical_sha256(resolved),
        cli_overrides=dict(cli_overrides or {}),
        seed=seed,
        replicate=replicate,
        run_input=semantic_input,
        run_input_sha256=computed_input_hash,
        fingerprints=dict(fingerprints or {}),
        upstream=dict(upstream or {}),
        generated_files=tuple(file_models),
    )
    return finalize_manifest(draft)


__all__ = [
    "GeneratedFile",
    "MANIFEST_NAME",
    "Phase",
    "ProfileSnapshot",
    "RunManifest",
    "build_profile_snapshot",
    "build_run_manifest",
    "finalize_manifest",
    "manifest_sha256",
]
