from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest

from anchor_bench.core.config_model import ConfigModel
from anchor_bench.core.hashing import UnsafePathError
from anchor_bench.core.profiles import load_profile, profile_path, require_logical_name


class ExampleProfile(ConfigModel):
    schema_version: Literal[1]
    required: int
    enabled: bool = True
    mappings: tuple[str, ...] = ()


def _write(directory: Path, name: str, text: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_selected_profile_is_loaded_without_default_merge(tmp_path: Path) -> None:
    profiles = tmp_path / "profiles"
    _write(profiles, "default", "this: is intentionally invalid for the schema\n")
    _write(
        profiles,
        "selected",
        "schema_version: 1\nrequired: 7\nenabled: false\nmappings: [joint_a]\n",
    )
    loaded = load_profile(profiles, "selected", ExampleProfile)
    assert loaded.name == "selected"
    assert loaded.model.required == 7
    assert loaded.model.enabled is False
    assert loaded.model.mappings == ("joint_a",)


def test_selected_profile_cannot_inherit_a_missing_required_field(tmp_path: Path) -> None:
    profiles = tmp_path / "profiles"
    _write(profiles, "default", "schema_version: 1\nrequired: 7\n")
    _write(profiles, "selected", "schema_version: 1\n")
    with pytest.raises(ValueError, match="missing required key 'required'"):
        load_profile(profiles, "selected", ExampleProfile)


def test_profile_schema_is_strict_and_duplicate_keys_fail(tmp_path: Path) -> None:
    profiles = tmp_path / "profiles"
    _write(profiles, "extra", "schema_version: 1\nrequired: 1\nunknown: true\n")
    with pytest.raises(ValueError, match="unknown keys: unknown"):
        load_profile(profiles, "extra", ExampleProfile)

    _write(profiles, "duplicate", "schema_version: 1\nrequired: 1\nrequired: 2\n")
    with pytest.raises(ValueError, match="duplicate YAML key"):
        load_profile(profiles, "duplicate", ExampleProfile)


def test_raw_format_changes_raw_hash_not_validated_content_hash(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first = _write(first_dir, "same", "schema_version: 1\nrequired: 3\n")
    second = _write(
        second_dir,
        "same",
        "# formatting is provenance, not semantic identity\nrequired: 3\nschema_version: 1\n",
    )
    first_loaded = load_profile(first_dir, "same", ExampleProfile)
    second_loaded = load_profile(second_dir, "same", ExampleProfile)
    assert first_loaded.path == first
    assert second_loaded.path == second
    assert first_loaded.raw_sha256 != second_loaded.raw_sha256
    assert first_loaded.content_sha256 == second_loaded.content_sha256
    assert first_loaded.content == second_loaded.content


def test_manifest_data_uses_repository_relative_posix_path(tmp_path: Path) -> None:
    profiles = tmp_path / "configs" / "robots" / "g1" / "train" / "profiles"
    _write(profiles, "default", "schema_version: 1\nrequired: 1\n")
    loaded = load_profile(profiles, "default", ExampleProfile)
    data = loaded.manifest_data(repository_root=tmp_path)
    assert data["path"] == "configs/robots/g1/train/profiles/default.yaml"
    assert data["raw_yaml"] == "schema_version: 1\nrequired: 1\n"


def test_logical_names_and_profile_containment_are_strict(tmp_path: Path) -> None:
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    assert require_logical_name("right_arm_gru") == "right_arm_gru"
    for bad in ("../x", "a/b", "", "UPPER", "has space", "-lead", "x.yaml"):
        with pytest.raises(UnsafePathError):
            require_logical_name(bad)
    with pytest.raises(UnsafePathError):
        profile_path(profiles, "../escape", must_exist=False)


def test_profile_symlink_escape_is_rejected(tmp_path: Path) -> None:
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    outside = tmp_path / "outside.yaml"
    outside.write_text("schema_version: 1\nrequired: 1\n", encoding="utf-8")
    (profiles / "escape.yaml").symlink_to(outside)
    with pytest.raises(UnsafePathError, match="escapes"):
        load_profile(profiles, "escape", ExampleProfile)
