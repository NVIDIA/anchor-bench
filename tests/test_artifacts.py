from pathlib import Path

import pytest
import yaml

from anchor_bench.core import artifacts
from anchor_bench.core.artifacts import (
    get_or_create_output_folder,
    resolve_repo_root,
)
from anchor_bench.schema.base import BaseConfig


def test_get_or_create_output_folder_writes_config_and_input_data(tmp_path: Path):
    config = BaseConfig(type="example")
    input_data = {"dataset": "measurements.parquet", "samples": 42}

    output_folder, already_existed = get_or_create_output_folder(
        ("example", 42),
        config=config,
        input_data=input_data,
        output_root=tmp_path,
    )

    assert already_existed is False
    assert output_folder.parent == tmp_path
    assert len(output_folder.name) == 8
    int(output_folder.name, 16)
    assert yaml.safe_load((output_folder / "config.yaml").read_text()) == {
        "type": "example"
    }
    assert yaml.safe_load((output_folder / "input_data.yaml").read_text()) == (
        input_data
    )


def test_get_or_create_output_folder_returns_existing_run_without_writing(
    tmp_path: Path,
):
    config = BaseConfig(type="example")
    arguments = {
        "output_root": tmp_path,
        "hash_input": "same-input",
        "config": config,
        "input_data": {},
    }

    output_folder, already_existed = get_or_create_output_folder(**arguments)
    assert already_existed is False
    original_config = (output_folder / "config.yaml").read_text()
    original_input_data = (output_folder / "input_data.yaml").read_text()

    existing_folder, already_existed = get_or_create_output_folder(
        "same-input",
        config=BaseConfig(type="different"),
        input_data={"different": "input"},
        output_root=tmp_path,
    )

    assert existing_folder == output_folder
    assert already_existed is True
    assert (output_folder / "config.yaml").read_text() == original_config
    assert (output_folder / "input_data.yaml").read_text() == original_input_data


def test_get_or_create_output_folder_uses_repository_output_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(artifacts, "resolve_repo_root", lambda: tmp_path)

    output_folder, already_existed = get_or_create_output_folder(
        "default-root",
        config=BaseConfig(type="example"),
        input_data={},
    )

    assert already_existed is False
    assert output_folder.parent == tmp_path / "output"


def test_resolve_repo_root_returns_project_root():
    assert resolve_repo_root() == Path(__file__).resolve().parents[1]
