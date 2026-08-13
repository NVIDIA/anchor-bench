from pathlib import Path

import pytest
import yaml

from anchor_bench.core import artifacts
from anchor_bench.core.artifacts import (
    get_or_create_output_folder,
    resolve_repo_root,
)
from anchor_bench.schema.base import BaseConfig
from anchor_bench.schema.mlops import MLOpsConfig


class ConfigWithMLOps(BaseConfig):
    """Minimal top-level config used to exercise artifact serialization."""

    mlops: MLOpsConfig


def test_get_or_create_output_folder_writes_config(tmp_path: Path):
    config = BaseConfig(type="example")

    output_folder, already_completed = get_or_create_output_folder(
        config,
        output_root=tmp_path,
    )

    assert already_completed is False
    assert output_folder.parent == tmp_path
    assert len(output_folder.name) == 8
    int(output_folder.name, 16)
    assert yaml.safe_load((output_folder / "config.yaml").read_text()) == {
        "type": "example"
    }


def test_get_or_create_output_folder_preserves_mlops_options(tmp_path: Path):
    config = ConfigWithMLOps(
        type="example",
        mlops=MLOpsConfig(
            backend="custom",
            experiment_name="training",
            run_name="baseline",
            upload_dataset=True,
            options={"mode": "offline"},
        ),
    )

    output_folder, _ = get_or_create_output_folder(config, output_root=tmp_path)

    assert yaml.safe_load((output_folder / "config.yaml").read_text()) == {
        "type": "example",
        "mlops": {
            "backend": "custom",
            "experiment_name": "training",
            "run_name": "baseline",
            "upload_dataset": True,
            "options": {"mode": "offline"},
        },
    }


def test_get_or_create_output_folder_reports_completion_marker_without_writing(
    tmp_path: Path,
):
    config = BaseConfig(type="example")
    arguments = {
        "output_root": tmp_path,
        "config": config,
    }

    output_folder, already_completed = get_or_create_output_folder(**arguments)
    assert already_completed is False
    config_path = output_folder / "config.yaml"
    config_path.write_text("existing content")

    existing_folder, already_completed = get_or_create_output_folder(**arguments)

    assert existing_folder == output_folder
    assert already_completed is False
    assert config_path.read_text() == "existing content"

    (output_folder / ".completed").touch()
    completed_folder, already_completed = get_or_create_output_folder(**arguments)

    assert completed_folder == output_folder
    assert already_completed is True
    assert config_path.read_text() == "existing content"


def test_get_or_create_output_folder_uses_repository_output_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(artifacts, "resolve_repo_root", lambda: tmp_path)

    output_folder, already_completed = get_or_create_output_folder(
        BaseConfig(type="example"),
    )

    assert already_completed is False
    assert output_folder.parent == tmp_path / "output"


def test_get_or_create_output_folder_uses_explicit_output_folder(tmp_path: Path):
    requested_folder = tmp_path / "named-output"

    output_folder, already_completed = get_or_create_output_folder(
        BaseConfig(type="example"),
        output_root=tmp_path / "ignored-root",
        output_folder=requested_folder,
    )

    assert output_folder == requested_folder
    assert already_completed is False
    assert yaml.safe_load((output_folder / "config.yaml").read_text()) == {
        "type": "example"
    }


def test_resolve_repo_root_returns_project_root():
    assert resolve_repo_root() == Path(__file__).resolve().parents[1]
