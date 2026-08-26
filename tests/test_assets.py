# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import newton.utils
import pytest
from pydantic import ValidationError

from anchor_bench.assets import resolve_usd_path
from anchor_bench.errors import AssetResolutionError
from anchor_bench.schema.training import SimulationConfig

ASSET_REF = "a96f0973b6ae69c90609f9fafef9d2c1db2d6431"


def test_resolves_local_path(tmp_path: Path) -> None:
    asset = tmp_path / "robot.usda"
    config = SimulationConfig.model_validate(
        {"usd": {"type": "local", "path": str(asset)}}
    )

    assert resolve_usd_path(config.usd) == asset


def test_downloads_revision_pinned_newton_asset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset_root = tmp_path / "robotstudio_so101"
    usd_path = asset_root / "usd_structured" / "so101.usda"
    usd_path.parent.mkdir(parents=True)
    usd_path.write_text("#usda 1.0", encoding="utf-8")
    calls = []

    def download_asset(folder: str, *, ref: str) -> Path:
        calls.append((folder, ref))
        return asset_root

    monkeypatch.setattr(newton.utils, "download_asset", download_asset)
    config = SimulationConfig.model_validate(
        {
            "usd": {
                "type": "newton-assets",
                "folder": "robotstudio_so101",
                "path": "usd_structured/so101.usda",
                "ref": ASSET_REF,
            }
        }
    )

    assert resolve_usd_path(config.usd) == usd_path
    assert calls == [("robotstudio_so101", ASSET_REF)]
    assert config.usd.path == "usd_structured/so101.usda"


def test_requires_usd_source() -> None:
    with pytest.raises(ValidationError, match="Field required"):
        SimulationConfig.model_validate({})


def test_rejects_removed_usd_path() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        SimulationConfig.model_validate({"usd_path": "robot.usda"})


@pytest.mark.parametrize("path", ["/absolute.usda", "../robot.usda", "a/../b.usda"])
def test_rejects_unsafe_newton_asset_paths(path: str) -> None:
    with pytest.raises(ValidationError, match="normalized relative POSIX path"):
        SimulationConfig.model_validate(
            {
                "usd": {
                    "type": "newton-assets",
                    "folder": "robotstudio_so101",
                    "path": path,
                    "ref": ASSET_REF,
                }
            }
        )


def test_reports_missing_downloaded_asset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        newton.utils,
        "download_asset",
        lambda folder, *, ref: tmp_path,
    )
    config = SimulationConfig.model_validate(
        {
            "usd": {
                "type": "newton-assets",
                "folder": "robotstudio_so101",
                "path": "usd_structured/so101.usda",
                "ref": ASSET_REF,
            }
        }
    )

    with pytest.raises(AssetResolutionError, match="does not exist"):
        resolve_usd_path(config.usd)
