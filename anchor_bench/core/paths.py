
from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def config_root() -> Path:
    return repo_root() / "configs"


def robot_config_root() -> Path:
    return config_root() / "robots"


def asset_root() -> Path:
    return repo_root() / "assets"


def output_root() -> Path:
    return repo_root() / "outputs"


def third_party_root() -> Path:
    return repo_root() / "third_party"
