
from __future__ import annotations

from dataclasses import dataclass
import tomllib

from anchor_bench.core.paths import repo_root


@dataclass(frozen=True)
class ExternalVersions:
    isaac_sim_version: str
    python_version: str
    newton_version: str
    newton_package: str
    newton_repo: str
    newton_tag: str
    isaac_lab_repo: str
    isaac_lab_branch: str
    isaac_lab_commit: str


def load_external_versions() -> ExternalVersions:
    path = repo_root() / "versions" / "compatibility.toml"
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return ExternalVersions(
        isaac_sim_version=data["isaac_sim"]["version"],
        python_version=data["isaac_sim"]["python"],
        newton_version=data["newton"]["version"],
        newton_package=data["newton"]["package"],
        newton_repo=data["newton"]["repo"],
        newton_tag=data["newton"]["tag"],
        isaac_lab_repo=data["isaac_lab"]["repo"],
        isaac_lab_branch=data["isaac_lab"]["branch"],
        isaac_lab_commit=data["isaac_lab"]["commit"],
    )
