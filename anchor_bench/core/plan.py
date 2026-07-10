from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class CommandPlan:
    """Human-readable command preview and its non-secret resolved metadata."""

    workflow: str
    robot: str
    command: list[str]
    dry_run: bool
    notes: list[str]
    runs: list[list[str]] | None = None
    metadata: dict[str, Any] | None = None
    warnings: list[str] | None = None

    def to_yaml(self) -> str:
        """Serialize the command plan as stable user-facing YAML."""

        data: dict[str, Any] = {
            "workflow": self.workflow,
            "robot": self.robot,
            "command": self.command,
            "dry_run": self.dry_run,
            "notes": self.notes,
        }
        if self.runs is not None:
            data["runs"] = self.runs
        if self.metadata is not None:
            data["metadata"] = self.metadata
        if self.warnings:
            data["warnings"] = self.warnings
        return yaml.safe_dump(data, sort_keys=False)


def command_identity_metadata(
    run_dir: Path,
    run_input_sha256: str,
    *,
    preview: bool,
) -> dict[str, str]:
    """Label a run identity as authoritative or explicitly provisional."""

    if preview:
        return {
            "identity": "preview",
            "preview_input_sha256": run_input_sha256,
            "preview_run_dir": str(run_dir),
        }
    return {
        "identity": "authoritative",
        "run_input_sha256": run_input_sha256,
        "run_dir": str(run_dir),
    }


__all__ = ["CommandPlan", "command_identity_metadata"]
