from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import yaml


@dataclass(frozen=True)
class CommandPlan:
    workflow: str
    robot: str
    command: list[str]
    dry_run: bool
    notes: list[str]
    runs: list[list[str]] | None = None
    metadata: dict[str, Any] | None = None
    warnings: list[str] | None = None

    def to_yaml(self) -> str:
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
