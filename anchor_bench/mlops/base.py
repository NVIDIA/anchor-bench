"""Backend-neutral interfaces for experiment tracking."""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Mapping, Optional, Union

ParameterValue = Union[str, int, float, bool]


class MLOpsRun(ABC):
    """One active experiment run owned by an MLOps backend."""

    @property
    @abstractmethod
    def run_id(self) -> str:
        """Return the backend's stable identifier for this run."""

    @abstractmethod
    def log_params(self, params: Mapping[str, ParameterValue]) -> None:
        """Log immutable run parameters."""

    @abstractmethod
    def log_metrics(
        self,
        metrics: Mapping[str, float],
        *,
        step: Optional[int] = None,
    ) -> None:
        """Log scalar metrics, optionally at a training step."""

    @abstractmethod
    def set_tags(self, tags: Mapping[str, str]) -> None:
        """Add or replace searchable run metadata."""

    @abstractmethod
    def log_artifact(
        self,
        path: Union[str, Path],
        *,
        artifact_path: Optional[str] = None,
    ) -> None:
        """Log a local file or directory as a run artifact."""


class MLOpsBackend(ABC):
    """Factory and lifecycle boundary implemented by tracking backends."""

    @abstractmethod
    def start_run(
        self,
        experiment_name: str,
        *,
        run_name: Optional[str] = None,
        tags: Optional[Mapping[str, str]] = None,
    ) -> AbstractContextManager[MLOpsRun]:
        """Create an active run that must be used as a context manager."""
