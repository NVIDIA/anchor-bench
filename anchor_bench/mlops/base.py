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
        """Log scalar metrics at an optional run-global, monotonic step."""

    @abstractmethod
    def set_tags(self, tags: Mapping[str, str]) -> None:
        """Add or replace searchable run metadata."""

    @abstractmethod
    def log_artifact(
        self,
        path: Union[str, Path],
        *,
        collection: Optional[str] = None,
        artifact_type: str = "anchor-bench",
    ) -> None:
        """Log a local file or directory in a logical artifact collection."""


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
