# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend-neutral interfaces for experiment tracking."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Mapping, Optional, Union

import pandas as pd

ParameterValue = Union[str, int, float, bool]


class MLOpsRun(ABC):
    """One active experiment run owned by an MLOps backend."""

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

    def log_dataset(
        self,
        dataset: pd.DataFrame,
        *,
        source: str,
        digest: str,
        name: str = "dataset",
        context: str = "input",
        path: Optional[Union[str, Path]] = None,
        metadata: Optional[Mapping[str, ParameterValue]] = None,
        upload: bool = False,
    ) -> None:
        """Log dataset lineage and optionally upload its materialized payload.

        Backends can override this method to additionally use native dataset-input
        concepts. The default implementation remains useful to third-party backends
        by recording portable parameters and reusing artifact logging for uploads.
        """

        prefix = f"datasets/{name}"
        parameters: dict[str, ParameterValue] = {
            f"{prefix}/source": source,
            f"{prefix}/digest": digest,
            f"{prefix}/context": context,
            f"{prefix}/rows": len(dataset.index),
            f"{prefix}/columns": json.dumps(
                [str(column) for column in dataset.columns],
                separators=(",", ":"),
            ),
        }
        if metadata is not None:
            parameters.update(
                {f"{prefix}/metadata/{key}": value for key, value in metadata.items()}
            )
        self.log_params(parameters)

        if upload:
            if path is None:
                raise ValueError(
                    "A local dataset path is required when upload is enabled."
                )
            self.log_artifact(
                path,
                collection=name,
                artifact_type="dataset",
            )

        self.log_params({f"{prefix}/uploaded": upload})

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
        """Create an active run that must be used as a context manager.

        The context owns backend success/failure finalization. Implementations
        must propagate exceptions from the task body so callers can keep local
        completion state consistent with the tracking service.
        """
