# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load and resample tabular datasets from local and remote sources."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Hashable
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from decimal import Decimal, InvalidOperation
from glob import glob, has_magic
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Iterator, Mapping, Optional, Sequence, Union
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from anchor_bench.data.datasets import (
    download_from_huggingface,
    download_huggingface_dataset,
)
from anchor_bench.errors import DatasetAlignmentError, DatasetLoadError
from anchor_bench.schema.dataset import HuggingFaceDatasetSource

if TYPE_CHECKING:
    from anchor_bench.mlops import MLOpsRun

DEFAULT_HTTP_TIMEOUT_SECONDS = 30.0
NANOSECONDS_PER_SECOND = 1_000_000_000
REQUIRED_COLUMNS = {"time_ns", "experiment", "field", "value"}

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _MaterializedDatasetFile:
    path: Path
    source: str
    metadata: dict[str, str] = dataclass_field(default_factory=dict)


class DatasetLoader:
    """Load a Parquet dataset and resample its signals."""

    @staticmethod
    def load(
        source: Union[str, Path, HuggingFaceDatasetSource],
        *,
        dt: float,
        http_timeout_seconds: float = DEFAULT_HTTP_TIMEOUT_SECONDS,
        dataset_tracker: Optional[MLOpsRun] = None,
        dataset_name: str = "dataset",
        dataset_context: str = "input",
        upload_dataset: bool = False,
        joint_name_mapping: Optional[Mapping[str, str]] = None,
        feature_name_mapping: Optional[Mapping[str, str]] = None,
    ) -> pd.DataFrame:
        """Acquire, optionally track, and resample one or more Parquet files.

        Each materialized file is reported before concatenation and resampling.
        Optional upload also happens here so remotely downloaded data still exists
        until the backend has finished consuming it.
        """

        with DatasetLoader._get_files(
            source,
            http_timeout_seconds=http_timeout_seconds,
        ) as dataset_files:
            datasets = []
            for dataset_file in dataset_files:
                logger.debug("Reading dataset file path=%s", dataset_file.path)
                dataset = DatasetLoader._read_file(
                    dataset_file.path,
                    dataset_file.source,
                )
                datasets.append(dataset)
                if dataset_tracker is not None:
                    file_name = dataset_file.metadata.get(
                        "repo_path", dataset_file.path.stem
                    )
                    tracked_name = (
                        dataset_name
                        if len(dataset_files) == 1
                        else f"{dataset_name}-{file_name}"
                    )
                    # Fingerprint the materialized file rather than the DataFrame so
                    # the digest identifies exactly the bytes a later run can reuse.
                    metadata: dict[str, Union[str, int]] = {
                        "digest_algorithm": "sha256",
                        "format": "parquet",
                        "size_bytes": dataset_file.path.stat().st_size,
                    }
                    metadata.update(dataset_file.metadata)
                    dataset_tracker.log_dataset(
                        dataset,
                        source=dataset_file.source,
                        digest=DatasetLoader._sha256(
                            dataset_file.path,
                            dataset_file.source,
                        ),
                        name=tracked_name,
                        context=dataset_context,
                        path=dataset_file.path,
                        metadata=metadata,
                        upload=upload_dataset,
                    )
        dataset = pd.concat(datasets, ignore_index=True)
        resampled = DatasetLoader._resample(
            dataset,
            dt,
            joint_name_mapping=joint_name_mapping,
            feature_name_mapping=feature_name_mapping,
        )
        logger.info(
            "Loaded dataset source=%s files=%d rows=%d resampled_rows=%d dt=%s",
            source,
            len(datasets),
            len(dataset.index),
            len(resampled.index),
            dt,
        )
        return resampled

    @staticmethod
    @contextmanager
    def _get_files(
        source: Union[str, Path, HuggingFaceDatasetSource],
        *,
        http_timeout_seconds: float,
    ) -> Iterator[Sequence[_MaterializedDatasetFile]]:
        """Yield validated local files, downloading remote data when needed."""

        if isinstance(source, HuggingFaceDatasetSource):
            files = download_huggingface_dataset(source)
            yield [
                _MaterializedDatasetFile(
                    path=file.path,
                    source=(
                        f"hf://datasets/{source.repo_id}@{source.revision}/"
                        f"{file.repo_path}"
                    ),
                    metadata={
                        "provider": source.provider,
                        "repo_id": source.repo_id,
                        "revision": source.revision,
                        "repo_path": file.repo_path,
                    },
                )
                for file in files
            ]
            return

        source_text = str(source)
        scheme = urlparse(source_text).scheme
        if scheme in ("http", "https"):
            if has_magic(source_text):
                raise DatasetLoadError("URL dataset sources do not support globbing.")
            # TODO: We should also cache these downloads at some point.
            with TemporaryDirectory(prefix="anchor-bench-dataset-") as folder:
                path = download_from_huggingface(
                    source_text,
                    folder,
                    http_timeout_seconds=http_timeout_seconds,
                )
                yield [
                    _MaterializedDatasetFile(
                        path=path,
                        source=source_text,
                    )
                ]
            return
        paths = DatasetLoader._local_paths(source)
        metadata = {"source_pattern": source_text} if has_magic(source_text) else {}
        yield [
            _MaterializedDatasetFile(
                path=path,
                source=str(path),
                metadata=metadata,
            )
            for path in paths
        ]

    @staticmethod
    def _local_paths(source: Union[str, Path]) -> list[Path]:
        """Expand and validate a local file path or glob pattern."""

        source_text = str(source)
        if not has_magic(source_text):
            return [DatasetLoader._local_path(source)]
        expanded_pattern = str(Path(source).expanduser())
        matches = sorted(
            path
            for candidate in glob(expanded_pattern, recursive=True)
            if (path := Path(candidate)).is_file()
        )
        if not matches:
            raise DatasetLoadError(
                f"Dataset glob did not match any files: {expanded_pattern}."
            )
        return matches

    @staticmethod
    def _read_file(
        dataset_file: Path,
        source: Union[str, Path],
    ) -> pd.DataFrame:
        """Read a local Parquet file into memory."""

        try:
            return pd.read_parquet(dataset_file)
        except Exception as error:
            raise DatasetLoadError(
                f"Could not load Parquet dataset from {source}: {error}."
            ) from error

    @staticmethod
    def _sha256(dataset_file: Path, source: Union[str, Path]) -> str:
        """Return the SHA-256 digest for one materialized dataset file."""

        digest = hashlib.sha256()
        try:
            with dataset_file.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
        except OSError as error:
            raise DatasetLoadError(
                f"Could not fingerprint Parquet dataset from {source}: {error}."
            ) from error
        return digest.hexdigest()

    @staticmethod
    def _local_path(source: Union[str, Path]) -> Path:
        """Validate and expand a local dataset path."""

        expanded_path = Path(source).expanduser()
        if not expanded_path.is_file():
            raise DatasetLoadError(
                f"Dataset file does not exist or is not a file: {expanded_path}."
            )
        return expanded_path

    @staticmethod
    def _resample(
        dataset: pd.DataFrame,
        dt: float,
        *,
        joint_name_mapping: Optional[Mapping[str, str]] = None,
        feature_name_mapping: Optional[Mapping[str, str]] = None,
    ) -> pd.DataFrame:
        """Resample every experiment onto its own uniform time grid."""

        dt_ns = DatasetLoader._validate_resampling(dataset, dt)
        if joint_name_mapping or feature_name_mapping:
            dataset = DatasetLoader._map_field_names(
                dataset,
                joint_name_mapping or {},
                feature_name_mapping or {},
            )
        dataset = dataset.copy(deep=False)
        dataset["field"] = dataset["field"].astype("category")
        resampled_experiments: list[pd.DataFrame] = []

        for experiment, episode in dataset.groupby("experiment", sort=False):
            resampled_experiments.append(
                DatasetLoader._resample_experiment(
                    episode,
                    experiment,
                    dt_ns,
                )
            )

        if not resampled_experiments:
            raise DatasetAlignmentError("Dataset does not contain any experiments.")

        return pd.concat(resampled_experiments)

    @staticmethod
    def _map_field_names(
        dataset: pd.DataFrame,
        joint_name_mapping: Mapping[str, str],
        feature_name_mapping: Mapping[str, str],
    ) -> pd.DataFrame:
        """Translate configured joint prefixes and feature suffixes."""

        def mapped_field(field: object) -> object:
            if not isinstance(field, str):
                return field
            joint_name, separator, feature = field.rpartition("/")
            mapped_feature = feature_name_mapping.get(feature, feature)
            if not separator:
                return mapped_feature
            mapped_joint = joint_name_mapping.get(joint_name, joint_name)
            return f"{mapped_joint}/{mapped_feature}"

        source_fields = dataset["field"]
        unique_fields = source_fields.drop_duplicates()
        mapped_unique_fields = unique_fields.map(mapped_field)
        if mapped_unique_fields.equals(unique_fields):
            return dataset

        field_mapping: dict[object, object] = dict(
            zip(unique_fields.tolist(), mapped_unique_fields.tolist())
        )
        if mapped_unique_fields.duplicated().any():
            field_pairs = pd.DataFrame(
                {
                    "experiment": dataset["experiment"],
                    "field": dataset["field"],
                }
            ).drop_duplicates()
            field_pairs["mapped"] = field_pairs["field"].map(field_mapping.get)
            collisions = field_pairs.duplicated(
                subset=["experiment", "mapped"],
                keep=False,
            )
            if collisions.any():
                names = ", ".join(
                    sorted(map(str, field_pairs.loc[collisions, "mapped"].unique()))
                )
                raise DatasetAlignmentError(
                    "Joint or feature name mapping produces conflicting fields: "
                    + names
                    + "."
                )
        mapped_dataset = dataset.copy()
        mapped_dataset["field"] = source_fields.map(field_mapping.get)
        return mapped_dataset

    @staticmethod
    def _validate_resampling(
        dataset: pd.DataFrame,
        dt: float,
    ) -> int:
        """Validate the data required for safe interpolation."""

        missing_columns = REQUIRED_COLUMNS.difference(dataset.columns)
        if missing_columns:
            names = ", ".join(sorted(missing_columns))
            raise DatasetAlignmentError(
                f"Dataset is missing required columns: {names}."
            )
        if dataset.loc[:, list(REQUIRED_COLUMNS)].isnull().any().any():
            raise DatasetAlignmentError(
                "Dataset alignment columns cannot contain nulls."
            )
        if not pd.api.types.is_integer_dtype(dataset["time_ns"]):
            raise DatasetAlignmentError("time_ns must contain integer nanoseconds.")
        if not pd.api.types.is_numeric_dtype(dataset["value"]):
            raise DatasetAlignmentError("value must contain numeric samples.")

        return DatasetLoader._dt_nanoseconds(dt)

    @staticmethod
    def _dt_nanoseconds(dt: float) -> int:
        """Convert a positive interval in seconds to whole nanoseconds."""

        try:
            dt_ns = Decimal(str(dt)) * NANOSECONDS_PER_SECOND
        except (InvalidOperation, ValueError) as error:
            raise DatasetAlignmentError(
                "dt must be a positive number of seconds."
            ) from error

        if not dt_ns.is_finite() or dt_ns <= 0:
            raise DatasetAlignmentError("dt must be a positive number of seconds.")
        if dt_ns != dt_ns.to_integral_value():
            raise DatasetAlignmentError(
                "dt must resolve to a whole number of nanoseconds."
            )
        return int(dt_ns)

    @staticmethod
    def _resample_experiment(
        episode: pd.DataFrame,
        experiment: object,
        dt_ns: int,
    ) -> pd.DataFrame:
        """Resample one experiment over the intersection of all signals."""

        if episode.duplicated(subset=["time_ns", "field"]).any():
            raise DatasetAlignmentError(
                f"Experiment {experiment!r} contains duplicate field timestamps."
            )

        signals: dict[Hashable, pd.DataFrame] = {}
        for field, group in episode.groupby(
            "field",
            sort=False,
            observed=True,
        ):
            signal: pd.DataFrame = group.loc[:, ["time_ns", "value"]]
            signals[field] = signal.sort_values(by=["time_ns"])

        start = max(int(signal["time_ns"].iloc[0]) for signal in signals.values())
        end = min(int(signal["time_ns"].iloc[-1]) for signal in signals.values())
        if start > end:
            raise DatasetAlignmentError(
                f"Experiment {experiment!r} has no overlapping signal range."
            )

        sample_count = ((end - start) // dt_ns) + 1
        target_time = start + np.arange(sample_count, dtype=np.int64) * dt_ns
        target_offset = (target_time - start).astype(np.float64)
        resampled_values: dict[Hashable, np.ndarray] = {}

        for field, signal in signals.items():
            source_time = signal["time_ns"].to_numpy(dtype=np.int64)
            source_value = signal["value"].to_numpy(dtype=np.float64)
            source_offset = (source_time - start).astype(np.float64)
            resampled_values[field] = np.interp(
                target_offset,
                source_offset,
                source_value,
            )

        index = pd.MultiIndex.from_arrays(
            [[experiment] * len(target_time), target_time],
            names=["experiment", "time_ns"],
        )
        return pd.DataFrame(resampled_values, index=index)
