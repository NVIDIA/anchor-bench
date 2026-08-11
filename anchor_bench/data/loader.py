"""Load and resample tabular datasets from local and remote sources."""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from decimal import Decimal, InvalidOperation
from glob import glob, has_magic
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Iterator, Optional, Sequence, Union
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from anchor_bench.data.datasets import download_from_huggingface
from anchor_bench.errors import DatasetAlignmentError, DatasetLoadError

if TYPE_CHECKING:
    from anchor_bench.mlops import MLOpsRun

DEFAULT_HTTP_TIMEOUT_SECONDS = 30.0
NANOSECONDS_PER_SECOND = 1_000_000_000
REQUIRED_COLUMNS = {"time_ns", "experiment", "field", "value"}


class DatasetLoader:
    """Load a Parquet dataset and resample its signals."""

    @staticmethod
    def load(
        source: Union[str, Path],
        *,
        dt: float,
        http_timeout_seconds: float = DEFAULT_HTTP_TIMEOUT_SECONDS,
        dataset_tracker: Optional[MLOpsRun] = None,
        dataset_name: str = "dataset",
        dataset_context: str = "input",
        upload_dataset: bool = False,
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
            for index, dataset_file in enumerate(dataset_files):
                dataset = DatasetLoader._read_file(dataset_file, source)
                datasets.append(dataset)
                if dataset_tracker is not None:
                    tracked_name = (
                        dataset_name
                        if len(dataset_files) == 1
                        else f"{dataset_name}-{index}"
                    )
                    # Fingerprint the materialized file rather than the DataFrame so
                    # the digest identifies exactly the bytes a later run can reuse.
                    dataset_tracker.log_dataset(
                        dataset,
                        source=str(source),
                        digest=DatasetLoader._sha256(dataset_file, source),
                        name=tracked_name,
                        context=dataset_context,
                        path=dataset_file,
                        metadata={
                            "digest_algorithm": "sha256",
                            "format": "parquet",
                            "size_bytes": dataset_file.stat().st_size,
                        },
                        upload=upload_dataset,
                    )
        dataset = pd.concat(datasets, ignore_index=True)
        return DatasetLoader._resample(dataset, dt)

    @staticmethod
    @contextmanager
    def _get_files(
        source: Union[str, Path],
        *,
        http_timeout_seconds: float,
    ) -> Iterator[Sequence[Path]]:
        """Yield validated local files, downloading remote data when needed."""

        source_text = str(source)
        scheme = urlparse(source_text).scheme
        if scheme in ("http", "https"):
            if has_magic(source_text):
                raise DatasetLoadError("URL dataset sources do not support globbing.")
            # TODO: We should also cache these downloads at some point.
            with TemporaryDirectory(prefix="anchor-bench-dataset-") as folder:
                yield [
                    download_from_huggingface(
                        source_text,
                        folder,
                        http_timeout_seconds=http_timeout_seconds,
                    )
                ]
            return
        yield DatasetLoader._local_paths(source)

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
    def _resample(dataset: pd.DataFrame, dt: float) -> pd.DataFrame:
        """Resample every experiment onto its own uniform time grid."""

        dt_ns = DatasetLoader._validate_resampling(dataset, dt)
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

        fields = list(dict.fromkeys(episode["field"].tolist()))
        if episode.duplicated(subset=["time_ns", "field"]).any():
            raise DatasetAlignmentError(
                f"Experiment {experiment!r} contains duplicate field timestamps."
            )

        signals = {}
        for field in fields:
            signal = episode.loc[
                episode["field"] == field,
                ["time_ns", "value"],
            ]
            signals[field] = signal.sort_values(by="time_ns")

        start = max(int(signal["time_ns"].iloc[0]) for signal in signals.values())
        end = min(int(signal["time_ns"].iloc[-1]) for signal in signals.values())
        if start > end:
            raise DatasetAlignmentError(
                f"Experiment {experiment!r} has no overlapping signal range."
            )

        sample_count = ((end - start) // dt_ns) + 1
        target_time = start + np.arange(sample_count, dtype=np.int64) * dt_ns
        target_offset = (target_time - start).astype(np.float64)
        resampled_values: dict[str, np.ndarray] = {}

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
