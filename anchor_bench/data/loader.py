"""Load and resample tabular datasets from local and remote sources."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

import numpy as np
import pandas as pd

from anchor_bench.errors import DatasetAlignmentError, DatasetLoadError

DEFAULT_HTTP_TIMEOUT_SECONDS = 30.0
NANOSECONDS_PER_SECOND = 1_000_000_000
REQUIRED_COLUMNS = {"time_ns", "experiment", "field", "value"}
WINDOWS_DRIVE_PATH = re.compile(r"^[A-Za-z]:[\\/](?![\\/])")


class DatasetLoader:
    """Load a Parquet dataset and resample its signals."""

    @staticmethod
    def load(
        source: str,
        *,
        dt: float,
        http_timeout_seconds: float = DEFAULT_HTTP_TIMEOUT_SECONDS,
    ) -> pd.DataFrame:
        """Load a dataset and resample it to a uniform interval.

        Remote URLs must return Parquet bytes directly.
        """

        scheme = urlparse(source).scheme
        is_remote = scheme in ("http", "https")
        is_windows_path = WINDOWS_DRIVE_PATH.match(source) is not None
        if scheme and not is_remote and not is_windows_path:
            raise DatasetLoadError(
                f"Unsupported dataset URL scheme {scheme!r}: {source}."
            )

        try:
            if is_remote:
                with urlopen(source, timeout=http_timeout_seconds) as response:
                    read_source = BytesIO(response.read())
            else:
                read_source = DatasetLoader._local_path(source)
            dataset = pd.read_parquet(read_source)
        except DatasetLoadError:
            raise
        except Exception as error:
            raise DatasetLoadError(
                f"Could not load Parquet dataset from {source}: {error}."
            ) from error

        return DatasetLoader._resample(dataset, dt)

    @staticmethod
    def _local_path(source: str) -> Path:
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
