"""Load and align tabular datasets from local and remote sources."""

from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

import numpy as np
import pandas as pd

from anchor_bench.errors import DatasetAlignmentError, DatasetLoadError

DEFAULT_HTTP_TIMEOUT_SECONDS = 30.0
REQUIRED_COLUMNS = {"time_ns", "experiment", "field", "value"}
WINDOWS_DRIVE_PATH = re.compile(r"^[A-Za-z]:[\\/](?![\\/])")


class DatasetLoader:
    """Load a Parquet dataset and align its signals."""

    @staticmethod
    def load(
        source: str,
        *,
        reference_field: str,
        http_timeout_seconds: float = DEFAULT_HTTP_TIMEOUT_SECONDS,
    ) -> pd.DataFrame:
        """Load a dataset and align it to its recorded reference clock.

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

        return DatasetLoader._align(dataset, reference_field)

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
    def _align(dataset: pd.DataFrame, reference_field: str) -> pd.DataFrame:
        """Align every experiment to its recorded reference clock."""

        DatasetLoader._validate_alignment(dataset, reference_field)
        aligned_experiments: list[pd.DataFrame] = []

        for experiment, episode in dataset.groupby("experiment", sort=False):
            aligned_experiments.append(
                DatasetLoader._align_experiment(
                    episode,
                    experiment,
                    reference_field,
                )
            )

        if not aligned_experiments:
            raise DatasetAlignmentError("Dataset does not contain any experiments.")

        return pd.concat(aligned_experiments)

    @staticmethod
    def _validate_alignment(
        dataset: pd.DataFrame,
        reference_field: str,
    ) -> None:
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
        if not reference_field:
            raise DatasetAlignmentError("reference_field cannot be empty.")
        if not pd.api.types.is_integer_dtype(dataset["time_ns"]):
            raise DatasetAlignmentError("time_ns must contain integer nanoseconds.")
        if not pd.api.types.is_numeric_dtype(dataset["value"]):
            raise DatasetAlignmentError("value must contain numeric samples.")

    @staticmethod
    def _align_experiment(
        episode: pd.DataFrame,
        experiment: object,
        reference_field: str,
    ) -> pd.DataFrame:
        """Align one experiment to its recorded reference timestamps."""

        fields = list(dict.fromkeys(episode["field"].tolist()))
        reference_fields = [
            field
            for field in fields
            if field == reference_field or field.endswith(f"/{reference_field}")
        ]
        if not reference_fields:
            raise DatasetAlignmentError(
                f"Experiment {experiment!r} has no reference field "
                f"matching {reference_field!r}."
            )
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

        reference_time = signals[reference_fields[0]]["time_ns"].to_numpy(
            dtype=np.int64
        )
        for field in reference_fields[1:]:
            timestamps = signals[field]["time_ns"].to_numpy(dtype=np.int64)
            if not np.array_equal(timestamps, reference_time):
                raise DatasetAlignmentError(
                    f"Experiment {experiment!r} reference fields do not share "
                    "the same timestamps."
                )

        start = max(int(signal["time_ns"].iloc[0]) for signal in signals.values())
        end = min(int(signal["time_ns"].iloc[-1]) for signal in signals.values())
        target_time = reference_time[
            (reference_time >= start) & (reference_time <= end)
        ]
        if not len(target_time):
            raise DatasetAlignmentError(
                f"Experiment {experiment!r} has no overlapping signal range."
            )

        origin_ns = start
        target_offset = (target_time - origin_ns).astype(np.float64)
        aligned_values: dict[str, np.ndarray] = {}

        for field, signal in signals.items():
            source_time = signal["time_ns"].to_numpy(dtype=np.int64)
            source_value = signal["value"].to_numpy(dtype=np.float64)
            if field in reference_fields:
                indexed_values = pd.Series(source_value, index=source_time)
                aligned_values[field] = indexed_values.loc[target_time].to_numpy()
            else:
                source_offset = (source_time - origin_ns).astype(np.float64)
                aligned_values[field] = np.interp(
                    target_offset,
                    source_offset,
                    source_value,
                )

        index = pd.MultiIndex.from_arrays(
            [[experiment] * len(target_time), target_time],
            names=["experiment", "time_ns"],
        )
        return pd.DataFrame(aligned_values, index=index)
