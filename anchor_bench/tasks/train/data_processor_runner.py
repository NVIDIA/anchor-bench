"""Execute configured training data processors."""

from __future__ import annotations

from typing import Mapping, Optional

import pandas as pd


class DataProcessorRunner:
    """Execute configured built-in data processors in declaration order."""

    @staticmethod
    def run(
        dataset: pd.DataFrame,
        data_processors: Mapping[str, Optional[str]],
    ) -> pd.DataFrame:
        """Return a shallow dataset copy with generated features appended."""

        import anchor_bench.train.data_processors as data_processor_registry

        result = dataset.copy(deep=False)
        for feature, data_processor_name in data_processors.items():
            if data_processor_name is None:
                continue

            data_processor = getattr(data_processor_registry, data_processor_name)
            generated = DataProcessorRunner._validate_result(
                result,
                data_processor(result).process(),
                feature,
                data_processor_name,
            )

            result[generated.columns] = generated

        return result

    @staticmethod
    def _validate_result(
        dataset: pd.DataFrame,
        generated: object,
        feature: str,
        data_processor_name: str,
    ) -> pd.DataFrame:
        """Validate generated columns before attaching them to the dataset."""

        if not isinstance(generated, pd.DataFrame):
            raise TypeError(
                f"DataProcessor {data_processor_name!r} must return a DataFrame."
            )
        if generated.empty:
            raise ValueError(
                f"DataProcessor {data_processor_name!r} returned an empty DataFrame."
            )
        if not generated.index.equals(dataset.index):
            raise ValueError(
                f"DataProcessor {data_processor_name!r} returned a different index."
            )
        if not generated.columns.is_unique:
            raise ValueError(
                f"DataProcessor {data_processor_name!r} returned duplicate columns."
            )

        collisions = [
            column for column in generated.columns if column in dataset.columns
        ]
        if collisions:
            names = ", ".join(map(str, collisions))
            raise ValueError(f"Generated columns already exist: {names}.")

        unexpected = [
            column
            for column in generated.columns
            if not isinstance(column, str)
            or (column != feature and not column.endswith(f"/{feature}"))
        ]
        if unexpected:
            names = ", ".join(map(str, unexpected))
            raise ValueError(
                f"DataProcessor {data_processor_name!r} returned columns unrelated "
                f"to feature {feature!r}: {names}."
            )
        return generated
