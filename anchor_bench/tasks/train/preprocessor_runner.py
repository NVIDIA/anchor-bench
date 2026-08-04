"""Execute configured training preprocessors."""

from __future__ import annotations

from typing import Mapping, Optional

import pandas as pd

import anchor_bench.train.preprocessors as preprocessor_registry


class PreprocessorRunner:
    """Execute configured built-in preprocessors in declaration order."""

    @staticmethod
    def run(
        dataset: pd.DataFrame,
        preprocessors: Mapping[str, Optional[str]],
    ) -> pd.DataFrame:
        """Return a shallow dataset copy with generated features appended."""

        result = dataset.copy(deep=False)
        for feature, preprocessor_name in preprocessors.items():
            if preprocessor_name is None:
                continue

            preprocessor = getattr(preprocessor_registry, preprocessor_name)
            generated = PreprocessorRunner._validate_result(
                result,
                preprocessor(result).process(),
                feature,
                preprocessor_name,
            )

            result[generated.columns] = generated

        return result

    @staticmethod
    def _validate_result(
        dataset: pd.DataFrame,
        generated: object,
        feature: str,
        preprocessor_name: str,
    ) -> pd.DataFrame:
        """Validate generated columns before attaching them to the dataset."""

        if not isinstance(generated, pd.DataFrame):
            raise TypeError(
                f"Preprocessor {preprocessor_name!r} must return a DataFrame."
            )
        if generated.empty:
            raise ValueError(
                f"Preprocessor {preprocessor_name!r} returned an empty DataFrame."
            )
        if not generated.index.equals(dataset.index):
            raise ValueError(
                f"Preprocessor {preprocessor_name!r} returned a different index."
            )
        if not generated.columns.is_unique:
            raise ValueError(
                f"Preprocessor {preprocessor_name!r} returned duplicate columns."
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
                f"Preprocessor {preprocessor_name!r} returned columns unrelated "
                f"to feature {feature!r}: {names}."
            )
        return generated
