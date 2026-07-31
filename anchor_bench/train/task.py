"""Training task implementation."""

from __future__ import annotations

from typing import Sequence, Tuple

import pandas as pd


class TrainingTask:
    """Prepare aligned datasets for network training."""

    @staticmethod
    def _map_features(
        dataset: pd.DataFrame,
        *,
        input_features: Sequence[str],
        output_feature: str,
        input_entities: Sequence[str],
        output_entities: Sequence[str],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Select feature-major inputs and entity-ordered training targets."""

        input_columns = [
            f"{entity}/{feature}"
            for feature in input_features
            for entity in input_entities
        ]
        output_columns = [f"{entity}/{output_feature}" for entity in output_entities]
        missing_columns = [
            column
            for column in input_columns + output_columns
            if column not in dataset.columns
        ]
        if missing_columns:
            names = ", ".join(missing_columns)
            raise ValueError(f"Dataset is missing mapped feature columns: {names}.")

        inputs = dataset.loc[:, input_columns].copy()
        targets = dataset.loc[:, output_columns].copy()
        return inputs, targets
