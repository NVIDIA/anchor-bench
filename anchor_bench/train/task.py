"""Training task implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import pandas as pd


@dataclass(frozen=True)
class EntityMapping:
    """Describe one ordered input-to-output entity mapping."""

    inputs: Sequence[str]
    outputs: Sequence[str]


@dataclass(frozen=True)
class MappedTrainingSequence:
    """Hold one training sequence prepared for a model mapping."""

    experiment: object
    name: str
    inputs: pd.DataFrame
    targets: pd.DataFrame


class TrainingTask:
    """Prepare aligned datasets for network training."""

    @staticmethod
    def _map_features(
        dataset: pd.DataFrame,
        *,
        input_features: Sequence[str],
        output_feature: str,
        mappings: Mapping[str, EntityMapping],
    ) -> list[MappedTrainingSequence]:
        """Prepare one ordered training sequence for every configured mapping."""

        if not mappings:
            raise ValueError("Training requires at least one entity mapping.")
        if "experiment" not in dataset.index.names:
            raise ValueError("Mapped training data requires an experiment index.")

        ordered_mappings: list[tuple[str, list[str], list[str]]] = []
        expected_shape = None
        for name, mapping in mappings.items():
            input_entities = list(mapping.inputs)
            output_entities = list(mapping.outputs)
            if not input_entities or not output_entities:
                raise ValueError(
                    f"Mapping {name!r} requires at least one input and output entity."
                )

            shape = (len(input_entities), len(output_entities))
            if expected_shape is None:
                expected_shape = shape
            elif shape != expected_shape:
                raise ValueError(
                    "Shared mappings must have identical input and output widths; "
                    f"mapping {name!r} has shape {shape[0]}->{shape[1]}, expected "
                    f"{expected_shape[0]}->{expected_shape[1]}."
                )
            ordered_mappings.append((name, input_entities, output_entities))

        mapped_sequences = []
        for experiment, episode in dataset.groupby(level="experiment", sort=False):
            for name, input_entities, output_entities in ordered_mappings:
                inputs, targets = TrainingTask._map_one(
                    episode,
                    input_features=input_features,
                    output_feature=output_feature,
                    input_entities=input_entities,
                    output_entities=output_entities,
                )
                mapped_sequences.append(
                    MappedTrainingSequence(
                        experiment=experiment,
                        name=name,
                        inputs=inputs,
                        targets=targets,
                    )
                )
        return mapped_sequences

    @staticmethod
    def _map_one(
        dataset: pd.DataFrame,
        *,
        input_features: Sequence[str],
        output_feature: str,
        input_entities: Sequence[str],
        output_entities: Sequence[str],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Select one mapping in feature-major and entity-configured order."""

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
