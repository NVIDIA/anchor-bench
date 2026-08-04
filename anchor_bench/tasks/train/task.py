"""Training task implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import pandas as pd

from anchor_bench.core.simulation.registry import create_simulation_context
from anchor_bench.data import DatasetLoader
from anchor_bench.schema.training import TrainingConfig
from anchor_bench.tasks.base_task import BaseTask
from anchor_bench.tasks.train.preprocessor_runner import PreprocessorRunner


@dataclass(frozen=True)
class EntityMapping:
    """Describe one ordered input-to-output entity mapping."""

    inputs: Sequence[str]
    outputs: Sequence[str]


@dataclass(frozen=True)
class MappedTrainingSequence:
    """Preserve one experiment and mapping as a recurrent sequence."""

    experiment: object
    name: str
    inputs: pd.DataFrame
    targets: pd.DataFrame


class TrainingTask(BaseTask[TrainingConfig]):
    """Prepare aligned datasets for network training."""

    def __init__(self, config: TrainingConfig) -> None:
        """Initialize training-data preparation from validated configuration."""

        self.config = config

    @classmethod
    def name(cls) -> str:
        """Return the training configuration discriminator."""

        return "training"

    def run(self) -> list[MappedTrainingSequence]:
        """Load, preprocess, and map configured training sequences."""

        dataset = DatasetLoader.load(
            self.config.dataset.source,
            dt=float(self.config.dataset.resampling.dt),
        )

        if any(self.config.input_features.values()):
            simulation = self.config.simulation
            create_simulation_context(simulation.backend, simulation.usd_path)

        dataset = PreprocessorRunner.run(dataset, self.config.input_features)
        mappings = {
            name: EntityMapping(inputs=mapping.inputs, outputs=mapping.outputs)
            for name, mapping in self.config.mapping.items()
        }
        return self._map_features(
            dataset,
            input_features=list(self.config.input_features),
            output_feature=self.config.output_features,
            mappings=mappings,
        )

    @staticmethod
    def _map_features(
        dataset: pd.DataFrame,
        *,
        input_features: Sequence[str],
        output_feature: str,
        mappings: Mapping[str, EntityMapping],
    ) -> list[MappedTrainingSequence]:
        """Prepare one ordered training sequence for every configured mapping."""

        TrainingTask._validate_map_features(dataset, mappings=mappings)
        resolved_mappings = []
        for name, mapping in mappings.items():
            input_columns, output_columns = TrainingTask._resolve_mapping_columns(
                dataset,
                input_features=input_features,
                output_feature=output_feature,
                mapping=mapping,
            )
            resolved_mappings.append((name, input_columns, output_columns))

        mapped_sequences = []
        for experiment, episode in dataset.groupby(level="experiment", sort=False):
            for name, input_columns, output_columns in resolved_mappings:
                inputs, targets = TrainingTask._select_mapped_columns(
                    episode,
                    input_columns=input_columns,
                    output_columns=output_columns,
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
    def _validate_map_features(
        dataset: pd.DataFrame,
        *,
        mappings: Mapping[str, EntityMapping],
    ) -> None:
        """Validate shared mapping shapes and recurrent sequence indexing."""

        if not mappings:
            raise ValueError("Training requires at least one entity mapping.")
        if "experiment" not in dataset.index.names:
            raise ValueError("Mapped training data requires an experiment index.")

        expected_shape = None
        for name, mapping in mappings.items():
            if not mapping.inputs or not mapping.outputs:
                raise ValueError(
                    f"Mapping {name!r} requires at least one input and output entity."
                )

            shape = (len(mapping.inputs), len(mapping.outputs))
            if expected_shape is None:
                expected_shape = shape
            elif shape != expected_shape:
                raise ValueError(
                    "Shared mappings must have identical input and output widths; "
                    f"mapping {name!r} has shape {shape[0]}->{shape[1]}, expected "
                    f"{expected_shape[0]}->{expected_shape[1]}."
                )

    @staticmethod
    def _resolve_mapping_columns(
        dataset: pd.DataFrame,
        *,
        input_features: Sequence[str],
        output_feature: str,
        mapping: EntityMapping,
    ) -> tuple[list[str], list[str]]:
        """Resolve and validate one mapping's ordered dataset columns."""

        input_columns = [
            f"{entity}/{feature}"
            for feature in input_features
            for entity in mapping.inputs
        ]
        output_columns = [f"{entity}/{output_feature}" for entity in mapping.outputs]
        required_columns = list(dict.fromkeys(input_columns + output_columns))
        missing_columns = [
            column for column in required_columns if column not in dataset.columns
        ]
        if missing_columns:
            names = ", ".join(missing_columns)
            raise ValueError(f"Dataset is missing mapped feature columns: {names}.")
        return input_columns, output_columns

    @staticmethod
    def _select_mapped_columns(
        dataset: pd.DataFrame,
        *,
        input_columns: Sequence[str],
        output_columns: Sequence[str],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Copy mapped inputs and targets from one experiment."""

        inputs = dataset.loc[:, input_columns].copy()
        targets = dataset.loc[:, output_columns].copy()
        return inputs, targets
