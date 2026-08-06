"""PyTorch datasets and data loaders for mapped training sequences."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from anchor_bench.schema.training import TrainingConfig
from anchor_bench.tasks.train.normalization import (
    FeatureNormalization,
    NormalizationStatistics,
    fit_normalization,
)
from anchor_bench.tasks.train.task import MappedTrainingSequence


class SequenceDataset(Dataset[tuple[Tensor, Tensor]]):
    """Expose fixed-length windows without crossing sequence boundaries."""

    def __init__(
        self,
        sequences: Sequence[MappedTrainingSequence],
        *,
        window_length: int,
        stride: int,
        input_normalization: FeatureNormalization | None = None,
        target_normalization: FeatureNormalization | None = None,
    ) -> None:
        """Convert mapped sequences to tensors and index their valid windows."""

        self._sequences: list[tuple[Tensor, Tensor]] = []
        self._window_indices: list[tuple[int, int]] = []
        expected_shape: tuple[int, int] | None = None

        for sequence in sequences:
            if not sequence.inputs.index.equals(sequence.targets.index):
                raise ValueError(
                    f"Sequence {sequence.name!r} inputs and targets must share "
                    "the same index."
                )

            shape = (sequence.inputs.shape[1], sequence.targets.shape[1])
            if expected_shape is None:
                expected_shape = shape
            elif shape != expected_shape:
                raise ValueError(
                    "All training sequences must have identical input and target "
                    f"widths; sequence {sequence.name!r} has shape "
                    f"{shape[0]}->{shape[1]}, expected "
                    f"{expected_shape[0]}->{expected_shape[1]}."
                )

            input_values = sequence.inputs.to_numpy(dtype=np.float64, copy=True)
            target_values = sequence.targets.to_numpy(dtype=np.float64, copy=True)
            if input_normalization is not None:
                input_values = input_normalization.normalize_array(input_values)
            if target_normalization is not None:
                target_values = target_normalization.normalize_array(target_values)
            inputs = torch.from_numpy(input_values.astype(np.float32, copy=False))
            targets = torch.from_numpy(target_values.astype(np.float32, copy=False))
            sequence_index = len(self._sequences)
            self._sequences.append((inputs, targets))
            self._window_indices.extend(
                (sequence_index, start)
                for start in range(
                    0,
                    len(inputs) - window_length + 1,
                    stride,
                )
            )

        self._window_length = window_length
        self.input_size, self.output_size = expected_shape or (0, 0)

    def __len__(self) -> int:
        """Return the number of complete sequence windows."""

        return len(self._window_indices)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """Return one input and target window as tensor views."""

        sequence_index, start = self._window_indices[index]
        inputs, targets = self._sequences[sequence_index]
        stop = start + self._window_length
        return inputs[start:stop], targets[start:stop]

    @property
    def sequences(self) -> tuple[tuple[Tensor, Tensor], ...]:
        """Return complete normalized sequences for stateful training."""

        return tuple(self._sequences)


@dataclass(frozen=True)
class TrainingDataLoaders:
    """Hold training and validation data loaders."""

    train: DataLoader[tuple[Tensor, Tensor]]
    validation: DataLoader[tuple[Tensor, Tensor]]
    validation_sequences: tuple[MappedTrainingSequence, ...]
    normalization: NormalizationStatistics


@dataclass(frozen=True)
class ValidationDataset:
    """Provide validators with model sequences and complete processed signals."""

    sequences: tuple[MappedTrainingSequence, ...]
    dataframe: pd.DataFrame


def create_validation_dataset(
    dataset: pd.DataFrame,
    sequences: Sequence[MappedTrainingSequence],
) -> ValidationDataset:
    """Select the complete processed rows used by held-out sequences."""

    indices: dict[object, pd.Index] = {}
    for sequence in sequences:
        previous = indices.setdefault(sequence.experiment, sequence.inputs.index)
        if not previous.equals(sequence.inputs.index):
            raise ValueError(
                f"Validation mappings for experiment {sequence.experiment!r} "
                "must share the same index."
            )

    frames = []
    for experiment, index in indices.items():
        if not index.isin(dataset.index).all():
            raise ValueError(
                f"Validation rows for experiment {experiment!r} are missing "
                "from the processed dataset."
            )
        frames.append(dataset.reindex(index))

    dataframe = pd.concat(frames) if frames else dataset.iloc[0:0]
    return ValidationDataset(tuple(sequences), dataframe)


def create_data_loaders(
    sequences: Sequence[MappedTrainingSequence],
    config: TrainingConfig,
) -> TrainingDataLoaders:
    """Split mapped timesteps, window them, and build configured data loaders."""

    training = config.training
    train_sequences, validation_sequences = _split_sequences(
        sequences,
        validation_ratio=float(training.validation_ratio),
        seed=training.seed,
    )
    normalization = fit_normalization(
        train_sequences,
        inputs=training.normalization.inputs,
        outputs=training.normalization.outputs,
    )
    dataset_arguments = {
        "window_length": training.sequence.length,
        "stride": training.sequence.length,
        "input_normalization": normalization.inputs,
        "target_normalization": normalization.outputs,
    }
    train_dataset = SequenceDataset(train_sequences, **dataset_arguments)
    validation_dataset = SequenceDataset(validation_sequences, **dataset_arguments)

    return TrainingDataLoaders(
        train=DataLoader(
            train_dataset,
            batch_size=training.batch_size,
            shuffle=False,
        ),
        validation=DataLoader(
            validation_dataset,
            batch_size=training.batch_size,
            shuffle=False,
        ),
        validation_sequences=tuple(validation_sequences),
        normalization=normalization,
    )


def _split_sequences(
    sequences: Sequence[MappedTrainingSequence],
    *,
    validation_ratio: float,
    seed: int,
) -> tuple[list[MappedTrainingSequence], list[MappedTrainingSequence]]:
    """Hold out complete experiments so recurrent trajectories never leak."""

    experiments = list(dict.fromkeys(sequence.experiment for sequence in sequences))
    if validation_ratio == 0.0:
        return list(sequences), []
    if len(experiments) < 2:
        return list(sequences), []
    random.Random(seed).shuffle(experiments)
    validation_count = max(1, round(len(experiments) * validation_ratio))
    validation_count = min(validation_count, len(experiments) - 1)
    validation_experiments = set(experiments[:validation_count])
    training = [
        sequence
        for sequence in sequences
        if sequence.experiment not in validation_experiments
    ]
    validation = [
        sequence
        for sequence in sequences
        if sequence.experiment in validation_experiments
    ]
    return training, validation
