"""PyTorch datasets and data loaders for mapped training sequences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from anchor_bench.schema.training import TrainingConfig
from anchor_bench.tasks.train.task import MappedTrainingSequence


class SequenceDataset(Dataset[tuple[Tensor, Tensor]]):
    """Expose fixed-length windows without crossing sequence boundaries."""

    def __init__(
        self,
        sequences: Sequence[MappedTrainingSequence],
        *,
        window_length: int,
        stride: int,
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

            inputs = torch.from_numpy(
                sequence.inputs.to_numpy(dtype=np.float32, copy=True)
            )
            targets = torch.from_numpy(
                sequence.targets.to_numpy(dtype=np.float32, copy=True)
            )
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


@dataclass(frozen=True)
class TrainingDataLoaders:
    """Hold training and validation data loaders."""

    train: DataLoader[tuple[Tensor, Tensor]]
    validation: DataLoader[tuple[Tensor, Tensor]]


def create_data_loaders(
    sequences: Sequence[MappedTrainingSequence],
    config: TrainingConfig,
) -> TrainingDataLoaders:
    """Split mapped timesteps, window them, and build configured data loaders."""

    training = config.training
    train_sequences, validation_sequences = _split_sequences(
        sequences,
        validation_ratio=float(training.validation_ratio),
    )
    dataset_arguments = {
        "window_length": training.sequence.length,
        "stride": training.sequence.stride,
    }
    train_dataset = SequenceDataset(train_sequences, **dataset_arguments)
    validation_dataset = SequenceDataset(validation_sequences, **dataset_arguments)
    generator = torch.Generator().manual_seed(training.seed)

    return TrainingDataLoaders(
        train=DataLoader(
            train_dataset,
            batch_size=training.batch_size,
            shuffle=len(train_dataset) > 0,
            generator=generator,
        ),
        validation=DataLoader(
            validation_dataset,
            batch_size=training.batch_size,
            shuffle=False,
        ),
    )


def _split_sequences(
    sequences: Sequence[MappedTrainingSequence],
    *,
    validation_ratio: float,
) -> tuple[list[MappedTrainingSequence], list[MappedTrainingSequence]]:
    """Split each sequence before windowing so train and validation never overlap."""

    train_sequences = []
    validation_sequences = []
    for sequence in sequences:
        split = int(len(sequence.inputs) * (1.0 - validation_ratio))
        train_sequences.append(_slice_sequence(sequence, stop=split))
        validation_sequences.append(_slice_sequence(sequence, start=split))
    return train_sequences, validation_sequences


def _slice_sequence(
    sequence: MappedTrainingSequence,
    *,
    start: int | None = None,
    stop: int | None = None,
) -> MappedTrainingSequence:
    """Create a mapped sequence view over one contiguous timestep range."""

    return MappedTrainingSequence(
        experiment=sequence.experiment,
        name=sequence.name,
        inputs=sequence.inputs.iloc[start:stop],
        targets=sequence.targets.iloc[start:stop],
    )
