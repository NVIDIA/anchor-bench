"""Generic supervised network trainer."""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from anchor_bench.schema.training import LossConfig, OptimizerConfig, TrainingConfig
from anchor_bench.tasks.train.data import SequenceDataset, create_data_loaders
from anchor_bench.tasks.train.model_registry import get_model
from anchor_bench.tasks.train.normalization import NormalizationStatistics
from anchor_bench.tasks.train.task import MappedTrainingSequence


@dataclass(frozen=True)
class EpochLoss:
    """Record mean training and optional validation loss for one epoch."""

    epoch: int
    training: float
    validation: Optional[float]


@dataclass(frozen=True)
class TrainingResult:
    """Return the trained model and its loss history."""

    model: nn.Module
    history: list[EpochLoss]
    normalization: NormalizationStatistics
    artifact_path: Optional[Path] = None


class Trainer:
    """Train a configured model on prepared sequence data."""

    def __init__(
        self,
        config: TrainingConfig,
        dataset: Sequence[MappedTrainingSequence],
        *,
        device: Optional[torch.device | str] = None,
        checkpoint_directory: Optional[Path] = None,
    ) -> None:
        """Create the model, loss function, optimizer, and data loaders."""

        _set_seed(config.training.seed)
        self.config = config
        self.checkpoint_directory = checkpoint_directory
        if config.training.checkpoint is not None and checkpoint_directory is None:
            raise ValueError("Checkpointing requires a checkpoint directory.")
        self.data_loaders = create_data_loaders(dataset, config)
        train_dataset = self.data_loaders.train.dataset
        if not isinstance(train_dataset, SequenceDataset):
            raise TypeError("Training loader must contain a SequenceDataset.")
        if not train_dataset.sequences:
            raise ValueError("Training data does not contain any mapped sequences.")

        self.device = torch.device(
            device
            if device is not None
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        self.model = get_model(
            config.model,
            input_size=train_dataset.input_size,
            output_size=train_dataset.output_size,
        ).to(self.device)
        self.loss_function = _create_loss(config.training.loss)
        self.optimizer = _create_optimizer(
            config.training.optimizer,
            self.model.parameters(),
        )
        feedback = config.training.feedback
        if feedback is None:
            self.feedback_indices = None
        else:
            inputs = tuple(
                index
                for index, column in enumerate(dataset[0].inputs.columns)
                if column.endswith(f"/{feedback.input}")
            )
            outputs = tuple(
                index
                for index, column in enumerate(dataset[0].targets.columns)
                if column.endswith(f"/{feedback.output}")
            )
            if not inputs or len(inputs) != len(outputs):
                raise ValueError(
                    "Feedback input and output must map to the same number of columns."
                )
            self.feedback_indices = (inputs, outputs)

    def train(self) -> TrainingResult:
        """Run the configured epoch loop and return its model and history."""

        history = []
        for epoch in range(1, self.config.training.epochs + 1):
            training_loss = self._train_epoch()
            validation_loss = self._validate_epoch()
            history.append(
                EpochLoss(
                    epoch=epoch,
                    training=training_loss,
                    validation=validation_loss,
                )
            )

            checkpoint = self.config.training.checkpoint
            if checkpoint is not None and epoch % checkpoint.every_n_epochs == 0:
                self._save_checkpoint(epoch, history)
        self.model.eval()
        return TrainingResult(
            model=self.model,
            history=history,
            normalization=self.data_loaders.normalization,
        )

    def _save_checkpoint(
        self,
        epoch: int,
        history: Sequence[EpochLoss],
    ) -> None:
        """Atomically save one resumable epoch checkpoint."""

        if self.checkpoint_directory is None:
            raise RuntimeError("Checkpoint directory unexpectedly disappeared.")
        self.checkpoint_directory.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_directory / f"epoch-{epoch:06d}.pt"
        temporary_path = path.with_suffix(".tmp")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": {
                    name: tensor.detach().cpu()
                    for name, tensor in self.model.state_dict().items()
                },
                "optimizer_state_dict": self.optimizer.state_dict(),
                "history": [asdict(entry) for entry in history],
                "normalization": asdict(self.data_loaders.normalization),
                "config": self.config.model_dump(mode="json", exclude_none=True),
            },
            temporary_path,
        )
        temporary_path.replace(path)

    def _train_epoch(self) -> float:
        """Update model weights while carrying state through each motion."""

        loss = self._run_stateful_epoch(
            self._sequence_dataset(self.data_loaders.train),
            training=True,
        )
        if loss is None:
            raise RuntimeError("Training dataset unexpectedly contains no chunks.")
        return loss

    def _validate_epoch(self) -> Optional[float]:
        """Measure validation loss while carrying state through each motion."""

        return self._run_stateful_epoch(
            self._sequence_dataset(self.data_loaders.validation),
            training=False,
        )

    def _run_stateful_epoch(
        self,
        dataset: SequenceDataset,
        *,
        training: bool,
    ) -> Optional[float]:
        """Run complete motions with state carried across contiguous chunks."""

        self.model.train(training)
        total_loss = 0.0
        chunk_count = 0
        chunk_length = self.config.training.sequence.length
        batch_size = self.config.training.batch_size
        noise = float(self.config.training.augmentation.input_noise_std)
        sequences_by_length: dict[int, list[tuple[Tensor, Tensor]]] = {}
        for sequence in dataset.sequences:
            sequences_by_length.setdefault(len(sequence[0]), []).append(sequence)

        with torch.set_grad_enabled(training):
            for sequences in sequences_by_length.values():
                for batch_start in range(0, len(sequences), batch_size):
                    batch = sequences[batch_start : batch_start + batch_size]
                    inputs = torch.stack([item[0] for item in batch]).to(self.device)
                    targets = torch.stack([item[1] for item in batch]).to(self.device)
                    state: Tensor | None = None
                    previous_physical: Tensor | None = None

                    for start in range(0, inputs.shape[1], chunk_length):
                        stop = start + chunk_length
                        input_chunk = inputs[:, start:stop]
                        target_chunk = targets[:, start:stop]
                        if training and noise > 0.0:
                            input_chunk = (
                                input_chunk + torch.randn_like(input_chunk) * noise
                            )

                        if training:
                            self.optimizer.zero_grad(set_to_none=True)
                        predictions, state, previous_physical = (
                            self._forward_with_state(
                                input_chunk,
                                state=state,
                                previous_physical=previous_physical,
                            )
                        )
                        loss = self.loss_function(predictions, target_chunk)
                        if training:
                            loss.backward()
                            nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                float(self.config.training.gradient.clip_norm),
                            )
                            self.optimizer.step()
                            state = state.detach()
                            if previous_physical is not None:
                                previous_physical = previous_physical.detach()

                        total_loss += float(loss.detach())
                        chunk_count += 1

        if chunk_count == 0:
            return None
        return total_loss / chunk_count

    @staticmethod
    def _sequence_dataset(
        data_loader: DataLoader[tuple[Tensor, Tensor]],
    ) -> SequenceDataset:
        """Return the sequence dataset stored in a configured loader."""

        dataset = data_loader.dataset
        if not isinstance(dataset, SequenceDataset):
            raise TypeError("Training loader must contain a SequenceDataset.")
        return dataset

    def _forward_with_state(
        self,
        inputs: Tensor,
        *,
        state: Tensor | None = None,
        previous_physical: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """Run a batch while preserving recurrent and feedback state."""

        feedback = self.config.training.feedback
        indices = self.feedback_indices
        if feedback is None or indices is None:
            output = self.model(inputs, state)
            if output.state is None:
                raise RuntimeError("GRU model did not return recurrent state.")
            return output.predictions, output.state, None

        input_indices, output_indices = indices
        normalization = self.data_loaders.normalization

        def selected(
            values: tuple[float, ...], selected_indices: tuple[int, ...]
        ) -> Tensor:
            """Select normalization values for configured feedback columns."""

            return inputs.new_tensor(tuple(values[index] for index in selected_indices))

        input_mean = selected(normalization.inputs.mean, input_indices)
        input_std = selected(
            normalization.inputs.standard_deviation,
            input_indices,
        )
        output_mean = selected(normalization.outputs.mean, output_indices)
        output_std = selected(
            normalization.outputs.standard_deviation,
            output_indices,
        )
        if previous_physical is None:
            previous_physical = inputs.new_full(
                (inputs.shape[0], 1, len(input_indices)),
                feedback.initial_value,
            )
        feedback_value = previous_physical
        predictions = []
        for timestep in range(inputs.shape[1]):
            step_inputs = inputs[:, timestep : timestep + 1].clone()
            step_inputs[:, :, input_indices] = (feedback_value - input_mean) / input_std
            output = self.model(step_inputs, state)
            if output.state is None:
                raise RuntimeError("GRU model did not return recurrent state.")
            predictions.append(output.predictions)
            state = output.state
            feedback_value = (
                output.predictions[:, :, output_indices] * output_std + output_mean
            )
        if state is None:
            raise RuntimeError("GRU input sequence must not be empty.")
        return torch.cat(predictions, dim=1), state, feedback_value


def _create_loss(config: LossConfig) -> nn.Module:
    """Create the configured regression loss function."""

    if config.type == "mse":
        return nn.MSELoss()
    if config.type == "mae":
        return nn.L1Loss()
    return nn.HuberLoss(delta=float(config.delta))


def _create_optimizer(
    config: OptimizerConfig,
    parameters: Iterable[nn.Parameter],
) -> torch.optim.Optimizer:
    """Create the optimizer selected by the training configuration."""

    return torch.optim.AdamW(
        parameters,
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )


def _set_seed(seed: int) -> None:
    """Seed model initialization, data shuffling, and augmentation."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
