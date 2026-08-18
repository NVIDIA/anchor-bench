# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic supervised network trainer."""

from __future__ import annotations

import logging
import random
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from anchor_bench.logging_utils import should_log_progress
from anchor_bench.mlops import get_active_run
from anchor_bench.schema.training import LossConfig, OptimizerConfig, TrainingConfig
from anchor_bench.tasks.train.data import SequenceDataset, create_data_loaders
from anchor_bench.tasks.train.model_registry import get_model
from anchor_bench.tasks.train.normalization import NormalizationStatistics
from anchor_bench.tasks.train.task import MappedTrainingSequence

logger = logging.getLogger(__name__)


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


class Trainer:
    """Train a configured model on prepared sequence data."""

    def __init__(
        self,
        config: TrainingConfig,
        dataset: Sequence[MappedTrainingSequence],
        *,
        device: Optional[torch.device | str] = None,
        checkpoint_directory: Optional[Path] = None,
        report_metrics: bool = True,
        checkpoint_collection: str = "training-checkpoints",
    ) -> None:
        """Create the model, loss function, optimizer, and data loaders.

        report_metrics controls per-epoch curves and console progress, which
        searches disable to avoid nested reporting. Checkpoint artifacts remain
        independently enabled.
        """

        _set_seed(config.training.seed)
        self.config = config
        self.checkpoint_directory = checkpoint_directory
        self.report_metrics = report_metrics
        self.checkpoint_collection = checkpoint_collection
        if config.training.checkpoint is not None and checkpoint_directory is None:
            raise ValueError("Checkpointing requires a checkpoint directory.")
        self.data_loaders = create_data_loaders(dataset, config)
        train_dataset = self.data_loaders.train.dataset
        if not isinstance(train_dataset, SequenceDataset):
            raise TypeError("Training loader must contain a SequenceDataset.")
        if len(train_dataset) == 0:
            raise ValueError(
                "Training data does not contain a complete training sequence window."
            )

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

        log = logger.info if self.report_metrics else logger.debug
        log(
            "Starting training epochs=%d device=%s training_batches=%d "
            "validation_batches=%d",
            self.config.training.epochs,
            self.device,
            len(self.data_loaders.train),
            len(self.data_loaders.validation),
        )
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
            metrics = {"training/loss": training_loss}
            if validation_loss is not None:
                metrics["validation/loss"] = validation_loss
            mlops_run = get_active_run()
            # Epochs are one-based and provide the monotonic step shared by
            # training and validation loss.
            if self.report_metrics and mlops_run is not None:
                mlops_run.log_metrics(metrics, step=epoch)
            if self.report_metrics:
                self._log_epoch(epoch, training_loss, validation_loss)

            checkpoint = self.config.training.checkpoint
            if checkpoint is not None and epoch % checkpoint.every_n_epochs == 0:
                self._save_checkpoint(epoch, history)
        self.model.eval()
        return TrainingResult(
            model=self.model,
            history=history,
            normalization=self.data_loaders.normalization,
        )

    def _log_epoch(
        self,
        epoch: int,
        training_loss: float,
        validation_loss: Optional[float],
    ) -> None:
        """Report every epoch at DEBUG and rate-limited progress at INFO."""

        total = self.config.training.epochs
        arguments = (epoch, total, training_loss, validation_loss)
        log = logger.info if should_log_progress(epoch, total) else logger.debug
        log(
            "Training epoch=%d/%d training_loss=%g validation_loss=%s",
            *arguments,
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
        logger.info("Saved training checkpoint epoch=%d path=%s", epoch, path)
        mlops_run = get_active_run()
        # Publish only the atomically renamed file so a backend never observes a
        # partial checkpoint.
        if mlops_run is not None:
            mlops_run.log_artifact(
                path,
                collection=self.checkpoint_collection,
                artifact_type="model",
            )

    def _train_epoch(self) -> float:
        """Update model weights for every independent training window."""

        self.model.train()
        total_loss = 0.0
        sample_count = 0
        for inputs, targets in self.data_loaders.train:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            noise = float(self.config.training.augmentation.input_noise_std)
            if noise > 0.0:
                inputs = inputs + torch.randn_like(inputs) * noise

            self.optimizer.zero_grad(set_to_none=True)
            predictions = self._forward(inputs)
            loss = self.loss_function(predictions, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.model.parameters(),
                float(self.config.training.gradient.clip_norm),
            )
            self.optimizer.step()

            batch_size = inputs.shape[0]
            total_loss += float(loss.detach()) * batch_size
            sample_count += batch_size

        if sample_count == 0:
            raise RuntimeError("Training loader unexpectedly produced no batches.")
        return total_loss / sample_count

    def _validate_epoch(self) -> Optional[float]:
        """Measure validation loss without updating model weights."""

        self.model.eval()
        return self._evaluate(self.data_loaders.validation)

    def _evaluate(
        self,
        data_loader: DataLoader[tuple[Tensor, Tensor]],
    ) -> Optional[float]:
        """Return the sample-weighted loss for one data loader."""

        total_loss = 0.0
        sample_count = 0
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                loss = self.loss_function(self._forward(inputs), targets)
                batch_size = inputs.shape[0]
                total_loss += float(loss) * batch_size
                sample_count += batch_size

        if sample_count == 0:
            return None
        return total_loss / sample_count

    def _forward(self, inputs: Tensor) -> Tensor:
        """Run an independent input batch and return its predictions."""

        predictions, _, _ = self._forward_with_state(inputs)
        return predictions

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
