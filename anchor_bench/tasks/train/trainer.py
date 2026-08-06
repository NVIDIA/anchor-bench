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
        """Update model weights for every training batch."""

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
            output = self.model(inputs)
            loss = self.loss_function(output.predictions, targets)
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
                output = self.model(inputs)
                loss = self.loss_function(output.predictions, targets)
                batch_size = inputs.shape[0]
                total_loss += float(loss) * batch_size
                sample_count += batch_size

        if sample_count == 0:
            return None
        return total_loss / sample_count


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
