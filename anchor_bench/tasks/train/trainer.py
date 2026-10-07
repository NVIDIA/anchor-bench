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

_DEVICE_BATCH_CACHE_LIMIT_BYTES = 512 * 1024 * 1024


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


@dataclass(frozen=True)
class _FeedbackLayout:
    """Describe feedback columns and their tensor positions."""

    input_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    input_indices: tuple[int, ...]
    target_indices: tuple[int, ...]


@dataclass(frozen=True)
class _FeedbackNormalization:
    """Cache feedback normalization constants on the training device."""

    input_mean: Tensor
    input_standard_deviation: Tensor
    output_mean: Tensor
    output_standard_deviation: Tensor
    input_indices: Tensor
    output_indices: Tensor


@dataclass(frozen=True)
class _CapturedTrainingStep:
    """Replay one fixed-shape optimizer step through a CUDA graph."""

    graph: torch.cuda.CUDAGraph
    inputs: Tensor
    targets: Tensor
    loss: Tensor


@dataclass(frozen=True)
class _CapturedEvaluationStep:
    """Replay one fixed-shape validation step through a CUDA graph."""

    graph: torch.cuda.CUDAGraph
    inputs: Tensor
    targets: Tensor
    loss: Tensor


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
        enable_cuda_graphs: bool = True,
        cache_data_on_device: bool | None = None,
    ) -> None:
        """Create the model, loss function, optimizer, and data loaders.

        report_metrics controls per-epoch curves and console progress, which
        searches disable to avoid nested reporting. Checkpoint artifacts remain
        independently enabled. CUDA graphs accelerate autoregressive training
        by default. Small datasets are cached on the selected CUDA device unless
        cache_data_on_device is explicitly disabled.
        """

        _set_seed(config.training.seed)
        self.config = config
        self.checkpoint_directory = checkpoint_directory
        self.report_metrics = report_metrics
        self.checkpoint_collection = checkpoint_collection
        if config.training.checkpoint is not None and checkpoint_directory is None:
            raise ValueError("Checkpointing requires a checkpoint directory.")
        feedback = config.training.feedback
        if feedback is None:
            self.feedback_indices = None
        else:
            self.feedback_indices = _feedback_indices(
                dataset,
                input_feature=feedback.input,
                output_feature=feedback.output,
            )
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
        self._cuda_graphs_enabled = (
            enable_cuda_graphs
            and self.device.type == "cuda"
            and self.feedback_indices is not None
        )
        self.optimizer = _create_optimizer(
            config.training.optimizer,
            self.model.parameters(),
            capturable=self._cuda_graphs_enabled,
        )
        self.feedback_normalization = self._feedback_normalization()
        should_cache_data = (
            self.device.type == "cuda"
            if cache_data_on_device is None
            else cache_data_on_device
        )
        self._training_batches, self._validation_batches = self._prepare_batches(
            cache_on_device=should_cache_data,
        )
        self._training_graph_warmed = False
        self._training_graph_batch_shape: tuple[int, ...] | None = None
        self._training_graph: _CapturedTrainingStep | None = None
        self._evaluation_graph: _CapturedEvaluationStep | None = None

    def _prepare_batches(
        self,
        *,
        cache_on_device: bool,
    ) -> tuple[
        Iterable[tuple[Tensor, Tensor]],
        Iterable[tuple[Tensor, Tensor]],
    ]:
        """Cache bounded training and validation batches on the model device."""

        train = self.data_loaders.train
        validation = self.data_loaders.validation
        if not cache_on_device:
            return train, validation

        byte_count = _loader_nbytes(train) + _loader_nbytes(validation)
        byte_limit = _DEVICE_BATCH_CACHE_LIMIT_BYTES
        if self.device.type == "cuda":
            free_bytes, _ = torch.cuda.mem_get_info(self.device)
            byte_limit = min(byte_limit, free_bytes // 10)
        if byte_count > byte_limit:
            logger.info(
                "Keeping training batches on CPU bytes=%d device_cache_limit=%d",
                byte_count,
                byte_limit,
            )
            return train, validation

        training_batches = tuple(
            (inputs.to(self.device), targets.to(self.device))
            for inputs, targets in train
        )
        validation_batches = tuple(
            (inputs.to(self.device), targets.to(self.device))
            for inputs, targets in validation
        )
        logger.info(
            "Cached training batches device=%s bytes=%d training_batches=%d "
            "validation_batches=%d",
            self.device,
            byte_count,
            len(training_batches),
            len(validation_batches),
        )
        return training_batches, validation_batches

    def _feedback_normalization(self) -> _FeedbackNormalization | None:
        """Create device-local feedback normalization constants once."""

        indices = self.feedback_indices
        if indices is None:
            return None
        input_indices, output_indices = indices
        normalization = self.data_loaders.normalization

        def selected(
            values: tuple[float, ...], selected_indices: tuple[int, ...]
        ) -> Tensor:
            return torch.tensor(
                tuple(values[index] for index in selected_indices),
                device=self.device,
                dtype=torch.float32,
            )

        return _FeedbackNormalization(
            input_mean=selected(normalization.inputs.mean, input_indices),
            input_standard_deviation=selected(
                normalization.inputs.standard_deviation,
                input_indices,
            ),
            output_mean=selected(normalization.outputs.mean, output_indices),
            output_standard_deviation=selected(
                normalization.outputs.standard_deviation,
                output_indices,
            ),
            input_indices=torch.tensor(
                input_indices,
                device=self.device,
                dtype=torch.int64,
            ),
            output_indices=torch.tensor(
                output_indices,
                device=self.device,
                dtype=torch.int64,
            ),
        )

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
        for inputs, targets in self._training_batches:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            loss = self._train_batch(inputs, targets)

            batch_size = inputs.shape[0]
            loss_value = float(loss.detach())
            del loss
            total_loss += loss_value * batch_size
            sample_count += batch_size

        if sample_count == 0:
            raise RuntimeError("Training loader unexpectedly produced no batches.")
        return total_loss / sample_count

    def _train_batch(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """Run or capture one optimizer step for an input batch."""

        graph = self._training_graph
        if graph is not None and inputs.shape == graph.inputs.shape:
            graph.inputs.copy_(inputs)
            graph.targets.copy_(targets)
            graph.graph.replay()
            return graph.loss

        # Unlike Warp, PyTorch needs an eager step to initialize lazy optimizer
        # and cuDNN state before capture. This remains a real training update.
        if self._cuda_graphs_enabled and not self._training_graph_warmed:
            self._training_graph_warmed = True
            self._training_graph_batch_shape = tuple(inputs.shape)
            return self._eager_training_step(inputs, targets)

        if (
            self._cuda_graphs_enabled
            and self._training_graph is None
            and tuple(inputs.shape) == self._training_graph_batch_shape
        ):
            return self._capture_training_step(inputs, targets)

        return self._eager_training_step(inputs, targets)

    def _eager_training_step(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """Execute one optimizer step eagerly."""

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
        return loss

    def _capture_training_step(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """Capture and execute one fixed-shape autoregressive optimizer step."""

        static_inputs = inputs.clone()
        static_targets = targets.clone()
        with torch.cuda.device(self.device):
            graph = torch.cuda.CUDAGraph()
            self.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.graph(graph):
                noise = float(self.config.training.augmentation.input_noise_std)
                model_inputs = static_inputs
                if noise > 0.0:
                    model_inputs = model_inputs + torch.randn_like(model_inputs) * noise
                predictions = self._forward(model_inputs)
                loss = self.loss_function(predictions, static_targets)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    float(self.config.training.gradient.clip_norm),
                )
                self.optimizer.step()
        self._training_graph = _CapturedTrainingStep(
            graph=graph,
            inputs=static_inputs,
            targets=static_targets,
            loss=loss,
        )
        graph.replay()
        return loss

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
            batches: Iterable[tuple[Tensor, Tensor]] = data_loader
            if data_loader is self.data_loaders.validation:
                batches = self._validation_batches
            for inputs, targets in batches:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                loss = self._evaluate_batch(inputs, targets)
                batch_size = inputs.shape[0]
                total_loss += float(loss) * batch_size
                sample_count += batch_size

        if sample_count == 0:
            return None
        return total_loss / sample_count

    def _evaluate_batch(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """Run or capture one fixed-shape validation step."""

        graph = self._evaluation_graph
        if graph is not None and inputs.shape == graph.inputs.shape:
            graph.inputs.copy_(inputs)
            graph.targets.copy_(targets)
            graph.graph.replay()
            return graph.loss
        if not self._cuda_graphs_enabled or graph is not None:
            return self.loss_function(self._forward(inputs), targets)

        static_inputs = inputs.clone()
        static_targets = targets.clone()
        with torch.cuda.device(self.device):
            cuda_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(cuda_graph):
                loss = self.loss_function(self._forward(static_inputs), static_targets)
        self._evaluation_graph = _CapturedEvaluationStep(
            graph=cuda_graph,
            inputs=static_inputs,
            targets=static_targets,
            loss=loss,
        )
        cuda_graph.replay()
        return loss

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

        normalization = self.feedback_normalization
        if normalization is None:
            raise RuntimeError("Feedback normalization is missing.")
        if previous_physical is None:
            previous_physical = inputs.new_full(
                (inputs.shape[0], 1, normalization.input_indices.numel()),
                feedback.initial_value,
            )
        feedback_value = previous_physical
        predictions = []
        for timestep in range(inputs.shape[1]):
            step_inputs = inputs[:, timestep : timestep + 1].clone()
            step_inputs.index_copy_(
                2,
                normalization.input_indices,
                (feedback_value - normalization.input_mean)
                / normalization.input_standard_deviation,
            )
            output = self.model(step_inputs, state)
            if output.state is None:
                raise RuntimeError("GRU model did not return recurrent state.")
            predictions.append(output.predictions)
            state = output.state
            feedback_value = (
                output.predictions.index_select(2, normalization.output_indices)
                * normalization.output_standard_deviation
                + normalization.output_mean
            )
        if state is None:
            raise RuntimeError("GRU input sequence must not be empty.")
        return torch.cat(predictions, dim=1), state, feedback_value


def _feedback_indices(
    dataset: Sequence[MappedTrainingSequence],
    *,
    input_feature: str,
    output_feature: str,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Pair feedback columns by entity and validate every sequence layout."""

    if not dataset:
        raise ValueError("Feedback training requires at least one mapped sequence.")

    reference = dataset[0]
    reference_layout = _feedback_layout(
        reference,
        input_feature=input_feature,
        output_feature=output_feature,
    )
    for sequence in dataset[1:]:
        layout = _feedback_layout(
            sequence,
            input_feature=input_feature,
            output_feature=output_feature,
        )
        if layout != reference_layout:
            raise ValueError(
                f"Sequence {sequence.name!r} feedback layout differs from "
                f"reference sequence {reference.name!r}; expected "
                f"{reference_layout!r}, got {layout!r}."
            )
    return reference_layout.input_indices, reference_layout.target_indices


def _feedback_layout(
    sequence: MappedTrainingSequence,
    *,
    input_feature: str,
    output_feature: str,
) -> _FeedbackLayout:
    """Return ordered feedback columns and entity-paired tensor indices."""

    input_columns, inputs_by_entity = _feedback_columns(
        sequence.inputs.columns,
        feature=input_feature,
        role="input",
        sequence_name=sequence.name,
    )
    target_columns, targets_by_entity = _feedback_columns(
        sequence.targets.columns,
        feature=output_feature,
        role="target",
        sequence_name=sequence.name,
    )
    if not input_columns:
        raise ValueError(
            f"Sequence {sequence.name!r} has no feedback input columns ending "
            f"in '/{input_feature}'."
        )

    input_entities = set(inputs_by_entity)
    target_entities = set(targets_by_entity)
    if input_entities != target_entities:
        missing_targets = sorted(input_entities - target_entities)
        missing_inputs = sorted(target_entities - input_entities)
        raise ValueError(
            f"Sequence {sequence.name!r} feedback entities differ between inputs "
            f"and targets; missing targets for {missing_targets!r}, missing inputs "
            f"for {missing_inputs!r}."
        )

    input_indices = tuple(inputs_by_entity.values())
    target_indices = tuple(targets_by_entity[entity] for entity in inputs_by_entity)
    columns = (*sequence.inputs.columns, *sequence.targets.columns)
    if len(input_entities) == 1 and all(
        isinstance(column, str)
        and "/" in column
        and column.rsplit("/", 1)[0] in input_entities
        for column in columns
    ):
        # Per-joint mappings share tensor slots even when their joint names differ.
        input_columns = tuple(column.rsplit("/", 1)[1] for column in input_columns)
        target_columns = tuple(column.rsplit("/", 1)[1] for column in target_columns)
    return _FeedbackLayout(
        input_columns=input_columns,
        target_columns=target_columns,
        input_indices=input_indices,
        target_indices=target_indices,
    )


def _feedback_columns(
    columns: Iterable[object],
    *,
    feature: str,
    role: str,
    sequence_name: str,
) -> tuple[tuple[str, ...], dict[str, int]]:
    """Collect one feedback column index per entity in dataframe order."""

    suffix = f"/{feature}"
    selected: list[str] = []
    by_entity: dict[str, int] = {}
    for index, column in enumerate(columns):
        if not isinstance(column, str) or not column.endswith(suffix):
            continue
        entity = column[: -len(suffix)]
        if not entity:
            raise ValueError(
                f"Sequence {sequence_name!r} has a feedback {role} column "
                f"without an entity prefix: {column!r}."
            )
        if entity in by_entity:
            raise ValueError(
                f"Sequence {sequence_name!r} has duplicate feedback {role} "
                f"columns for entity {entity!r}."
            )
        selected.append(column)
        by_entity[entity] = index
    return tuple(selected), by_entity


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
    *,
    capturable: bool = False,
) -> torch.optim.Optimizer:
    """Create the optimizer selected by the training configuration."""

    return torch.optim.AdamW(
        parameters,
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
        capturable=capturable,
    )


def _loader_nbytes(data_loader: DataLoader[tuple[Tensor, Tensor]]) -> int:
    """Return the exact bytes exposed by a fixed-window sequence loader."""

    dataset = data_loader.dataset
    if not isinstance(dataset, SequenceDataset):
        raise TypeError("Training loader must contain a SequenceDataset.")
    return (
        len(dataset)
        * dataset.window_length
        * (dataset.input_size + dataset.output_size)
        * torch.tensor([], dtype=torch.float32).element_size()
    )


def _set_seed(seed: int) -> None:
    """Seed model initialization, data shuffling, and augmentation."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
