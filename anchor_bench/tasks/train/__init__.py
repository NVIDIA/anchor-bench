# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Training task interfaces."""

from anchor_bench.tasks.train.artifacts import save_training_artifacts
from anchor_bench.tasks.train.data import (
    SequenceDataset,
    TrainingDataLoaders,
    TrainingDataSplit,
    ValidationDataset,
    create_data_loaders,
    create_validation_dataset,
)
from anchor_bench.tasks.train.data_processor_runner import DataProcessorRunner
from anchor_bench.tasks.train.data_processors import BaseDataProcessor
from anchor_bench.tasks.train.model_registry import MODEL_REGISTRY, get_model
from anchor_bench.tasks.train.models import BaseModel, GruModel, ModelOutput
from anchor_bench.tasks.train.normalization import (
    FeatureNormalization,
    NormalizationStatistics,
    fit_normalization,
)
from anchor_bench.tasks.train.task import (
    EntityMapping,
    MappedTrainingSequence,
    SearchResult,
    SearchTrial,
    TrainingTask,
)
from anchor_bench.tasks.train.trainer import (
    EpochLoss,
    Trainer,
    TrainingResult,
)
from anchor_bench.tasks.train.utils import (
    friction_torque,
    position_error,
    solver_pd,
    torque_residual,
)
from anchor_bench.tasks.train.validator_runner import ValidatorRunner
from anchor_bench.tasks.train.validators import (
    BaseValidator,
    ValidationResult,
    ValidatorType,
)

__all__ = [
    "MODEL_REGISTRY",
    "BaseDataProcessor",
    "BaseModel",
    "BaseValidator",
    "DataProcessorRunner",
    "EntityMapping",
    "EpochLoss",
    "FeatureNormalization",
    "GruModel",
    "MappedTrainingSequence",
    "ModelOutput",
    "NormalizationStatistics",
    "SearchResult",
    "SearchTrial",
    "SequenceDataset",
    "Trainer",
    "TrainingDataLoaders",
    "TrainingDataSplit",
    "TrainingResult",
    "TrainingTask",
    "ValidationDataset",
    "ValidationResult",
    "ValidatorRunner",
    "ValidatorType",
    "create_data_loaders",
    "create_validation_dataset",
    "fit_normalization",
    "friction_torque",
    "get_model",
    "position_error",
    "save_training_artifacts",
    "solver_pd",
    "torque_residual",
]
