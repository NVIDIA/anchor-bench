"""Persistence for completed training runs."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Sequence

import torch

from anchor_bench.core.artifacts import write_yaml

if TYPE_CHECKING:
    from anchor_bench.schema.training import TrainingConfig
    from anchor_bench.tasks.train.data import TrainingDataSplit
    from anchor_bench.tasks.train.task import MappedTrainingSequence, SearchResult
    from anchor_bench.tasks.train.trainer import TrainingResult
    from anchor_bench.train.validators import ValidationResult


def save_training_artifacts(
    result: TrainingResult,
    sequences: Sequence[MappedTrainingSequence],
    validation_results: Sequence[ValidationResult],
    config: TrainingConfig,
    output_folder: Path,
    *,
    search_result: SearchResult | None = None,
    data_split: TrainingDataSplit | None = None,
    evaluation_split: Literal["validation", "test"] = "validation",
) -> Path:
    """Write one portable, config-addressed training artifact folder."""

    state_dict = {
        name: tensor.detach().cpu()
        for name, tensor in result.model.state_dict().items()
    }
    torch.save(state_dict, output_folder / "model.pt")
    write_yaml(
        output_folder / "history.yaml",
        [asdict(entry) for entry in result.history],
    )
    write_yaml(
        output_folder / "normalization.yaml",
        asdict(result.normalization),
    )
    write_yaml(
        output_folder / f"{evaluation_split}.yaml",
        [asdict(validation_result) for validation_result in validation_results],
    )
    if data_split is not None:
        write_yaml(
            output_folder / "splits.yaml",
            {
                "training": _experiment_names(data_split.training),
                "validation": _experiment_names(data_split.validation),
                "test": _experiment_names(data_split.test),
            },
        )
    write_yaml(
        output_folder / "metadata.yaml",
        {
            "format_version": 1,
            "model_type": config.model.type,
            "seed": config.training.seed,
            "input_features": [
                {
                    "name": feature,
                    "data_processor": data_processor,
                }
                for feature, data_processor in config.input_features.items()
            ],
            "output_features": [
                {
                    "name": feature,
                    "data_processor": data_processor,
                }
                for feature, data_processor in config.output_features.items()
            ],
            "simulation": config.simulation.model_dump(mode="json"),
            "mappings": _feature_layouts(sequences),
        },
    )
    if search_result is not None:
        write_yaml(
            output_folder / "search.yaml",
            {
                "objective": {
                    "metric": search_result.objective_metric,
                    "direction": search_result.objective_direction,
                },
                "best_trial": search_result.best_trial,
                "best_value": search_result.best_value,
                "best_parameters": dict(search_result.best_parameters),
                "best_config": search_result.best_config.model_dump(
                    mode="json", exclude_none=True
                ),
                "trials": [asdict(trial) for trial in search_result.trials],
            },
        )
    return output_folder


def _feature_layouts(
    sequences: Sequence[MappedTrainingSequence],
) -> dict[str, dict[str, list[str]]]:
    """Record each mapping's resolved input and target column order once."""

    layouts = {}
    for sequence in sequences:
        layouts.setdefault(
            sequence.name,
            {
                "inputs": sequence.inputs.columns.tolist(),
                "targets": sequence.targets.columns.tolist(),
            },
        )
    return layouts


def _experiment_names(
    sequences: Sequence[MappedTrainingSequence],
) -> list[str]:
    """Return experiment names once in their deterministic sequence order."""

    return list(dict.fromkeys(str(sequence.experiment) for sequence in sequences))
