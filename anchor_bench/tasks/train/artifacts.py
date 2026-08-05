"""Persistence for completed training runs."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import torch

from anchor_bench.core.artifacts import write_yaml

if TYPE_CHECKING:
    from anchor_bench.schema.training import TrainingConfig
    from anchor_bench.tasks.train.task import MappedTrainingSequence
    from anchor_bench.tasks.train.trainer import TrainingResult


def save_training_artifacts(
    result: TrainingResult,
    sequences: Sequence[MappedTrainingSequence],
    config: TrainingConfig,
    output_folder: Path,
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
        output_folder / "metadata.yaml",
        {
            "format_version": 1,
            "model_type": config.model.type,
            "seed": config.training.seed,
            "mappings": _feature_layouts(sequences),
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
