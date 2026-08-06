"""Training-split normalization and physical-unit inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from anchor_bench.tasks.train.models import ModelOutput

if TYPE_CHECKING:
    from anchor_bench.tasks.train.task import MappedTrainingSequence

NormalizationMethod = Literal["none", "standard"]


@dataclass(frozen=True)
class AutoregressiveFeedback:
    """Describe one prediction feature fed into a later input timestep."""

    input_start: int
    input_stop: int
    output_start: int
    output_stop: int
    input_mean: tuple[float, ...]
    input_standard_deviation: tuple[float, ...]
    output_mean: tuple[float, ...]
    output_standard_deviation: tuple[float, ...]
    initial_value: float


@dataclass(frozen=True)
class FeatureNormalization:
    """Store one fitted transform for an ordered feature vector."""

    method: NormalizationMethod
    mean: tuple[float, ...]
    standard_deviation: tuple[float, ...]

    def normalize_array(self, values: np.ndarray) -> np.ndarray:
        """Normalize a feature array using the fitted training statistics."""

        return (values - np.asarray(self.mean)) / np.asarray(self.standard_deviation)

    def normalize_tensor(self, values: Tensor) -> Tensor:
        """Normalize a tensor without moving it away from its current device."""

        mean = values.new_tensor(self.mean)
        standard_deviation = values.new_tensor(self.standard_deviation)
        return (values - mean) / standard_deviation

    def denormalize_tensor(self, values: Tensor) -> Tensor:
        """Restore a normalized tensor to its original physical units."""

        mean = values.new_tensor(self.mean)
        standard_deviation = values.new_tensor(self.standard_deviation)
        return values * standard_deviation + mean


@dataclass(frozen=True)
class NormalizationStatistics:
    """Hold fitted input and output transforms for one trained model."""

    inputs: FeatureNormalization
    outputs: FeatureNormalization

    def wrap(
        self,
        model: nn.Module,
        feedback: AutoregressiveFeedback | None = None,
    ) -> nn.Module:
        """Expose a normalized model through a physical-unit inference interface."""

        return _PhysicalUnitModel(model, self, feedback)


@dataclass(frozen=True)
class _InferenceState:
    """Carry model state and the prior physical feedback prediction."""

    model: Any
    previous_prediction: Tensor


class _PhysicalUnitModel(nn.Module):
    """Normalize model inputs and denormalize its predictions during inference."""

    def __init__(
        self,
        model: nn.Module,
        statistics: NormalizationStatistics,
        feedback: AutoregressiveFeedback | None,
    ) -> None:
        """Retain the trained model, normalization, and optional feedback layout."""

        super().__init__()
        self.model = model
        self.statistics = statistics
        self.feedback = feedback

    def forward(self, inputs: Tensor, state: Any = None) -> ModelOutput[Any]:
        """Run inference while keeping the caller in physical units."""

        normalized_inputs = self.statistics.inputs.normalize_tensor(inputs)
        feedback = self.feedback
        if feedback is None:
            output = self.model(normalized_inputs, state)
            return ModelOutput(
                predictions=self.statistics.outputs.denormalize_tensor(
                    output.predictions
                ),
                state=output.state,
            )

        if state is not None and not isinstance(state, _InferenceState):
            raise TypeError("Feedback inference requires its returned model state.")
        model_state = None if state is None else state.model
        previous = (
            inputs.new_full(
                (inputs.shape[0], 1, feedback.output_stop - feedback.output_start),
                feedback.initial_value,
            )
            if state is None
            else state.previous_prediction
        )
        input_mean = inputs.new_tensor(feedback.input_mean)
        input_std = inputs.new_tensor(feedback.input_standard_deviation)
        physical_predictions = []
        for timestep in range(inputs.shape[1]):
            normalized_feedback = (previous - input_mean) / input_std
            step_inputs = torch.cat(
                (
                    normalized_inputs[
                        :, timestep : timestep + 1, : feedback.input_start
                    ],
                    normalized_feedback,
                    normalized_inputs[
                        :, timestep : timestep + 1, feedback.input_stop :
                    ],
                ),
                dim=-1,
            )
            output = self.model(step_inputs, model_state)
            model_state = output.state
            physical = self.statistics.outputs.denormalize_tensor(output.predictions)
            physical_predictions.append(physical)
            previous = physical[:, :, feedback.output_start : feedback.output_stop]
        return ModelOutput(
            predictions=torch.cat(physical_predictions, dim=1),
            state=_InferenceState(
                model=model_state,
                previous_prediction=previous,
            ),
        )


def fit_normalization(
    sequences: Sequence[MappedTrainingSequence],
    *,
    inputs: NormalizationMethod,
    outputs: NormalizationMethod,
) -> NormalizationStatistics:
    """Fit ordered input and output statistics from training sequences only."""

    return NormalizationStatistics(
        inputs=_fit_features(sequences, attribute="inputs", method=inputs),
        outputs=_fit_features(sequences, attribute="targets", method=outputs),
    )


def _fit_features(
    sequences: Sequence[MappedTrainingSequence],
    *,
    attribute: Literal["inputs", "targets"],
    method: NormalizationMethod,
) -> FeatureNormalization:
    """Fit one transform across matching feature positions in all sequences."""

    arrays = [
        getattr(sequence, attribute).to_numpy(dtype=np.float64, copy=False)
        for sequence in sequences
        if len(getattr(sequence, attribute)) > 0
    ]
    if not arrays:
        raise ValueError("Cannot fit normalization without training timesteps.")
    values = np.concatenate(arrays, axis=0)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Training {attribute} contain non-finite values.")

    if method == "none":
        mean = np.zeros(values.shape[1], dtype=np.float64)
        standard_deviation = np.ones(values.shape[1], dtype=np.float64)
    else:
        mean = values.mean(axis=0)
        standard_deviation = values.std(axis=0)
        standard_deviation[standard_deviation == 0.0] = 1.0

    return FeatureNormalization(
        method=method,
        mean=tuple(float(value) for value in mean),
        standard_deviation=tuple(float(value) for value in standard_deviation),
    )


__all__ = [
    "FeatureNormalization",
    "NormalizationStatistics",
    "fit_normalization",
]
