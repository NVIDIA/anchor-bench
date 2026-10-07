# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Post-training trajectory replay with Newton."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

from anchor_bench.tasks.train.data import ValidationDataset
from anchor_bench.tasks.train.models import GruModel
from anchor_bench.tasks.train.newton_export import (
    BUILTIN_FEATURES,
    FEATURE_NAMES,
    export_newton_gru,
)
from anchor_bench.tasks.train.normalization import _PhysicalUnitModel
from anchor_bench.tasks.train.task import MappedTrainingSequence
from anchor_bench.tasks.train.utils import position_error
from anchor_bench.tasks.train.validators.base import (
    BaseValidator,
    ValidationResult,
    ValidatorType,
)
from anchor_bench.tasks.train.validators.newton_replay_adapter import (
    _NewtonReplayAdapter,
    _ReplayState,
)


@dataclass(frozen=True)
class NewtonReplayMapping:
    """Bind one model input sequence to its ordered output joints."""

    inputs: pd.DataFrame
    output_joints: tuple[str, ...]


@dataclass(frozen=True)
class NewtonReplaySequence:
    """Contain one aligned experiment used for closed-loop replay."""

    experiment: object
    mappings: tuple[NewtonReplayMapping, ...]
    command: pd.DataFrame
    position: pd.DataFrame
    velocity: pd.DataFrame
    target_feature: str


class NewtonReplayValidator(BaseValidator):
    """Replay trained effort predictions and compare simulated trajectories."""

    def __init__(
        self,
        *,
        name: str,
        type: ValidatorType,
        config: Mapping[str, Any],
    ) -> None:
        """Validate the small Newton replay configuration."""

        super().__init__(name=name, type=type, config=config)
        if type != "simulation":
            raise ValueError("NewtonReplayValidator must have type 'simulation'.")

        signals = self.config.get("signals", ["position", "velocity"])
        if (
            not isinstance(signals, list)
            or not signals
            or any(signal not in {"position", "velocity"} for signal in signals)
            or len(signals) != len(set(signals))
        ):
            raise ValueError(
                "Newton replay signals must be a non-empty unique list containing "
                "only 'position' and 'velocity'."
            )
        substeps = self.config.get("substeps", 1)
        if isinstance(substeps, bool) or not isinstance(substeps, int) or substeps < 1:
            raise ValueError("Newton replay substeps must be a positive integer.")
        disable_contacts = self.config.get("disable_contacts", False)
        if not isinstance(disable_contacts, bool):
            raise TypeError("Newton replay disable_contacts must be a boolean.")

        self.signals = tuple(signals)
        self.disable_contacts = disable_contacts
        self.substeps = substeps

    def validate(
        self,
        model: Any,
        validation_dataset: Any,
    ) -> ValidationResult:
        """Replay every experiment and return aggregate and per-experiment metrics."""

        sequences = self._prepare_sequences(validation_dataset)
        output_joints = self._output_joints(sequences)
        adapter = _NewtonReplayAdapter(
            self.simulation_context,
            output_joints=output_joints,
            residual=self._uses_residual_effort(sequences),
            disable_contacts=self.disable_contacts,
            substeps=self.substeps,
        )
        metrics: dict[str, float] = {}
        aggregate: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
            signal: [] for signal in self.signals
        }
        try:
            with self._configure_actuator(model, validation_dataset, adapter):
                return self._validate_sequences(
                    model, sequences, adapter, output_joints, metrics, aggregate
                )
        finally:
            adapter.close()

    def _validate_sequences(
        self,
        model: Any,
        sequences: Sequence[NewtonReplaySequence],
        adapter: _NewtonReplayAdapter,
        output_joints: tuple[str, ...],
        metrics: dict[str, float],
        aggregate: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    ) -> ValidationResult:
        if callable(getattr(model, "eval", None)):
            model.eval()
        for sequence in sequences:
            simulated = self._replay(model, sequence, adapter)
            prefix = f"experiment/{sequence.experiment}"
            recorded = {
                "position": sequence.position,
                "velocity": sequence.velocity,
            }
            for signal in self.signals:
                values = (
                    recorded[signal].loc[:, output_joints].to_numpy(dtype=float),
                    simulated[signal],
                )
                metrics.update(_metrics(prefix, signal, output_joints, *values))
                aggregate[signal].append(values)

        for signal, pairs in aggregate.items():
            actual, simulated = (
                np.concatenate(values, axis=0) for values in zip(*pairs)
            )
            metrics.update(
                _metrics("aggregate", signal, output_joints, actual, simulated)
            )
        return ValidationResult(name=self.name, type=self.type, metrics=metrics)

    @staticmethod
    @contextmanager
    def _configure_actuator(
        model: Any, dataset: ValidationDataset, adapter: _NewtonReplayAdapter
    ) -> Iterator[None]:
        """Use the same exporter for replay and persisted deployment artifacts."""
        if not isinstance(model, _PhysicalUnitModel) or not isinstance(
            model.model, GruModel
        ):
            raise TypeError(
                "Newton GRU replay requires a trained GRU with normalization statistics."
            )
        with TemporaryDirectory(prefix="anchor-gru-") as directory:
            path = export_newton_gru(
                model.model,
                model.statistics,
                dataset.sequences,
                Path(directory) / "model.onnx",
            )
            adapter.configure_gru(str(path))
            yield

    @staticmethod
    def _prepare_sequences(dataset: Any) -> tuple[NewtonReplaySequence, ...]:
        """Build private replay sequences from complete held-out processed data."""

        if not isinstance(dataset, ValidationDataset):
            raise TypeError("NewtonReplayValidator requires a ValidationDataset.")
        mapped_sequences = dataset.sequences
        if not mapped_sequences:
            raise ValueError("Newton replay requires at least one validation sequence.")
        if any(
            not isinstance(item, MappedTrainingSequence) for item in mapped_sequences
        ):
            raise TypeError(
                "NewtonReplayValidator requires mapped validation sequences."
            )

        grouped: dict[Any, list[MappedTrainingSequence]] = {}
        for sequence in mapped_sequences:
            grouped.setdefault(sequence.experiment, []).append(sequence)

        sequences = tuple(
            NewtonReplayValidator._build_sequence(
                experiment,
                mappings,
                NewtonReplayValidator._experiment_frame(
                    dataset.dataframe,
                    experiment,
                ),
            )
            for experiment, mappings in grouped.items()
        )
        for sequence in sequences:
            NewtonReplayValidator._validate_sequence(sequence)
        return sequences

    @staticmethod
    def _build_sequence(
        experiment: object,
        sequences: Sequence[MappedTrainingSequence],
        dataframe: pd.DataFrame,
    ) -> NewtonReplaySequence:
        """Derive one replay experiment from its mapped pandas sequences."""

        names = [sequence.name for sequence in sequences]
        if len(names) != len(set(names)):
            raise ValueError(
                f"Newton replay experiment {experiment!r} has duplicate mappings."
            )

        mappings = []
        target_features = set()
        for sequence in sequences:
            if not sequence.inputs.index.equals(sequence.targets.index):
                raise ValueError(
                    f"Newton replay mapping {sequence.name!r} inputs and targets "
                    "must share an index."
                )
            output_joints = []
            for column in sequence.targets.columns:
                joint, feature = NewtonReplayValidator._split_column(column)
                output_joints.append(joint)
                target_features.add(feature)
            mappings.append(
                NewtonReplayMapping(
                    inputs=sequence.inputs,
                    output_joints=tuple(output_joints),
                )
            )

        if len(target_features) != 1:
            raise ValueError(
                "Newton replay mapped targets must share one target feature."
            )
        output_joints = tuple(
            joint for mapping in mappings for joint in mapping.output_joints
        )
        return NewtonReplaySequence(
            experiment=experiment,
            mappings=tuple(mappings),
            command=NewtonReplayValidator._feature_frame(
                dataframe, "command_q", output_joints
            ),
            position=NewtonReplayValidator._feature_frame(
                dataframe, "actual_q", output_joints
            ),
            velocity=NewtonReplayValidator._feature_frame(
                dataframe, "actual_dq", output_joints
            ),
            target_feature=target_features.pop(),
        )

    @staticmethod
    def _experiment_frame(
        dataframe: pd.DataFrame,
        experiment: object,
    ) -> pd.DataFrame:
        """Select one experiment from the complete processed validation frame."""

        if "experiment" not in dataframe.index.names:
            raise ValueError("Newton replay data requires an experiment index.")
        try:
            return dataframe.xs(
                experiment,
                level="experiment",
                drop_level=False,
            )
        except KeyError as error:
            raise ValueError(
                f"Newton replay data is missing experiment {experiment!r}."
            ) from error

    @staticmethod
    def _feature_frame(
        dataframe: pd.DataFrame,
        feature: str,
        joints: Sequence[str],
    ) -> pd.DataFrame:
        """Select one named signal from complete processed validation data."""

        ordered_joints = tuple(dict.fromkeys(joints))
        columns = [f"{joint}/{feature}" for joint in ordered_joints]
        missing = [column for column in columns if column not in dataframe.columns]
        if missing:
            raise ValueError(
                f"Newton replay requires processed {feature!r} columns: "
                + ", ".join(missing)
            )
        frame = dataframe.loc[:, columns].copy()
        frame.columns = list(ordered_joints)
        return frame

    @staticmethod
    def _split_column(column: object) -> tuple[str, str]:
        """Split one mapped entity/feature column name."""

        if not isinstance(column, str) or "/" not in column:
            raise ValueError(
                "Newton replay mapped columns must use entity/feature names."
            )
        joint, feature = column.rsplit("/", 1)
        return joint, feature

    @staticmethod
    def _validate_sequence(sequence: NewtonReplaySequence) -> None:
        if not sequence.mappings:
            raise ValueError("Newton replay sequences require at least one mapping.")
        frames = (sequence.command, sequence.position, sequence.velocity)
        if any(frame.empty for frame in frames):
            raise ValueError("Newton replay state and command data must not be empty.")
        reference_index = sequence.command.index
        if any(not frame.index.equals(reference_index) for frame in frames[1:]):
            raise ValueError(
                "Newton replay state and command data must share an index."
            )
        if "time_ns" not in reference_index.names:
            raise ValueError("Newton replay data requires a time_ns index.")
        timestamps = reference_index.get_level_values("time_ns").to_numpy(
            dtype=np.int64
        )
        intervals = np.diff(timestamps)
        if np.any(intervals <= 0):
            raise ValueError(
                "Newton replay timestamps must increase within an experiment."
            )
        if len(intervals) > 1 and np.any(intervals != intervals[0]):
            raise ValueError("Newton replay timestamps must be uniformly spaced.")
        if sequence.position.columns.tolist() != sequence.velocity.columns.tolist():
            raise ValueError("Newton replay position and velocity joints must match.")

        input_frames = tuple(mapping.inputs for mapping in sequence.mappings)
        if any(
            frame.empty or not frame.index.equals(reference_index)
            for frame in input_frames
        ):
            raise ValueError("Newton replay model inputs must align with state data.")
        if any(not mapping.output_joints for mapping in sequence.mappings):
            raise ValueError("Newton replay mappings require output joints.")

        owners = tuple(
            joint for mapping in sequence.mappings for joint in mapping.output_joints
        )
        duplicates = {joint for joint in owners if owners.count(joint) > 1}
        if duplicates:
            raise ValueError(
                "Newton replay mappings overlap output joints: "
                + ", ".join(sorted(duplicates))
            )
        if (
            len(
                {
                    (mapping.inputs.shape[1], len(mapping.output_joints))
                    for mapping in sequence.mappings
                }
            )
            != 1
        ):
            raise ValueError(
                "Newton replay mappings must have equal input/output widths."
            )
        owner_set = set(owners)
        missing = sorted(
            (owner_set - set(sequence.command.columns))
            | (owner_set - set(sequence.position.columns))
        )
        if missing:
            raise ValueError(
                "Newton replay data is missing joints: " + ", ".join(missing)
            )
        if any(
            not np.all(np.isfinite(frame.to_numpy(dtype=float)))
            for frame in (*frames, *input_frames)
        ):
            raise ValueError("Newton replay data must contain only finite values.")

    @staticmethod
    def _output_joints(
        sequences: Sequence[NewtonReplaySequence],
    ) -> tuple[str, ...]:
        expected = tuple(
            joint
            for mapping in sequences[0].mappings
            for joint in mapping.output_joints
        )
        for sequence in sequences[1:]:
            current = tuple(
                joint
                for mapping in sequence.mappings
                for joint in mapping.output_joints
            )
            if current != expected:
                raise ValueError(
                    "Newton replay experiments must share ordered output joints."
                )
        return expected

    @staticmethod
    def _uses_residual_effort(sequences: Sequence[NewtonReplaySequence]) -> bool:
        targets = {sequence.target_feature for sequence in sequences}
        if len(targets) != 1:
            raise ValueError("Newton replay experiments must share one target feature.")
        target = next(iter(targets))
        if target not in {"actual_effort", "torque", "torque_residual"}:
            raise ValueError(
                f"Newton replay does not support target feature {target!r}."
            )
        return target == "torque_residual"

    def _replay(
        self,
        model: Any,
        sequence: NewtonReplaySequence,
        adapter: _NewtonReplayAdapter,
    ) -> dict[str, np.ndarray]:
        state = adapter.reset(
            position=sequence.position.iloc[0].to_dict(),
            velocity=sequence.velocity.iloc[0].to_dict(),
            command=sequence.command.iloc[0].to_dict(),
        )
        shape = (len(sequence.position), len(adapter.output_joints))
        simulated = {signal: np.empty(shape) for signal in ("position", "velocity")}
        output_columns = list(adapter.output_joints)
        for signal, frame in (
            ("position", sequence.position),
            ("velocity", sequence.velocity),
        ):
            simulated[signal][0] = frame.loc[frame.index[0], output_columns]

        previous_effort = {joint: 0.0 for joint in adapter.output_joints}
        timestamps = sequence.command.index.get_level_values("time_ns").to_numpy(
            dtype=np.int64
        )
        dt = (
            float(timestamps[1] - timestamps[0]) / 1_000_000_000.0
            if len(timestamps) > 1
            else 0.0
        )
        if len(timestamps) > 1:
            adapter.capture_step_graph(dt=dt)
        command_joints = tuple(sequence.command.columns)
        commands = sequence.command.to_numpy(dtype=float)
        input_columns = tuple(
            tuple(mapping.inputs.columns) for mapping in sequence.mappings
        )
        for row, command_values in enumerate(commands[:-1]):
            command = {
                joint: float(value)
                for joint, value in zip(command_joints, command_values)
            }
            features = np.asarray(
                [
                    self._live_inputs(columns, state, command, adapter, previous_effort)
                    for columns in input_columns
                ],
                dtype=np.float32,
            )[:, np.newaxis, :]
            effort = self._actuator_effort(
                adapter, features, command, input_columns, dt
            )
            previous_effort = effort
            state = adapter.step(command=command, effort=effort, dt=dt)
            for signal, values in (
                ("position", state.position),
                ("velocity", state.velocity),
            ):
                simulated[signal][row + 1] = [
                    values[joint] for joint in adapter.output_joints
                ]
        return simulated

    @staticmethod
    def _live_inputs(
        columns: Sequence[object],
        state: _ReplayState,
        command: Mapping[str, float],
        adapter: _NewtonReplayAdapter,
        previous_effort: Mapping[str, float],
    ) -> np.ndarray:
        values = []
        for column in columns:
            if not isinstance(column, str) or "/" not in column:
                raise ValueError(
                    "Newton replay model inputs must use entity/feature columns."
                )
            joint, feature = column.rsplit("/", 1)
            if joint not in state.position or joint not in command:
                raise ValueError(f"Newton replay cannot resolve input joint {joint!r}.")
            if feature == "actual_q":
                value = state.position[joint]
            elif feature == "actual_dq":
                value = state.velocity[joint]
            elif feature == "command_q":
                value = command[joint]
            elif feature == "command_dq":
                value = 0.0
            elif feature == "velocity_error":
                value = -state.velocity[joint]
            elif feature == "position_error":
                value = position_error(command[joint], state.position[joint])
            elif feature == "solver_pd":
                value = adapter.solver_pd(joint, command[joint], state)
            elif feature == "qfrc_bias":
                value = adapter.dynamic_bias(joint, state)
            elif feature == "previous_torque":
                value = previous_effort[joint]
            else:
                raise ValueError(
                    f"Newton replay cannot rebuild input feature {feature!r}."
                )
            values.append(float(value))
        return np.asarray(values, dtype=np.float32)

    @staticmethod
    def _actuator_effort(
        adapter: _NewtonReplayAdapter,
        features: np.ndarray,
        command: Mapping[str, float],
        input_columns: Sequence[Sequence[object]],
        dt: float,
    ) -> dict[str, float]:
        """Supply custom inputs while Newton computes built-in features from state."""
        custom_inputs: dict[str, dict[str, float]] = {}
        for row, columns in enumerate(input_columns):
            for column, label in enumerate(columns):
                joint, feature = str(label).rsplit("/", 1)
                name = FEATURE_NAMES.get(feature, feature)
                if name not in BUILTIN_FEATURES:
                    custom_inputs.setdefault(name, {})[joint] = float(
                        features[row, 0, column]
                    )
        return adapter.gru_effort(command=command, custom_inputs=custom_inputs, dt=dt)


def _metrics(
    prefix: str,
    signal: str,
    joints: Sequence[str],
    actual: np.ndarray,
    simulated: np.ndarray,
) -> dict[str, float]:
    if actual.shape != simulated.shape or actual.shape[1:] != (len(joints),):
        raise ValueError("Newton replay metric arrays must share [time, joint] shape.")
    delta = simulated - actual
    if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(simulated)):
        raise ValueError("Newton replay metric arrays must contain finite values.")
    scopes = [("", delta)] + [
        (f"/{joint}", delta[:, column]) for column, joint in enumerate(joints)
    ]
    return {
        f"{prefix}/{signal}{scope}/{metric}": score
        for scope, values in scopes
        for metric, score in (
            ("rmse", float(np.sqrt(np.mean(values**2)))),
            ("mae", float(np.mean(np.abs(values)))),
        )
    }


__all__ = [
    "NewtonReplayMapping",
    "NewtonReplaySequence",
    "NewtonReplayValidator",
]
