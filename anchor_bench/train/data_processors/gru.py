# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Feature data processors used by GRU actuator training."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from anchor_bench.train.data_processors.base import BaseDataProcessor
from anchor_bench.train.utils import (
    MuJoCoBiasEvaluator,
    friction_torque,
    position_error,
    solver_pd,
    torque_residual,
)


def _require_columns(dataset: pd.DataFrame, columns: List[str]) -> None:
    """Require source columns used by a generated feature."""

    missing = [column for column in columns if column not in dataset.columns]
    if missing:
        names = ", ".join(missing)
        raise ValueError(f"Dataset is missing data processor columns: {names}.")


def _feature_entities(dataset: pd.DataFrame, feature: str) -> List[str]:
    """Return entities that expose the requested feature column."""

    suffix = f"/{feature}"
    entities = [
        column[: -len(suffix)]
        for column in dataset.columns
        if isinstance(column, str) and column.endswith(suffix)
    ]
    if not entities:
        raise ValueError(f"Dataset does not contain any {feature} columns.")
    return entities


def _joint_parameters(
    model: Any,
    entities: List[str],
    field: str,
) -> Dict[str, float]:
    """Read one per-DOF model field for dataset entities."""

    labels = list(model.joint_label)
    dof_starts = model.joint_qd_start.numpy()
    model_values = getattr(model, field, None)
    if model_values is None:
        raise ValueError(f"USD model does not define {field}.")
    values = model_values.numpy()

    parameters = {}
    for entity in entities:
        joint_index = labels.index(entity)
        dof_start = int(dof_starts[joint_index])
        dof_end = int(dof_starts[joint_index + 1])
        if dof_end - dof_start != 1:
            raise ValueError(
                f"USD joint {entity!r} must have exactly one degree of freedom."
            )

        parameters[entity] = float(values[dof_start])
    return parameters


def _entity_columns(entities: List[str], features: List[str]) -> List[str]:
    """Build entity-qualified columns for the requested features."""

    return [f"{entity}/{feature}" for entity in entities for feature in features]


def _require_entity_columns(
    dataset: pd.DataFrame,
    entities: List[str],
    features: List[str],
) -> None:
    """Require entity-qualified source columns."""

    _require_columns(dataset, _entity_columns(entities, features))


class PositionErrorDataProcessor(BaseDataProcessor):
    """Generate commanded-minus-measured position for every commanded entity."""

    def process(self) -> pd.DataFrame:
        """Return one position-error column per commanded entity."""

        command_suffix = "/command_q"
        entities = [
            column[: -len(command_suffix)]
            for column in self.dataset.columns
            if isinstance(column, str) and column.endswith(command_suffix)
        ]
        if not entities:
            raise ValueError("Dataset does not contain any command_q columns.")

        _require_entity_columns(
            self.dataset,
            entities,
            ["command_q", "actual_q"],
        )

        return pd.DataFrame(
            {
                f"{entity}/position_error": position_error(
                    self.dataset[f"{entity}/command_q"],
                    self.dataset[f"{entity}/actual_q"],
                )
                for entity in entities
            },
            index=self.dataset.index,
        )


class SolverPDDataProcessor(BaseDataProcessor):
    """Generate solver PD torque from precomputed position-error columns.

    The dataset must first be processed by :class:`PositionErrorDataProcessor`.
    """

    def __init__(self, dataset: pd.DataFrame) -> None:
        """Load PD gains for dataset entities from the USD model."""

        super().__init__(dataset)
        self.entities = _feature_entities(dataset, "position_error")
        model = self.simulation_context.model
        self.kp = _joint_parameters(model, self.entities, "joint_target_ke")
        self.kd = _joint_parameters(model, self.entities, "joint_target_kd")

    def process(self) -> pd.DataFrame:
        """Return ``kp * position_error - kd * actual_dq`` per entity."""

        _require_entity_columns(
            self.dataset,
            self.entities,
            ["position_error", "actual_dq"],
        )

        return pd.DataFrame(
            {
                f"{entity}/solver_pd": solver_pd(
                    self.dataset[f"{entity}/position_error"],
                    self.dataset[f"{entity}/actual_dq"],
                    self.kp[entity],
                    self.kd[entity],
                )
                for entity in self.entities
            },
            index=self.dataset.index,
        )


class ResidualTorqueDataProcessor(BaseDataProcessor):
    """Generate residual torque from precomputed position-error columns.

    The dataset must first be processed by :class:`PositionErrorDataProcessor`.
    """

    def __init__(self, dataset: pd.DataFrame) -> None:
        """Load PD and friction parameters from the USD model."""

        super().__init__(dataset)
        self.entities = _feature_entities(dataset, "position_error")
        model = self.simulation_context.model
        self.kp = _joint_parameters(model, self.entities, "joint_target_ke")
        self.kd = _joint_parameters(model, self.entities, "joint_target_kd")
        self.dry_friction = _joint_parameters(
            model,
            self.entities,
            "joint_friction",
        )
        self.viscous_friction = _joint_parameters(
            model,
            self.entities,
            "joint_damping",
        )

    def process(self) -> pd.DataFrame:
        """Return measured effort minus PD, dry, and viscous torque."""

        _require_entity_columns(
            self.dataset,
            self.entities,
            ["position_error", "actual_dq", "actual_effort"],
        )

        residuals = {}
        for entity in self.entities:
            velocity = self.dataset[f"{entity}/actual_dq"]
            solver_effort = solver_pd(
                self.dataset[f"{entity}/position_error"],
                velocity,
                self.kp[entity],
                self.kd[entity],
            )
            friction_effort = friction_torque(
                velocity,
                self.dry_friction[entity],
                self.viscous_friction[entity],
            )
            residuals[f"{entity}/torque_residual"] = torque_residual(
                self.dataset[f"{entity}/actual_effort"],
                solver_effort,
                friction_effort,
            )

        return pd.DataFrame(residuals, index=self.dataset.index)


class PreviousTorqueDataProcessor(BaseDataProcessor):
    """Generate shifted measured torque for fitting feedback normalization."""

    def process(self) -> pd.DataFrame:
        """Shift effort within each experiment and initialize it at zero."""

        if "experiment" not in self.dataset.index.names:
            raise ValueError("Previous torque requires an experiment index.")
        entities = _feature_entities(self.dataset, "actual_effort")
        columns = _entity_columns(entities, ["actual_effort"])
        shifted = (
            self.dataset[columns]
            .groupby(
                level="experiment",
                sort=False,
            )
            .shift(fill_value=0.0)
        )
        shifted.columns = [f"{entity}/previous_torque" for entity in entities]
        return shifted


class DynamicBiasDataProcessor(BaseDataProcessor):
    """Generate MuJoCo ``qfrc_bias`` from recorded position and velocity."""

    def process(self) -> pd.DataFrame:
        """Evaluate bias at every recorded state without advancing simulation."""

        entities = _feature_entities(self.dataset, "actual_q")
        _require_entity_columns(self.dataset, entities, ["actual_q", "actual_dq"])
        model = self.simulation_context.model
        positions = self.dataset[_entity_columns(entities, ["actual_q"])].to_numpy(
            dtype=np.float64, copy=False
        )
        velocities = self.dataset[_entity_columns(entities, ["actual_dq"])].to_numpy(
            dtype=np.float64, copy=False
        )
        values = MuJoCoBiasEvaluator(model, entities).evaluate(positions, velocities)
        return pd.DataFrame(
            values,
            columns=pd.Index(
                [f"{entity}/qfrc_bias" for entity in entities],
            ),
            index=self.dataset.index,
        )
