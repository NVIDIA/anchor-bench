"""Feature preprocessors used by GRU actuator training."""

from __future__ import annotations

from typing import List, Mapping

import pandas as pd

from anchor_bench.train.preprocessors import BasePreprocessor


def _require_columns(dataset: pd.DataFrame, columns: List[str]) -> None:
    """Require source columns used by a generated feature."""

    missing = [column for column in columns if column not in dataset.columns]
    if missing:
        names = ", ".join(missing)
        raise ValueError(f"Dataset is missing preprocessor columns: {names}.")


def _parameter_entities(
    parameters: Mapping[str, Mapping[str, float]],
) -> List[str]:
    """Validate entity keys shared by related parameter mappings."""

    first_name, first_values = next(iter(parameters.items()))
    entities = list(first_values)
    if not entities:
        raise ValueError(f"{first_name} must contain at least one entity.")

    expected = set(entities)
    for name, values in parameters.items():
        if set(values) != expected:
            raise ValueError(f"{name} entities must match {first_name} entities.")
    return entities


def _entity_columns(entities: List[str], features: List[str]) -> List[str]:
    """Build entity-qualified columns for the requested features."""

    return [f"{entity}/{feature}" for entity in entities for feature in features]


def _solver_pd(
    dataset: pd.DataFrame,
    entity: str,
    kp: Mapping[str, float],
    kd: Mapping[str, float],
) -> pd.Series:
    """Calculate solver PD torque for one robot entity."""

    return (
        kp[entity] * dataset[f"{entity}/position_error"]
        - kd[entity] * dataset[f"{entity}/actual_dq"]
    )


class PositionErrorPreprocessor(BasePreprocessor):
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

        required = _entity_columns(entities, ["command_q", "actual_q"])
        _require_columns(self.dataset, required)

        return pd.DataFrame(
            {
                f"{entity}/position_error": (
                    self.dataset[f"{entity}/command_q"]
                    - self.dataset[f"{entity}/actual_q"]
                )
                for entity in entities
            },
            index=self.dataset.index,
        )


class SolverPDPreprocessor(BasePreprocessor):
    """Generate the solver PD torque for configured robot entities."""

    def __init__(
        self,
        dataset: pd.DataFrame,
        *,
        kp: Mapping[str, float],
        kd: Mapping[str, float],
    ) -> None:
        """Store entity-keyed proportional and derivative gains."""

        super().__init__(dataset)
        self.kp = dict(kp)
        self.kd = dict(kd)
        self.entities = _parameter_entities({"kp": self.kp, "kd": self.kd})

    def process(self) -> pd.DataFrame:
        """Return ``kp * position_error - kd * actual_dq`` per entity."""

        required = _entity_columns(
            self.entities,
            ["position_error", "actual_dq"],
        )
        _require_columns(self.dataset, required)

        return pd.DataFrame(
            {
                f"{entity}/solver_pd": _solver_pd(
                    self.dataset,
                    entity,
                    self.kp,
                    self.kd,
                )
                for entity in self.entities
            },
            index=self.dataset.index,
        )


class ResidualTorquePreprocessor(BasePreprocessor):
    """Generate measured torque after subtracting PD and friction baselines."""

    def __init__(
        self,
        dataset: pd.DataFrame,
        *,
        kp: Mapping[str, float],
        kd: Mapping[str, float],
        dry_friction: Mapping[str, float],
        viscous_friction: Mapping[str, float],
    ) -> None:
        """Store entity-keyed PD and friction parameters."""

        super().__init__(dataset)
        self.kp = dict(kp)
        self.kd = dict(kd)
        self.dry_friction = dict(dry_friction)
        self.viscous_friction = dict(viscous_friction)
        self.entities = _parameter_entities(
            {
                "kp": self.kp,
                "kd": self.kd,
                "dry_friction": self.dry_friction,
                "viscous_friction": self.viscous_friction,
            }
        )

    def process(self) -> pd.DataFrame:
        """Return measured effort minus PD, dry, and viscous torque."""

        required = _entity_columns(
            self.entities,
            ["position_error", "actual_dq", "actual_effort"],
        )
        _require_columns(self.dataset, required)

        residuals = {}
        for entity in self.entities:
            velocity = self.dataset[f"{entity}/actual_dq"]
            direction = (velocity > 0).astype(float) - (velocity < 0).astype(float)
            solver_pd = _solver_pd(
                self.dataset,
                entity,
                self.kp,
                self.kd,
            )
            friction = (
                self.dry_friction[entity] * direction
                + self.viscous_friction[entity] * velocity
            )
            residuals[f"{entity}/torque_residual"] = (
                self.dataset[f"{entity}/actual_effort"] - solver_pd - friction
            )

        return pd.DataFrame(residuals, index=self.dataset.index)
