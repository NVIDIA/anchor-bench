# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared MuJoCo dynamic-bias evaluation for training and replay."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib import import_module
from typing import Any

import numpy as np


class MuJoCoBiasEvaluator:
    """Evaluate CPU MuJoCo ``qfrc_bias`` from complete named scalar states."""

    def __init__(
        self,
        model: Any,
        state_entities: Sequence[str],
        *,
        output_entities: Sequence[str] | None = None,
        solver_type: Any = None,
    ) -> None:
        """Create a CPU solver and resolve state and output joint addresses."""

        self.state_entities = tuple(state_entities)
        self.output_entities = (
            self.state_entities if output_entities is None else tuple(output_entities)
        )
        if not self.state_entities:
            raise ValueError("qfrc_bias requires at least one entity.")
        if len(self.state_entities) != len(set(self.state_entities)):
            raise ValueError("qfrc_bias state entities must be unique.")
        if not self.output_entities:
            raise ValueError("qfrc_bias requires at least one output entity.")
        if len(self.output_entities) != len(set(self.output_entities)):
            raise ValueError("qfrc_bias output entities must be unique.")
        missing_outputs = sorted(set(self.output_entities) - set(self.state_entities))
        if missing_outputs:
            raise ValueError(
                "qfrc_bias output entities are missing from the state: "
                + ", ".join(missing_outputs)
            )

        if solver_type is None:
            newton = import_module("newton")
            solver_type = getattr(
                getattr(newton, "solvers", None),
                "SolverMuJoCo",
                None,
            )
        if solver_type is None:
            raise RuntimeError("qfrc_bias requires newton.solvers.SolverMuJoCo.")

        self._mujoco = import_module("mujoco")
        self._solver = solver_type(
            model,
            use_mujoco_cpu=True,
            disable_contacts=True,
        )
        missing_solver_access = [
            name
            for name in ("mj_model", "mj_data", "mj_q_start", "mj_qd_start")
            if getattr(self._solver, name, None) is None
        ]
        if missing_solver_access:
            raise RuntimeError(
                "qfrc_bias requires SolverMuJoCo model access; missing: "
                + ", ".join(missing_solver_access)
                + "."
            )

        missing_model_access = [
            name
            for name in (
                "joint_label",
                "joint_q_start",
                "joint_qd_start",
                "joint_dof_count",
            )
            if getattr(model, name, None) is None
        ]
        if missing_model_access:
            raise RuntimeError(
                "qfrc_bias requires Newton model joint access; missing: "
                + ", ".join(missing_model_access)
                + "."
            )

        labels = list(model.joint_label)
        missing_entities = sorted(set(self.state_entities) - set(labels))
        if missing_entities:
            raise ValueError(
                "qfrc_bias model is missing joints: " + ", ".join(missing_entities)
            )
        joint_q_start = np.asarray(model.joint_q_start.numpy(), dtype=np.int64)
        joint_qd_start = np.asarray(model.joint_qd_start.numpy(), dtype=np.int64)
        if (
            len(joint_q_start) != len(labels) + 1
            or len(joint_qd_start) != len(labels) + 1
        ):
            raise RuntimeError("qfrc_bias requires complete Newton joint layouts.")
        joint_indices = np.asarray(
            [labels.index(entity) for entity in self.state_entities],
            dtype=np.int64,
        )
        non_scalar = [
            entity
            for entity, index in zip(
                self.state_entities,
                joint_indices,
                strict=True,
            )
            if joint_q_start[index + 1] - joint_q_start[index] != 1
            or joint_qd_start[index + 1] - joint_qd_start[index] != 1
        ]
        if non_scalar:
            raise ValueError(
                "qfrc_bias requires scalar joints: " + ", ".join(non_scalar)
            )
        mj_q_start = np.asarray(self._solver.mj_q_start.numpy(), dtype=np.int64)
        mj_qd_start = np.asarray(self._solver.mj_qd_start.numpy(), dtype=np.int64)
        self._q_addresses = mj_q_start[joint_indices]
        self._qd_addresses = mj_qd_start[joint_indices]
        self._dof_addresses = joint_qd_start[joint_indices]
        invalid = [
            entity
            for entity, q_address, qd_address in zip(
                self.state_entities,
                self._q_addresses,
                self._qd_addresses,
                strict=True,
            )
            if q_address < 0 or qd_address < 0
        ]
        if invalid:
            raise ValueError(
                "qfrc_bias entities must have MuJoCo coordinates: "
                + ", ".join(invalid)
                + "."
            )
        output_indices = np.asarray(
            [labels.index(entity) for entity in self.output_entities],
            dtype=np.int64,
        )
        self._output_qd_addresses = mj_qd_start[output_indices]

        dof_ref_source = getattr(getattr(model, "mujoco", None), "dof_ref", None)
        self._dof_ref = (
            np.zeros(int(model.joint_dof_count), dtype=np.float64)
            if dof_ref_source is None
            else np.asarray(dof_ref_source.numpy(), dtype=np.float64)
        )
        if np.any(self._dof_addresses < 0) or np.any(
            self._dof_addresses >= len(self._dof_ref)
        ):
            raise ValueError("qfrc_bias entities must have Newton DOF coordinates.")
        self._default_qpos = np.array(
            self._solver.mj_data.qpos,
            dtype=np.float64,
            copy=True,
        )

    def evaluate(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
    ) -> np.ndarray:
        """Evaluate samples serially with MuJoCo's CPU ``mj_forward`` API."""

        position_values = np.asarray(positions, dtype=np.float64)
        velocity_values = np.asarray(velocities, dtype=np.float64)
        expected_width = len(self.state_entities)
        if (
            position_values.ndim != 2
            or velocity_values.shape != position_values.shape
            or position_values.shape[1] != expected_width
        ):
            raise ValueError(
                "qfrc_bias positions and velocities must have matching "
                f"samples-by-{expected_width} shapes."
            )

        values = np.empty(
            (len(position_values), len(self.output_entities)),
            dtype=np.float64,
        )
        for row in range(len(position_values)):
            self._solver.mj_data.qpos[:] = self._default_qpos
            self._solver.mj_data.qvel[:] = 0.0
            self._solver.mj_data.qpos[self._q_addresses] = (
                position_values[row] + self._dof_ref[self._dof_addresses]
            )
            self._solver.mj_data.qvel[self._qd_addresses] = velocity_values[row]
            self._mujoco.mj_forward(self._solver.mj_model, self._solver.mj_data)
            values[row] = self._solver.mj_data.qfrc_bias[self._output_qd_addresses]

        if not np.all(np.isfinite(values)):
            raise ValueError("SolverMuJoCo produced non-finite qfrc_bias values.")
        return values

    def evaluate_state(
        self,
        position: Mapping[str, float],
        velocity: Mapping[str, float],
    ) -> dict[str, float]:
        """Evaluate one named state and return one bias value per entity."""

        missing = sorted(
            (set(self.state_entities) - position.keys())
            | (set(self.state_entities) - velocity.keys())
        )
        if missing:
            raise ValueError(
                "qfrc_bias state is missing entities: " + ", ".join(missing)
            )
        values = self.evaluate(
            np.asarray([[position[entity] for entity in self.state_entities]]),
            np.asarray([[velocity[entity] for entity in self.state_entities]]),
        )[0]
        return {
            entity: float(value)
            for entity, value in zip(self.output_entities, values, strict=True)
        }
