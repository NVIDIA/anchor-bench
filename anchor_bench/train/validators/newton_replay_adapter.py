# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Newton solver adapter for closed-loop replay validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import newton
import numpy as np
import warp as wp

from anchor_bench.core.simulation.newton_stepper import NewtonStepExecutor
from anchor_bench.train.utils import (
    position_error,
)
from anchor_bench.train.utils import (
    solver_pd as calculate_solver_pd,
)
from anchor_bench.train.utils.mujoco_bias import MuJoCoBiasEvaluator


@dataclass(frozen=True)
class _ReplayState:
    """Hold the current simulated scalar-joint state."""

    position: Mapping[str, float]
    velocity: Mapping[str, float]


class _NewtonReplayAdapter:
    """Own replay-specific Newton state, controls, and feature calculations."""

    def __init__(
        self,
        simulation_context: Any,
        *,
        output_joints: tuple[str, ...],
        residual: bool,
        substeps: int,
        disable_contacts: bool = False,
    ) -> None:
        model = simulation_context.model
        q_starts = model.joint_q_start.numpy()
        dof_starts = model.joint_qd_start.numpy()
        target_starts = model.joint_target_q_start.numpy()
        self._indices = {}
        for index, label in enumerate(model.joint_label):
            if not (
                int(q_starts[index + 1] - q_starts[index]) == 1
                and int(dof_starts[index + 1] - dof_starts[index]) == 1
                and int(target_starts[index + 1] - target_starts[index]) == 1
            ):
                continue
            joint = label
            if joint in self._indices:
                raise ValueError(f"Newton model has duplicate scalar joint {joint!r}.")
            self._indices[joint] = (
                int(q_starts[index]),
                int(dof_starts[index]),
                int(target_starts[index]),
            )
        missing = sorted(set(output_joints) - self._indices.keys())
        if missing:
            raise ValueError(
                "Newton model is missing scalar joints: " + ", ".join(missing)
            )
        solver_type = newton.solvers.SolverMuJoCo
        self._model = model
        state = simulation_context.state
        state_next = model.state()
        self._control = model.control()
        self.output_joints = output_joints
        self._substeps = substeps
        self._stiffness = np.asarray(model.joint_target_ke.numpy()).copy()
        self._damping = np.asarray(model.joint_target_kd.numpy()).copy()
        self._original_modes: np.ndarray | None = None
        if not residual:
            modes = np.asarray(model.joint_target_mode.numpy()).copy()
            self._original_modes = modes.copy()
            for joint in output_joints:
                modes[self._indices[joint][1]] = int(newton.JointTargetMode.EFFORT)
            model.joint_target_mode.assign(modes)
            stiffness = self._stiffness.copy()
            damping = self._damping.copy()
            for joint in output_joints:
                dof = self._indices[joint][1]
                stiffness[dof] = 0.0
                damping[dof] = 0.0
            model.joint_target_ke.assign(stiffness)
            model.joint_target_kd.assign(damping)
        self._solver = solver_type(model, disable_contacts=disable_contacts)
        self._step_executor = NewtonStepExecutor(
            model=model,
            solver=self._solver,
            state=state,
            state_next=state_next,
            control=self._control,
            contacts=None,
            clear_forces=False,
        )
        self._bias_evaluator: MuJoCoBiasEvaluator | None = None
        self._bias_state: _ReplayState | None = None
        self._bias_values: dict[str, float] = {}

    def reset(
        self,
        *,
        position: Mapping[str, float],
        velocity: Mapping[str, float],
        command: Mapping[str, float],
    ) -> _ReplayState:
        """Restore the exact recorded generalized state."""

        unknown = sorted(
            (set(position) | set(velocity) | set(command)) - self._indices.keys()
        )
        if unknown:
            raise ValueError(
                "Newton replay contains unknown joints: " + ", ".join(unknown)
            )
        q = self._joint_array(self._model.joint_q, position, 0)
        _, state_qd = self._state_joint_arrays(self._step_executor.state)
        recorded_qd = self._joint_array(
            state_qd,
            velocity,
            1,
            clear=True,
        )
        self._solver.reset(self._step_executor.state)
        for state in (
            self._step_executor.state,
            self._step_executor.state_next,
        ):
            state_q, state_qd = self._state_joint_arrays(state)
            state_q.assign(q)
            state_qd.assign(recorded_qd)
        self._write_control(command, {})
        return self._read_state()

    def capture_step_graph(self, *, dt: float) -> None:
        """Capture the configured replay interval when the solver supports it."""

        self._step_executor.capture_graph(dt=dt, substeps=self._substeps)

    def step(
        self,
        *,
        command: Mapping[str, float],
        effort: Mapping[str, float],
        dt: float,
    ) -> _ReplayState:
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("Newton replay timestep must be finite and positive.")
        self._write_control(command, effort)
        self._step_executor.step(dt=dt, substeps=self._substeps)
        return self._read_state()

    def solver_pd(self, joint: str, command: float, state: _ReplayState) -> float:
        dof = self._indices[joint][1]
        kp = float(self._stiffness[dof])
        kd = float(self._damping[dof])
        return float(
            calculate_solver_pd(
                position_error(command, state.position[joint]),
                state.velocity[joint],
                kp,
                kd,
            )
        )

    def dynamic_bias(self, joint: str, state: _ReplayState) -> float:
        """Evaluate MuJoCo qfrc_bias at the current replay state."""

        if self._bias_evaluator is None:
            self._initialize_dynamic_bias()
        if self._bias_state is not state:
            evaluator = self._bias_evaluator
            if evaluator is None:
                raise RuntimeError("Newton replay bias evaluator was not initialized.")
            self._bias_values = evaluator.evaluate_state(
                state.position,
                state.velocity,
            )
            self._bias_state = state
        return self._bias_values[joint]

    def _initialize_dynamic_bias(self) -> None:
        """Create an independent CPU MuJoCo view for bias evaluation."""

        self._bias_evaluator = MuJoCoBiasEvaluator(
            self._model,
            tuple(self._indices),
            output_entities=self.output_joints,
            solver_type=type(self._solver),
        )

    def _write_control(
        self,
        command: Mapping[str, float],
        effort: Mapping[str, float],
    ) -> None:
        targets = self._joint_array(self._control.joint_target_q, command, 2)
        forces = self._joint_array(self._control.joint_f, effort, 1, clear=True)
        self._control.joint_target_q.assign(targets)
        self._control.joint_target_qd.assign(np.zeros_like(forces))
        self._control.joint_f.assign(forces)

    def _joint_array(
        self,
        source: Any,
        values: Mapping[str, float],
        coordinate: int,
        *,
        clear: bool = False,
    ) -> np.ndarray:
        current = np.asarray(source.numpy(), dtype=np.float32)
        result = np.zeros_like(current) if clear else current.copy()
        for joint, value in values.items():
            result[self._indices[joint][coordinate]] = float(value)
        return result

    @staticmethod
    def _state_joint_arrays(state: newton.State) -> tuple[wp.array, wp.array]:
        """Return required replay joint buffers after narrowing optional arrays."""

        joint_q = state.joint_q
        joint_qd = state.joint_qd
        if joint_q is None or joint_qd is None:
            raise RuntimeError("Newton replay requires joint state buffers.")
        return joint_q, joint_qd

    def _read_state(self) -> _ReplayState:
        state = self._step_executor.state
        joint_q, joint_qd = self._state_joint_arrays(state)
        q = joint_q.numpy()
        qd = joint_qd.numpy()
        return _ReplayState(
            position={
                joint: float(q[index[0]]) for joint, index in self._indices.items()
            },
            velocity={
                joint: float(qd[index[1]]) for joint, index in self._indices.items()
            },
        )

    def close(self) -> None:
        if self._original_modes is not None:
            self._model.joint_target_mode.assign(self._original_modes)
            self._model.joint_target_ke.assign(self._stiffness)
            self._model.joint_target_kd.assign(self._damping)
