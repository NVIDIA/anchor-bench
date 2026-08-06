"""Newton solver adapter for closed-loop replay validation."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Mapping

import numpy as np

from anchor_bench.train.utils import (
    position_error,
)
from anchor_bench.train.utils import (
    solver_pd as calculate_solver_pd,
)


@dataclass(frozen=True)
class _ReplayState:
    """Hold the current simulated scalar-joint state."""

    position: Mapping[str, float]
    velocity: Mapping[str, float]


class _NewtonStepper:
    """Own Newton solver state and advance one replay at a time."""

    def __init__(
        self,
        simulation_context: Any,
        *,
        output_joints: tuple[str, ...],
        residual: bool,
        substeps: int,
    ) -> None:
        newton = import_module("newton")
        model = simulation_context.model
        q_starts = model.joint_q_start.numpy()
        dof_starts = model.joint_qd_start.numpy()
        target_starts = model.joint_target_q_start.numpy()
        self._indices = {
            joint: (
                int(q_starts[index]),
                int(dof_starts[index]),
                int(target_starts[index]),
            )
            for index, joint in enumerate(model.joint_label)
            if int(q_starts[index + 1] - q_starts[index]) == 1
            and int(dof_starts[index + 1] - dof_starts[index]) == 1
            and int(target_starts[index + 1] - target_starts[index]) == 1
        }
        missing = sorted(set(output_joints) - self._indices.keys())
        if missing:
            raise ValueError(
                "Newton model is missing scalar joints: " + ", ".join(missing)
            )

        solver_type = getattr(getattr(newton, "solvers", None), "SolverMuJoCo", None)
        if solver_type is None:
            raise RuntimeError("Newton replay requires newton.solvers.SolverMuJoCo.")
        self._model = model
        self._state_0 = simulation_context.state
        self._state_1 = model.state()
        self._control = model.control()
        self.output_joints = output_joints
        self._substeps = substeps
        self._original_modes: np.ndarray | None = None
        if not residual:
            modes = np.asarray(model.joint_target_mode.numpy()).copy()
            self._original_modes = modes.copy()
            for joint in output_joints:
                modes[self._indices[joint][1]] = int(newton.JointTargetMode.EFFORT)
            model.joint_target_mode.assign(modes)
        self._solver = solver_type(model)
        self._bias_solver: Any = None
        self._bias_state: _ReplayState | None = None
        self._bias_values: dict[str, float] = {}

    def reset(
        self,
        *,
        position: Mapping[str, float],
        velocity: Mapping[str, float],
        command: Mapping[str, float],
    ) -> _ReplayState:
        unknown = sorted(
            (set(position) | set(velocity) | set(command)) - self._indices.keys()
        )
        if unknown:
            raise ValueError(
                "Newton replay contains unknown joints: " + ", ".join(unknown)
            )
        q = self._joint_array(self._model.joint_q, position, 0)
        qd = self._joint_array(self._state_0.joint_qd, velocity, 1, clear=True)
        self._solver.reset(self._state_0)
        for state in (self._state_0, self._state_1):
            state.joint_q.assign(q)
            state.joint_qd.assign(qd)
        self._write_control(command, {})
        return self._read_state()

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
        physics_dt = dt / self._substeps
        for _ in range(self._substeps):
            self._solver.step(
                self._state_0,
                self._state_1,
                self._control,
                None,
                physics_dt,
            )
            self._state_0, self._state_1 = self._state_1, self._state_0
        return self._read_state()

    def solver_pd(self, joint: str, command: float, state: _ReplayState) -> float:
        dof = self._indices[joint][1]
        kp = float(self._model.joint_target_ke.numpy()[dof])
        kd = float(self._model.joint_target_kd.numpy()[dof])
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

        if self._bias_solver is None:
            self._initialize_dynamic_bias()
        if self._bias_state is not state:
            solver = self._bias_solver
            if solver is None:
                raise RuntimeError("Newton replay bias solver was not initialized.")
            solver.mj_data.qpos[:] = self._bias_default_qpos
            solver.mj_data.qvel[:] = 0.0
            for name, value in state.position.items():
                if name not in self._bias_indices:
                    continue
                q_address, _, dof_address = self._bias_indices[name]
                solver.mj_data.qpos[q_address] = (
                    float(value) + self._bias_dof_ref[dof_address]
                )
            for name, value in state.velocity.items():
                if name in self._bias_indices:
                    solver.mj_data.qvel[self._bias_indices[name][1]] = float(value)
            self._mujoco.mj_forward(solver.mj_model, solver.mj_data)
            self._bias_values = {
                name: float(solver.mj_data.qfrc_bias[indices[1]])
                for name, indices in self._bias_indices.items()
            }
            self._bias_state = state
        return self._bias_values[joint]

    def _initialize_dynamic_bias(self) -> None:
        """Create an independent CPU MuJoCo view for bias evaluation."""

        self._mujoco = import_module("mujoco")
        solver_type = type(self._solver)
        solver = solver_type(
            self._model,
            use_mujoco_cpu=True,
            disable_contacts=True,
        )
        missing = [
            name
            for name in ("mj_model", "mj_data", "mj_q_start", "mj_qd_start")
            if getattr(solver, name, None) is None
        ]
        if missing:
            raise RuntimeError(
                "Newton replay qfrc_bias requires MuJoCo model access; missing: "
                + ", ".join(missing)
                + "."
            )
        q_starts = np.asarray(solver.mj_q_start.numpy(), dtype=np.int64)
        qd_starts = np.asarray(solver.mj_qd_start.numpy(), dtype=np.int64)
        labels = list(self._model.joint_label)
        self._bias_indices = {
            name: (int(q_starts[index]), int(qd_starts[index]), indices[1])
            for name, indices in self._indices.items()
            for index in [labels.index(name)]
            if int(q_starts[index]) >= 0 and int(qd_starts[index]) >= 0
        }
        source = getattr(getattr(self._model, "mujoco", None), "dof_ref", None)
        self._bias_dof_ref = (
            np.zeros(int(self._model.joint_dof_count), dtype=np.float64)
            if source is None
            else np.asarray(source.numpy(), dtype=np.float64)
        )
        self._bias_default_qpos = np.array(solver.mj_data.qpos, copy=True)
        self._bias_solver = solver

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

    def _read_state(self) -> _ReplayState:
        q = self._state_0.joint_q.numpy()
        qd = self._state_0.joint_qd.numpy()
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
