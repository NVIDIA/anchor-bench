"""Newton solver adapter for closed-loop replay validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import newton
import numpy as np

from anchor_bench.core.simulation.newton import NewtonSimulationContext


@dataclass(frozen=True)
class _ReplayState:
    """Hold the current simulated scalar-joint state."""

    position: Mapping[str, float]
    velocity: Mapping[str, float]


class _NewtonStepper:
    """Advance closed-loop replay through the shared Newton context."""

    def __init__(
        self,
        simulation_context: NewtonSimulationContext,
        *,
        output_joints: tuple[str, ...],
        residual: bool,
        substeps: int,
    ) -> None:
        model = simulation_context.model
        joint_q_start = model.joint_q_start
        joint_qd_start = model.joint_qd_start
        joint_target_q_start = model.joint_target_q_start
        if (
            joint_q_start is None
            or joint_qd_start is None
            or joint_target_q_start is None
        ):
            raise RuntimeError("Newton model does not expose joint index arrays.")
        q_starts = joint_q_start.numpy()
        dof_starts = joint_qd_start.numpy()
        target_starts = joint_target_q_start.numpy()
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

        self._simulation_context = simulation_context
        self.output_joints = output_joints
        self._substeps = substeps
        self._original_modes: np.ndarray | None = None
        if not residual:
            joint_target_mode = model.joint_target_mode
            if joint_target_mode is None:
                raise RuntimeError("Newton model does not expose joint target modes.")
            modes = np.asarray(joint_target_mode.numpy()).copy()
            self._original_modes = modes.copy()
            for joint in output_joints:
                modes[self._indices[joint][1]] = int(newton.JointTargetMode.EFFORT)
            joint_target_mode.assign(modes)

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
        context = self._simulation_context
        q = self._joint_array(context.model.joint_q, position, 0)
        qd = self._joint_array(context.state_in.joint_qd, velocity, 1, clear=True)
        context.solver.reset(context.state_in)
        for state in (context.state_in, context.state_out):
            state_joint_q = state.joint_q
            state_joint_qd = state.joint_qd
            if state_joint_q is None or state_joint_qd is None:
                raise RuntimeError("Newton state does not expose joint state arrays.")
            state_joint_q.assign(q)
            state_joint_qd.assign(qd)
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
            self._simulation_context.step(dt=physics_dt)
        return self._read_state()

    def solver_pd(self, joint: str, command: float, state: _ReplayState) -> float:
        dof = self._indices[joint][1]
        model = self._simulation_context.model
        joint_target_ke = model.joint_target_ke
        joint_target_kd = model.joint_target_kd
        if joint_target_ke is None or joint_target_kd is None:
            raise RuntimeError("Newton model does not expose joint target gains.")
        kp = float(joint_target_ke.numpy()[dof])
        kd = float(joint_target_kd.numpy()[dof])
        return kp * (command - state.position[joint]) - kd * state.velocity[joint]

    def _write_control(
        self,
        command: Mapping[str, float],
        effort: Mapping[str, float],
    ) -> None:
        control = self._simulation_context.control
        joint_target_q = control.joint_target_q
        joint_target_qd = control.joint_target_qd
        joint_f = control.joint_f
        if joint_target_q is None or joint_target_qd is None or joint_f is None:
            raise RuntimeError("Newton control does not expose joint control arrays.")
        targets = self._joint_array(joint_target_q, command, 2)
        forces = self._joint_array(joint_f, effort, 1, clear=True)
        joint_target_q.assign(targets)
        joint_target_qd.assign(np.zeros_like(forces))
        joint_f.assign(forces)

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
        state = self._simulation_context.state_in
        joint_q = state.joint_q
        joint_qd = state.joint_qd
        if joint_q is None or joint_qd is None:
            raise RuntimeError("Newton state does not expose joint state arrays.")
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
            joint_target_mode = self._simulation_context.model.joint_target_mode
            if joint_target_mode is None:
                raise RuntimeError("Newton model does not expose joint target modes.")
            joint_target_mode.assign(self._original_modes)
