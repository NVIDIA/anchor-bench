# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare a USD robot's simulated motion with a measured trajectory."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Dict, Union

import newton
import numpy as np

from anchor_bench.core.simulation.newton_stepper import NewtonStepExecutor
from anchor_bench.schema.actuator_sys_id import ActuatorSysIdConfig
from anchor_bench.tasks.actuator_sys_id.dataset import load_trajectories
from anchor_bench.tasks.actuator_sys_id.model import Trajectory

MIN_JOINT_RANGE_RADIANS = 1.0e-3


def usd_trajectory_accuracy(
    usd_file: Union[str, Path],
    trajectory_file: Union[str, Path],
    config: ActuatorSysIdConfig,
) -> float:
    """Return the USD robot's measured-trajectory accuracy as a percentage.

    ``config`` supplies the simulation, dataset, and joint settings; its USD and
    dataset paths are replaced by ``usd_file`` and ``trajectory_file``. Accuracy
    is 100 minus mean per-joint normalized RMSE, clipped to 0–100. Near-static
    joints are excluded because their normalized error is unstable.
    """

    measured, simulated = _replay_trajectories(usd_file, trajectory_file, config)
    return _position_accuracy(measured, simulated)


def usd_trajectory_mse(
    usd_file: Union[str, Path],
    trajectory_file: Union[str, Path],
    config: ActuatorSysIdConfig,
) -> float:
    """Return the position MSE used by actuator system identification."""

    measured, simulated = _replay_trajectories(usd_file, trajectory_file, config)
    return float(np.mean(np.square(simulated - measured)))


def _replay_trajectories(
    usd_file: Union[str, Path],
    trajectory_file: Union[str, Path],
    config: ActuatorSysIdConfig,
) -> tuple[np.ndarray, np.ndarray]:
    usd_path = Path(usd_file).expanduser().resolve()
    trajectory_path = Path(trajectory_file).expanduser().resolve()
    evaluation_config = config.model_copy(
        update={
            "simulation": config.simulation.model_copy(
                update={"usd_path": str(usd_path)}
            ),
            "dataset": config.dataset.model_copy(
                update={"source": str(trajectory_path)}
            ),
        }
    )
    joint_names = [
        name for group in evaluation_config.joint_parameters for name in group.names
    ]
    trajectories = load_trajectories(
        evaluation_config.dataset,
        joint_names=joint_names,
    )
    runner = _NewtonTrajectoryRunner(evaluation_config)
    score_start = runner.score_start_index
    if any(
        score_start >= trajectory.measured_position.shape[0]
        for trajectory in trajectories
    ):
        raise ValueError("trajectory buffer leaves no samples for accuracy scoring")
    measured = np.concatenate(
        [trajectory.measured_position[score_start:] for trajectory in trajectories],
        axis=0,
    )
    simulated = np.concatenate(
        [runner.simulate(trajectory)[score_start:] for trajectory in trajectories],
        axis=0,
    )
    return measured, simulated


class _NewtonTrajectoryRunner:
    """Replay measured commands using actuator properties authored in one USD."""

    def __init__(self, config: ActuatorSysIdConfig) -> None:
        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.add_usd(
            source=str(Path(config.simulation.usd_path).resolve()),
            floating=False,
        )
        self._model = builder.finalize()
        self._collision_pipeline = newton.CollisionPipeline(
            self._model, broad_phase="explicit"
        )
        self._solver = newton.solvers.SolverMuJoCo(
            self._model,
            solver="newton",
            iterations=config.simulation.solver.solver_iterations,
            ls_iterations=config.simulation.solver.line_search_iterations,
            integrator=config.simulation.solver.integrator,
        )
        self._newton = newton
        self._contacts = self._collision_pipeline.contacts()
        self._control_dt = 1.0 / config.simulation.control_freq
        self._substeps = (
            config.simulation.physics_freq // config.simulation.control_freq
        )
        self._buffer_steps = round(config.buffer_time / self._control_dt)
        self._joint_indices = _joint_indices(self._model)
        self._state = self._model.state()
        self._state_next = self._model.state()
        self._control = self._model.control()
        self._step_executor = NewtonStepExecutor(
            model=self._model,
            solver=self._solver,
            state=self._state,
            state_next=self._state_next,
            control=self._control,
            contacts=self._contacts,
        )
        self._step_executor.capture_graph(
            dt=self._control_dt,
            substeps=self._substeps,
        )
        self._delay_steps = {
            _joint_key(name): round(
                (group.controller.delay_duration_ms / 1000.0) / self._control_dt
            )
            for group in config.joint_parameters
            for name in group.names
        }

    def simulate(self, trajectory: Trajectory) -> np.ndarray:
        """Return simulated joint positions aligned with measured samples."""

        selected = [
            self._joint_indices[_joint_key(name)] for name in trajectory.joint_names
        ]
        state = self._state
        state_next = self._state_next
        control = self._control
        model_joint_q = self._model.joint_q
        model_joint_qd = self._model.joint_qd
        model_joint_target_q = self._model.joint_target_q
        control_joint_target_q = control.joint_target_q
        if model_joint_q is None or model_joint_qd is None:
            raise RuntimeError("Newton model does not expose joint state arrays.")
        if model_joint_target_q is None or control_joint_target_q is None:
            raise RuntimeError("Newton model does not expose joint target arrays.")
        initial_q = np.asarray(model_joint_q.numpy()).copy()
        initial_qd = np.asarray(model_joint_qd.numpy()).copy()
        for column, (q_index, qd_index, _) in enumerate(selected):
            initial_q[q_index] = trajectory.measured_position[0, column]
            initial_qd[qd_index] = 0.0

        self._solver.reset(state)
        for state_buffer in (state, state_next):
            state_joint_q = state_buffer.joint_q
            state_joint_qd = state_buffer.joint_qd
            if state_joint_q is None or state_joint_qd is None:
                raise RuntimeError("Newton state does not expose joint state arrays.")
            state_joint_q.assign(initial_q)
            state_joint_qd.assign(initial_qd)
            self._newton.eval_fk(
                self._model,
                state_joint_q,
                state_joint_qd,
                state_buffer,
            )

        target = np.asarray(model_joint_target_q.numpy()).copy()
        simulated = np.empty_like(trajectory.measured_position)
        simulated[0] = trajectory.measured_position[0]
        delays = [
            self._delay_steps[_joint_key(name)] for name in trajectory.joint_names
        ]
        for measured_index in range(1, trajectory.commanded_position.shape[0]):
            command_index = measured_index - 1
            command = np.asarray(
                [
                    trajectory.commanded_position[max(0, command_index - delay), column]
                    for column, delay in enumerate(delays)
                ]
            )
            _set_targets(target, command, selected)
            control_joint_target_q.assign(target)
            self._step_executor.step(
                dt=self._control_dt,
                substeps=self._substeps,
            )
            state_joint_q = self._step_executor.state.joint_q
            if state_joint_q is None:
                raise RuntimeError("Newton state does not expose joint positions.")
            positions = np.asarray(state_joint_q.numpy())
            simulated[measured_index] = [
                positions[q_index] for q_index, _, _ in selected
            ]
        return simulated

    @property
    def score_start_index(self) -> int:
        """Return the first measured sample after the unscored trajectory prefix."""

        return self._buffer_steps + 1


def _position_accuracy(measured: np.ndarray, simulated: np.ndarray) -> float:
    if measured.shape != simulated.shape or measured.ndim != 2:
        raise ValueError(
            "Measured and simulated trajectories must have equal 2-D shape."
        )
    ranges = np.ptp(measured, axis=0)
    moving = ranges > MIN_JOINT_RANGE_RADIANS
    if not moving.any():
        raise ValueError("Accuracy requires at least one moving joint.")
    rmse = np.sqrt(np.mean(np.square(simulated - measured), axis=0))
    normalized_rmse = rmse[moving] / ranges[moving]
    return float(np.clip(100.0 * (1.0 - normalized_rmse.mean()), 0.0, 100.0))


def _joint_indices(model: Any) -> Dict[str, tuple[int, int, int]]:
    result: Dict[str, tuple[int, int, int]] = {}
    q_starts = np.asarray(model.joint_q_start.numpy())
    qd_starts = np.asarray(model.joint_qd_start.numpy())
    target_starts = np.asarray(model.joint_target_q_start.numpy())
    for index, label in enumerate(model.joint_label):
        if q_starts[index + 1] - q_starts[index] != 1:
            continue
        if qd_starts[index + 1] - qd_starts[index] != 1:
            continue
        if target_starts[index + 1] - target_starts[index] != 1:
            continue
        result[_joint_key(label.rsplit("/", 1)[-1])] = (
            int(q_starts[index]),
            int(qd_starts[index]),
            int(target_starts[index]),
        )
    return result


def _set_targets(
    target: np.ndarray,
    command: np.ndarray,
    selected: Sequence[tuple[int, int, int]],
) -> None:
    for value, (_, _, target_index) in zip(command, selected, strict=True):
        target[target_index] = value


def _joint_key(name: str) -> str:
    return "".join(character for character in name.casefold() if character.isalnum())
