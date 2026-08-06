"""Compare a USD robot's simulated motion with a measured trajectory."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Dict, Union

import newton
import numpy as np

from anchor_bench.schema.actuator_sys_id import ActuatorSysIdConfig
from anchor_bench.schema.parse import parse
from anchor_bench.tasks.actuator_sys_id.dataset import load_trajectories
from anchor_bench.tasks.actuator_sys_id.model import Trajectory

MIN_JOINT_RANGE_RADIANS = 1.0e-3


def usd_trajectory_accuracy(
    usd_file: Union[str, Path],
    trajectory_file: Union[str, Path],
) -> float:
    """Return the USD robot's measured-trajectory accuracy as a percentage.

    Simulation and dataset settings come from ``config.yaml`` beside the USD.
    Accuracy is 100 minus mean per-joint normalized RMSE, clipped to 0–100.
    Near-static joints are excluded because their normalized error is unstable.
    """

    usd_path = Path(usd_file).expanduser().resolve()
    trajectory_path = Path(trajectory_file).expanduser().resolve()
    config_path = usd_path.parent / "config.yaml"
    config = parse(config_path)
    if not isinstance(config, ActuatorSysIdConfig):
        raise TypeError(f"Expected an actuator_sys_id config at {config_path}.")

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
    measured = np.concatenate(
        [trajectory.measured_position for trajectory in trajectories],
        axis=0,
    )
    simulated = np.concatenate(
        [runner.simulate(trajectory) for trajectory in trajectories],
        axis=0,
    )
    return _position_accuracy(measured, simulated)


class _NewtonTrajectoryRunner:
    """Replay measured commands using actuator properties authored in one USD."""

    def __init__(self, config: ActuatorSysIdConfig) -> None:
        newton.use_coord_layout_targets = True
        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.add_usd(source=str(Path(config.simulation.usd_path).resolve()))
        self._model = builder.finalize()
        self._solver = newton.solvers.SolverMuJoCo(
            self._model,
            solver="newton",
            iterations=config.simulation.solver.solver_iterations,
            ls_iterations=config.simulation.solver.line_search_iterations,
            integrator=config.simulation.solver.integrator,
        )
        self._newton = newton
        self._contacts = self._model.contacts()
        self._physics_dt = 1.0 / config.simulation.physics_freq
        self._control_dt = 1.0 / config.simulation.control_freq
        self._substeps = (
            config.simulation.physics_freq // config.simulation.control_freq
        )
        self._buffer_steps = round(config.buffer_time / self._control_dt)
        self._joint_indices = _joint_indices(self._model)
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
        state_in = self._model.state()
        state_out = self._model.state()
        control = self._model.control()
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

        self._solver.reset(state_in)
        for state in (state_in, state_out):
            state_joint_q = state.joint_q
            state_joint_qd = state.joint_qd
            if state_joint_q is None or state_joint_qd is None:
                raise RuntimeError("Newton state does not expose joint state arrays.")
            state_joint_q.assign(initial_q)
            state_joint_qd.assign(initial_qd)
            self._newton.eval_fk(
                self._model,
                state_joint_q,
                state_joint_qd,
                state,
            )

        target = np.asarray(model_joint_target_q.numpy()).copy()
        first_command = trajectory.commanded_position[0]
        _set_targets(target, first_command, selected)
        control_joint_target_q.assign(target)
        for _ in range(self._buffer_steps):
            for _ in range(self._substeps):
                state_in, state_out = self._step(state_in, state_out, control)

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
            for _ in range(self._substeps):
                state_in, state_out = self._step(state_in, state_out, control)
            state_joint_q = state_in.joint_q
            if state_joint_q is None:
                raise RuntimeError("Newton state does not expose joint positions.")
            positions = np.asarray(state_joint_q.numpy())
            simulated[measured_index] = [
                positions[q_index] for q_index, _, _ in selected
            ]
        return simulated

    def _step(
        self,
        state_in: Any,
        state_out: Any,
        control: Any,
    ) -> tuple[Any, Any]:
        state_in.clear_forces()
        self._solver.step(
            state_in=state_in,
            state_out=state_out,
            control=control,
            contacts=self._contacts,
            dt=self._physics_dt,
        )
        return state_out, state_in


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
