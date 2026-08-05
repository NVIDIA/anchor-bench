"""Post-training trajectory replay with Newton."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from anchor_bench.train.validators.base import (
    BaseValidator,
    ValidationResult,
    ValidatorType,
)


@dataclass(frozen=True)
class NewtonReplayMapping:
    """Bind one model input sequence to its ordered output joints."""

    name: str
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


@dataclass(frozen=True)
class _ReplayState:
    """Hold the current simulated scalar-joint state."""

    position: Mapping[str, float]
    velocity: Mapping[str, float]


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

        self.signals = tuple(signals)
        self.substeps = substeps

    def validate(
        self,
        model: Any,
        validation_dataset: Any,
    ) -> ValidationResult:
        """Replay every experiment and return aggregate and per-experiment metrics."""

        sequences = self._require_sequences(validation_dataset)
        output_joints = self._output_joints(sequences)
        stepper = _NewtonStepper(
            self.simulation_context,
            output_joints=output_joints,
            residual=self._uses_residual_effort(sequences),
            substeps=self.substeps,
        )
        metrics: dict[str, float] = {}
        aggregate: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
            signal: [] for signal in self.signals
        }
        try:
            if callable(getattr(model, "eval", None)):
                model.eval()
            for sequence in sequences:
                simulated = self._replay(model, sequence, stepper)
                prefix = f"experiment/{sequence.experiment}"
                for signal in self.signals:
                    actual = (
                        sequence.position if signal == "position" else sequence.velocity
                    )
                    actual_values = actual.loc[:, output_joints].to_numpy(dtype=float)
                    simulated_values = simulated[signal]
                    metrics.update(
                        _metrics(
                            prefix,
                            signal,
                            output_joints,
                            actual_values,
                            simulated_values,
                        )
                    )
                    aggregate[signal].append((actual_values, simulated_values))
        finally:
            stepper.close()

        for signal, values in aggregate.items():
            actual = np.concatenate([item[0] for item in values], axis=0)
            simulated = np.concatenate([item[1] for item in values], axis=0)
            metrics.update(
                _metrics("aggregate", signal, output_joints, actual, simulated)
            )
        return ValidationResult(name=self.name, type=self.type, metrics=metrics)

    @staticmethod
    def _require_sequences(dataset: Any) -> tuple[NewtonReplaySequence, ...]:
        """Require a non-empty sequence of replay experiments."""

        if isinstance(dataset, (str, bytes)) or not isinstance(dataset, Sequence):
            raise TypeError(
                "NewtonReplayValidator requires NewtonReplaySequence values."
            )
        sequences = tuple(dataset)
        if not sequences:
            raise ValueError("Newton replay requires at least one validation sequence.")
        if any(not isinstance(item, NewtonReplaySequence) for item in sequences):
            raise TypeError(
                "NewtonReplayValidator requires NewtonReplaySequence values."
            )
        for sequence in sequences:
            NewtonReplayValidator._validate_sequence(sequence)
        return sequences

    @staticmethod
    def _validate_sequence(sequence: NewtonReplaySequence) -> None:
        """Check dataframe alignment and mapping integrity for one experiment."""

        if not sequence.mappings:
            raise ValueError("Newton replay sequences require at least one mapping.")
        frames = [sequence.command, sequence.position, sequence.velocity]
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
        if len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0):
            raise ValueError(
                "Newton replay timestamps must increase within an experiment."
            )
        if sequence.position.columns.tolist() != sequence.velocity.columns.tolist():
            raise ValueError("Newton replay position and velocity joints must match.")

        widths = set()
        owners: set[str] = set()
        for mapping in sequence.mappings:
            if mapping.inputs.empty or not mapping.inputs.index.equals(reference_index):
                raise ValueError(
                    "Newton replay model inputs must align with state data."
                )
            if not mapping.output_joints:
                raise ValueError("Newton replay mappings require output joints.")
            duplicate = owners.intersection(mapping.output_joints)
            repeated = {
                joint
                for joint in mapping.output_joints
                if mapping.output_joints.count(joint) > 1
            }
            duplicate |= repeated
            if duplicate:
                raise ValueError(
                    "Newton replay mappings overlap output joints: "
                    + ", ".join(sorted(duplicate))
                )
            owners.update(mapping.output_joints)
            widths.add((mapping.inputs.shape[1], len(mapping.output_joints)))
        if len(widths) != 1:
            raise ValueError(
                "Newton replay mappings must have equal input/output widths."
            )
        missing = sorted(
            (owners - set(sequence.command.columns))
            | (owners - set(sequence.position.columns))
        )
        if missing:
            raise ValueError(
                "Newton replay data is missing joints: " + ", ".join(missing)
            )

        for frame in frames + [mapping.inputs for mapping in sequence.mappings]:
            if not np.all(np.isfinite(frame.to_numpy(dtype=float))):
                raise ValueError("Newton replay data must contain only finite values.")

    @staticmethod
    def _output_joints(
        sequences: Sequence[NewtonReplaySequence],
    ) -> tuple[str, ...]:
        """Require every experiment to own the same ordered output joints."""

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
        """Resolve whether Newton should retain its USD PD and friction baseline."""

        targets = {sequence.target_feature for sequence in sequences}
        if len(targets) != 1:
            raise ValueError("Newton replay experiments must share one target feature.")
        target = targets.pop()
        if target not in {"actual_effort", "torque", "torque_residual"}:
            raise ValueError(
                f"Newton replay does not support target feature {target!r}."
            )
        return target == "torque_residual"

    def _replay(
        self,
        model: Any,
        sequence: NewtonReplaySequence,
        stepper: _NewtonStepper,
    ) -> dict[str, np.ndarray]:
        """Run one closed-loop experiment from its recorded initial state."""

        state = stepper.reset(
            position=sequence.position.iloc[0].to_dict(),
            velocity=sequence.velocity.iloc[0].to_dict(),
            command=sequence.command.iloc[0].to_dict(),
        )
        simulated = {
            "position": np.empty((len(sequence.position), len(stepper.output_joints))),
            "velocity": np.empty((len(sequence.position), len(stepper.output_joints))),
        }
        simulated["position"][0] = sequence.position.loc[
            sequence.position.index[0], list(stepper.output_joints)
        ]
        simulated["velocity"][0] = sequence.velocity.loc[
            sequence.velocity.index[0], list(stepper.output_joints)
        ]

        hidden = None
        timestamps = sequence.command.index.get_level_values("time_ns").to_numpy(
            dtype=np.int64
        )
        for row in range(len(sequence.command) - 1):
            command = {
                joint: float(value)
                for joint, value in sequence.command.iloc[row].items()
            }
            features = np.asarray(
                [
                    self._live_inputs(mapping.inputs.iloc[row], state, command, stepper)
                    for mapping in sequence.mappings
                ],
                dtype=np.float32,
            )[:, np.newaxis, :]
            predictions, hidden = self._predict(
                model,
                features,
                hidden,
            )
            effort = _route_effort(predictions, sequence.mappings)
            dt = float(timestamps[row + 1] - timestamps[row]) / 1_000_000_000.0
            state = stepper.step(command=command, effort=effort, dt=dt)
            simulated["position"][row + 1] = [
                state.position[joint] for joint in stepper.output_joints
            ]
            simulated["velocity"][row + 1] = [
                state.velocity[joint] for joint in stepper.output_joints
            ]
        return simulated

    @staticmethod
    def _live_inputs(
        recorded: pd.Series,
        state: _ReplayState,
        command: Mapping[str, float],
        stepper: _NewtonStepper,
    ) -> np.ndarray:
        """Replace recorded state-dependent inputs with live simulated values."""

        values = []
        for column in recorded.index:
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
            elif feature == "position_error":
                value = command[joint] - state.position[joint]
            elif feature == "solver_pd":
                value = stepper.solver_pd(joint, command[joint], state)
            else:
                raise ValueError(
                    f"Newton replay cannot rebuild input feature {feature!r}."
                )
            values.append(float(value))
        return np.asarray(values, dtype=np.float32)

    @staticmethod
    def _predict(
        model: Any,
        features: np.ndarray,
        hidden: Any,
    ) -> tuple[np.ndarray, Any]:
        """Run one timestep and return its prediction and recurrent state."""

        torch = import_module("torch")
        parameters = getattr(model, "parameters", lambda: iter(()))()
        first_parameter = next(iter(parameters), None)
        device = None if first_parameter is None else first_parameter.device
        tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
        with torch.inference_mode():
            result = model(tensor, hidden)
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError(
                "Newton replay models must return (predictions, next_hidden)."
            )
        output, next_hidden = result
        values = output[:, -1, :].detach().cpu().numpy()
        if values.ndim != 2 or not np.all(np.isfinite(values)):
            raise ValueError("Newton replay model returned invalid predictions.")
        return values, next_hidden


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
        """Create one solver over the shared simulation model."""

        newton = import_module("newton")
        model = simulation_context.model
        labels = list(model.joint_label)
        q_starts = model.joint_q_start.numpy()
        dof_starts = model.joint_qd_start.numpy()
        self._indices: dict[str, tuple[int, int]] = {}
        for index, joint in enumerate(labels):
            if (
                int(q_starts[index + 1] - q_starts[index]) != 1
                or int(dof_starts[index + 1] - dof_starts[index]) != 1
            ):
                continue
            self._indices[joint] = (int(q_starts[index]), int(dof_starts[index]))
        missing = sorted(set(output_joints) - set(self._indices))
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

    def reset(
        self,
        *,
        position: Mapping[str, float],
        velocity: Mapping[str, float],
        command: Mapping[str, float],
    ) -> _ReplayState:
        """Reset solver state to the first recorded sample."""

        unknown = sorted(
            (set(position) | set(velocity) | set(command)) - self._indices.keys()
        )
        if unknown:
            raise ValueError(
                "Newton replay contains unknown joints: " + ", ".join(unknown)
            )
        q = np.asarray(self._model.joint_q.numpy(), dtype=np.float32).copy()
        qd = np.zeros(self._model.joint_dof_count, dtype=np.float32)
        for joint, value in position.items():
            q[self._indices[joint][0]] = float(value)
        for joint, value in velocity.items():
            qd[self._indices[joint][1]] = float(value)
        self._solver.reset(self._state_0)
        self._state_0.joint_q.assign(q)
        self._state_0.joint_qd.assign(qd)
        self._state_1.joint_q.assign(q)
        self._state_1.joint_qd.assign(qd)
        self._write_control(command, {})
        return self._read_state()

    def step(
        self,
        *,
        command: Mapping[str, float],
        effort: Mapping[str, float],
        dt: float,
    ) -> _ReplayState:
        """Apply one held control and advance the configured physical substeps."""

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
        """Calculate the USD-authored PD effort for one live joint state."""

        dof = self._indices[joint][1]
        kp = float(self._model.joint_target_ke.numpy()[dof])
        kd = float(self._model.joint_target_kd.numpy()[dof])
        return kp * (command - state.position[joint]) - kd * state.velocity[joint]

    def _write_control(
        self,
        command: Mapping[str, float],
        effort: Mapping[str, float],
    ) -> None:
        """Write position targets and neural effort into Newton control arrays."""

        targets = np.asarray(
            self._control.joint_target_q.numpy(), dtype=np.float32
        ).copy()
        forces = np.zeros(self._model.joint_dof_count, dtype=np.float32)
        target_velocities = np.zeros(self._model.joint_dof_count, dtype=np.float32)
        for joint, value in command.items():
            targets[self._indices[joint][0]] = float(value)
        for joint, value in effort.items():
            forces[self._indices[joint][1]] = float(value)
        self._control.joint_target_q.assign(targets)
        self._control.joint_target_qd.assign(target_velocities)
        self._control.joint_f.assign(forces)

    def _read_state(self) -> _ReplayState:
        """Read every resolved scalar joint from the active Newton state."""

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
        """Restore model actuation mode after direct-effort replay."""

        if self._original_modes is not None:
            self._model.joint_target_mode.assign(self._original_modes)


def _route_effort(
    predictions: np.ndarray,
    mappings: Sequence[NewtonReplayMapping],
) -> dict[str, float]:
    """Route each model output row onto its mapping's ordered joints."""

    expected = (len(mappings), len(mappings[0].output_joints))
    if predictions.shape != expected:
        raise ValueError(
            f"Newton replay model output has shape {predictions.shape}, expected {expected}."
        )
    return {
        joint: float(predictions[row, column])
        for row, mapping in enumerate(mappings)
        for column, joint in enumerate(mapping.output_joints)
    }


def _metrics(
    prefix: str,
    signal: str,
    joints: Sequence[str],
    actual: np.ndarray,
    simulated: np.ndarray,
) -> dict[str, float]:
    """Calculate flattened aggregate and per-joint RMSE and MAE metrics."""

    if actual.shape != simulated.shape or actual.shape[1:] != (len(joints),):
        raise ValueError("Newton replay metric arrays must share [time, joint] shape.")
    delta = simulated - actual
    if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(simulated)):
        raise ValueError("Newton replay metric arrays must contain finite values.")
    metrics = {
        f"{prefix}/{signal}/rmse": float(np.sqrt(np.mean(delta**2))),
        f"{prefix}/{signal}/mae": float(np.mean(np.abs(delta))),
    }
    for column, joint in enumerate(joints):
        metrics[f"{prefix}/{signal}/{joint}/rmse"] = float(
            np.sqrt(np.mean(delta[:, column] ** 2))
        )
        metrics[f"{prefix}/{signal}/{joint}/mae"] = float(
            np.mean(np.abs(delta[:, column]))
        )
    return metrics


__all__ = [
    "NewtonReplayMapping",
    "NewtonReplaySequence",
    "NewtonReplayValidator",
]
