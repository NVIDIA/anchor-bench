# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native Newton population evaluator for actuator system identification."""

from __future__ import annotations

import logging
from math import isclose
from pathlib import Path
from typing import ClassVar, Dict, List, Sequence, Tuple, cast

import newton
import numpy as np
import warp as wp
from pydantic import JsonValue

from anchor_bench.core.simulation.newton import NewtonSimulationContext
from anchor_bench.schema.actuator_sys_id import ActuatorSysIdConfig
from anchor_bench.tasks.actuator_sys_id.model import Candidate, Trajectory
from anchor_bench.tasks.actuator_sys_id.newton_actuation import (
    ExternalImplicitPD,
    NativePD,
    NewtonActuation,
)
from anchor_bench.tasks.actuator_sys_id.objective import PopulationEvaluator
from anchor_bench.tasks.actuator_sys_id.parameter_space import ParameterSpace

logger = logging.getLogger(__name__)


# Warp kernels require array constructors in annotations; these are DSL
# declarations rather than Python typing expressions.
@wp.kernel
def _write_targets(
    commands: wp.array3d(dtype=wp.float32),  # pyright: ignore[reportInvalidTypeForm]
    last_step: wp.array(dtype=wp.int32),  # pyright: ignore[reportInvalidTypeForm]
    step: wp.array(dtype=wp.int32),  # pyright: ignore[reportInvalidTypeForm]
    population_size: int,
    joint_target_indices: wp.array2d(  # pyright: ignore[reportInvalidTypeForm]
        dtype=wp.int32
    ),
    joint_target_q: wp.array(dtype=wp.float32),  # pyright: ignore[reportInvalidTypeForm]
) -> None:
    """Write each world's current delayed command, holding a slot's last one."""

    world = wp.tid()
    slot = world // population_size  # pyright: ignore[reportOperatorIssue]
    sample = wp.min(step[0], last_step[slot])
    for joint in range(joint_target_indices.shape[1]):
        joint_target_q[joint_target_indices[world, joint]] = commands[
            slot, sample, joint
        ]


@wp.kernel
def _accumulate_position_error(
    joint_q: wp.array(dtype=wp.float32),  # pyright: ignore[reportInvalidTypeForm]
    joint_q_indices: wp.array2d(  # pyright: ignore[reportInvalidTypeForm]
        dtype=wp.int32
    ),
    measured_positions: wp.array3d(  # pyright: ignore[reportInvalidTypeForm]
        dtype=wp.float64
    ),
    score_end: wp.array(dtype=wp.int32),  # pyright: ignore[reportInvalidTypeForm]
    step: wp.array(dtype=wp.int32),  # pyright: ignore[reportInvalidTypeForm]
    buffer_steps: int,
    population_size: int,
    score_sums: wp.array(  # pyright: ignore[reportInvalidTypeForm]
        dtype=wp.float64
    ),
) -> None:
    """Accumulate the current step's squared position error for each world."""

    world = wp.tid()
    slot = world // population_size  # pyright: ignore[reportOperatorIssue]
    sample = step[0]
    if sample < buffer_steps or sample >= score_end[slot]:
        return
    error_sum = wp.float64(0.0)
    for joint in range(joint_q_indices.shape[1]):
        error = (
            wp.float64(joint_q[joint_q_indices[world, joint]])
            - (measured_positions[slot, sample, joint])
        )
        error_sum += error * error
    score_sums[world] += error_sum


@wp.kernel
def _advance_step(
    step: wp.array(dtype=wp.int32),  # pyright: ignore[reportInvalidTypeForm]
) -> None:
    step[0] = step[0] + 1


class NewtonPopulationEvaluator(PopulationEvaluator):
    """Evaluate every candidate against a batch of trajectories in parallel.

    Worlds are laid out slot-major: world ``slot * num_envs + candidate``
    replays the batch's ``slot``-th trajectory with that candidate's
    parameters. Delayed commands and measurements live on the device, so each
    control interval needs no host work beyond its (captured) launch.
    """

    _PARAMETERS: ClassVar[Tuple[str, ...]] = (
        "stiffness",
        "damping",
        "armature",
        "dry_friction",
        "viscous_friction",
    )
    _PASSIVE_ARRAYS: ClassVar[Dict[str, str]] = {
        "armature": "joint_armature",
        "dry_friction": "joint_friction",
        "viscous_friction": "joint_damping",
    }

    def __init__(
        self,
        config: ActuatorSysIdConfig,
        parameter_space: ParameterSpace,
        simulation_context: NewtonSimulationContext,
        *,
        trajectory_count: int = 1,
    ) -> None:
        if trajectory_count < 1:
            raise ValueError("trajectory_count must be positive")
        self.config = config
        self.parameter_space = parameter_space
        self.joint_names = _joint_names(config)
        self._group_by_joint = {
            joint: group for group in config.joint_parameters for joint in group.names
        }
        self.num_envs = config.num_envs
        self.trajectory_slots = min(
            config.trajectory_batch_size or trajectory_count, trajectory_count
        )
        self.world_count = self.num_envs * self.trajectory_slots
        self.physics_dt = 1.0 / config.simulation.physics_freq
        self.control_dt = 1.0 / config.simulation.control_freq
        dataset_dt = float(config.dataset.resampling.dt)
        if not isclose(dataset_dt, self.control_dt, rel_tol=1.0e-9, abs_tol=1.0e-12):
            raise ValueError(
                "dataset resampling dt must match the simulation control timestep: "
                f"{dataset_dt} != {self.control_dt}"
            )
        self.control_substeps = (
            config.simulation.physics_freq // config.simulation.control_freq
        )
        self.buffer_steps = round(config.buffer_time / self.control_dt)
        self.delay_steps = np.asarray(
            _joint_delay_steps(config, self.control_dt),
            dtype=np.int64,
        )
        self._initialize_simulation(simulation_context)
        self._initialize_device_buffers()
        self._capture_step_graph()

    def _initialize_simulation(self, context: NewtonSimulationContext) -> None:
        initial = self.parameter_space.decode(self.parameter_space.initial_normalized)
        self._simulation_context = context

        def configure_template(template: newton.ModelBuilder) -> None:
            template_indices = _builder_joint_indices(template, self.joint_names)
            self._configure_template(template, template_indices, initial)

        solver = self.config.simulation.solver
        context.initialize(
            world_count=self.world_count,
            configure_builder=configure_template,
            configuration_key=self._builder_configuration_key(initial),
            solver_type=solver.type,
            solver_settings=solver.settings,
        )
        model = context.model
        (
            self._joint_q_indices,
            self._joint_dof_indices,
            self._joint_target_indices,
        ) = _model_joint_indices(
            model,
            self.joint_names,
            self.world_count,
        )
        implementation = {
            joint: group.controller.implementation
            for joint, group in self._group_by_joint.items()
        }
        native = [
            column
            for column, joint in enumerate(self.joint_names)
            if implementation[joint] == "native"
        ]
        self._actuations: List[NewtonActuation] = []
        if native:
            self._actuations.append(
                NativePD(
                    model,
                    joint_names=[self.joint_names[column] for column in native],
                    dof_indices=self._joint_dof_indices[:, native],
                )
            )
        external = [
            column
            for column, joint in enumerate(self.joint_names)
            if implementation[joint] == "external_implicit"
        ]
        if external:
            names = [self.joint_names[column] for column in external]
            self._actuations.append(
                ExternalImplicitPD(
                    model,
                    joint_names=names,
                    q_indices=self._joint_q_indices[:, external],
                    dof_indices=self._joint_dof_indices[:, external],
                    target_indices=self._joint_target_indices[:, external],
                    effort_limits=[
                        float(self._group_by_joint[name].controller.clamps.effort_limit)
                        for name in names
                    ],
                )
            )
        actuations = self._actuations

        def before_substep(
            state: newton.State, control: newton.Control | None, dt: float
        ) -> None:
            # Actuators accumulate into joint_f, so zero it once per substep
            # before any of them runs. The stepper always supplies a control.
            joint_f = cast("wp.array", cast("newton.Control", control).joint_f)
            joint_f.zero_()
            for actuation in actuations:
                actuation.before_substep(state, control, dt)

        context.set_before_substep(before_substep)
        model_joint_q = model.joint_q
        model_joint_qd = model.joint_qd
        model_joint_target_q = model.joint_target_q
        if model_joint_q is None or model_joint_qd is None:
            raise RuntimeError("Newton model does not expose joint state arrays")
        if model_joint_target_q is None:
            raise RuntimeError("Newton model does not expose joint position targets")
        self._base_q = np.asarray(model_joint_q.numpy(), dtype=np.float32)
        self._base_qd = np.asarray(model_joint_qd.numpy(), dtype=np.float32)
        self._base_targets = np.asarray(
            model_joint_target_q.numpy(),
            dtype=np.float32,
        )
        self._base_model_arrays: Dict[str, np.ndarray] = {}
        for model_array_name in self._PASSIVE_ARRAYS.values():
            model_array = getattr(model, model_array_name)
            if model_array is None:
                raise RuntimeError(f"Newton model does not expose {model_array_name}")
            self._base_model_arrays[model_array_name] = np.array(
                model_array.numpy(),
                dtype=np.float32,
                copy=True,
            )

    def _initialize_device_buffers(self) -> None:
        device = self._simulation_context.model.device
        self._device = device
        self._joint_q_indices_device = wp.array(
            self._joint_q_indices, dtype=wp.int32, device=device
        )
        self._joint_target_indices_device = wp.array(
            self._joint_target_indices, dtype=wp.int32, device=device
        )
        joint_target_q = self._simulation_context.control.joint_target_q
        if joint_target_q is None:
            raise RuntimeError("Newton control does not expose joint position targets")
        self._joint_target_q = joint_target_q
        self._score_sums = wp.zeros(self.world_count, dtype=wp.float64, device=device)
        self._step = wp.zeros(1, dtype=wp.int32, device=device)
        self._last_step = wp.zeros(self.trajectory_slots, dtype=wp.int32, device=device)
        self._score_end = wp.zeros_like(self._last_step)
        self._allocate_trajectory_buffers(1)
        logger.debug(
            "Initialized device score accumulation device=%s worlds=%d "
            "trajectory_slots=%d joints=%d",
            device,
            self.world_count,
            self.trajectory_slots,
            len(self.joint_names),
        )

    def _allocate_trajectory_buffers(self, step_capacity: int) -> None:
        shape = (self.trajectory_slots, step_capacity, len(self.joint_names))
        self._commands = wp.zeros(shape, dtype=wp.float32, device=self._device)
        self._measured = wp.zeros(shape, dtype=wp.float64, device=self._device)
        # The interval hook binds these arrays, so re-registering it drops any
        # graph captured against the previous allocation.
        self._simulation_context.set_after_interval(self._after_interval)

    def _capture_step_graph(self) -> None:
        if all(actuation.is_graphable() for actuation in self._actuations):
            self._simulation_context.capture_step_graph(
                dt=self.physics_dt * self.control_substeps,
                substeps=self.control_substeps,
            )

    def _builder_configuration_key(self, initial: Candidate) -> tuple[object, ...]:
        joint_configuration = tuple(
            (
                joint,
                tuple(initial[joint][name] for name in self._PARAMETERS),
                float(self._group_by_joint[joint].controller.clamps.effort_limit),
                float(self._group_by_joint[joint].controller.clamps.velocity_limit),
                self._group_by_joint[joint].controller.implementation,
            )
            for joint in self.joint_names
        )
        return (self.config.type, joint_configuration)

    def _configure_template(
        self,
        template: newton.ModelBuilder,
        indices: Dict[str, Tuple[int, int]],
        initial: Candidate,
    ) -> None:
        position_mode = int(newton.JointTargetMode.POSITION)
        effort_mode = int(newton.JointTargetMode.EFFORT)
        for joint in self.joint_names:
            _, dof_index = indices[joint]
            group = self._group_by_joint[joint]
            values = initial[joint]
            if group.controller.implementation == "external_implicit":
                # Newton's external actuator supplies effort; disable the native
                # drive with zero gains (VBD ignores joint_target_mode).
                template.joint_target_mode[dof_index] = effort_mode
                template.joint_target_ke[dof_index] = 0.0
                template.joint_target_kd[dof_index] = 0.0
                template.joint_velocity_limit[dof_index] = float("inf")
            else:
                template.joint_target_mode[dof_index] = position_mode
                template.joint_target_ke[dof_index] = values["stiffness"]
                template.joint_target_kd[dof_index] = values["damping"]
                template.joint_velocity_limit[dof_index] = float(
                    group.controller.clamps.velocity_limit
                )
            template.joint_armature[dof_index] = values["armature"]
            template.joint_friction[dof_index] = values["dry_friction"]
            template.joint_damping[dof_index] = values["viscous_friction"]
            template.joint_effort_limit[dof_index] = float(
                group.controller.clamps.effort_limit
            )

    def evaluate(
        self,
        population: Sequence[Candidate],
        trajectories: Sequence[Trajectory],
    ) -> np.ndarray:
        """Return true position MSE for every candidate over all episodes."""

        if len(population) != self.num_envs:
            raise ValueError(
                f"expected {self.num_envs} candidates, got {len(population)}"
            )
        if not trajectories:
            raise ValueError("system identification requires trajectories")
        steps = [self._trajectory_steps(trajectory) for trajectory in trajectories]
        if max(steps) > self._commands.shape[1]:
            self._allocate_trajectory_buffers(max(steps))
            self._capture_step_graph()
        self._apply_population(population)
        self._score_sums.zero_()
        for start in range(0, len(trajectories), self.trajectory_slots):
            stop = start + self.trajectory_slots
            self._evaluate_batch(trajectories[start:stop], steps[start:stop])
        transition_count = sum(steps) - self.buffer_steps * len(trajectories)
        world_sums = np.asarray(self._score_sums.numpy(), dtype=np.float64)
        scores = world_sums.reshape(self.trajectory_slots, self.num_envs).sum(axis=0)
        scores /= transition_count * len(self.joint_names)
        if not np.isfinite(scores).all():
            raise RuntimeError("Newton produced non-finite position error")
        return scores

    def run_summary(self) -> Dict[str, JsonValue]:
        context = self._simulation_context
        return {
            "solver": {
                "type": context.solver_type,
                "settings": dict(context.solver_settings),
            }
        }

    def export_model(self, best: Candidate, output_dir: Path) -> Path:
        from anchor_bench.tasks.actuator_sys_id.usd import write_identified_usd_layer

        path = output_dir / "identified.usda"
        write_identified_usd_layer(
            self.config.simulation.usd_path,
            path,
            best,
            {
                joint_name: group.controller.clamps
                for group in self.config.joint_parameters
                for joint_name in group.names
            },
            self.config.model_dump_json(),
        )
        return path

    def _apply_population(self, population: Sequence[Candidate]) -> None:
        # Every trajectory slot replays the same candidates.
        world_population = list(population) * self.trajectory_slots
        for parameter_name, model_array_name in self._PASSIVE_ARRAYS.items():
            model_array = getattr(self._simulation_context.model, model_array_name)
            if model_array is None:
                raise RuntimeError(f"Newton model does not expose {model_array_name}")
            values = np.array(self._base_model_arrays[model_array_name], copy=True)
            for world_index, candidate in enumerate(world_population):
                for joint_index, joint in enumerate(self.joint_names):
                    values[self._joint_dof_indices[world_index, joint_index]] = (
                        candidate[joint][parameter_name]
                    )
            model_array.assign(values)
        for actuation in self._actuations:
            actuation.apply_population(world_population)

        flags = (
            newton.ModelFlags.JOINT_DOF_PROPERTIES
            | newton.ModelFlags.ACTUATOR_PROPERTIES
        )
        self._simulation_context.solver.notify_model_changed(flags)

    def _trajectory_steps(self, trajectory: Trajectory) -> int:
        """Validate a trajectory and return its control-step count."""

        if trajectory.joint_names != self.joint_names:
            raise ValueError("trajectory joints do not match the Newton evaluator")
        if trajectory.measured_position.shape[0] < 2:
            raise ValueError("trajectory must contain at least two samples")
        transition_count = trajectory.measured_position.shape[0] - 1
        if self.buffer_steps >= transition_count:
            raise ValueError(
                "trajectory must contain at least one scored transition after "
                f"the {self.buffer_steps}-step buffer"
            )
        return transition_count

    def _evaluate_batch(
        self, batch: Sequence[Trajectory], steps: Sequence[int]
    ) -> None:
        """Replay up to one trajectory per slot, accumulating world errors."""

        shape = self._commands.shape
        commands = np.zeros(shape, dtype=np.float32)
        measured = np.zeros(shape, dtype=np.float64)
        last_step = np.zeros(self.trajectory_slots, dtype=np.int32)
        score_end = np.zeros(self.trajectory_slots, dtype=np.int32)
        initial = np.empty((self.trajectory_slots, shape[2]), dtype=np.float32)
        for slot in range(self.trajectory_slots):
            # Unused slots in a final partial batch replay a real trajectory
            # but score nothing.
            index = slot if slot < len(batch) else 0
            trajectory = batch[index]
            count = steps[index]
            commands[slot, :count] = self._delayed_commands(trajectory)[:count]
            measured[slot, :count] = trajectory.measured_position[1:]
            last_step[slot] = count - 1
            score_end[slot] = count if slot < len(batch) else 0
            initial[slot] = trajectory.measured_position[0]
        self._commands.assign(commands)
        self._measured.assign(measured)
        self._last_step.assign(last_step)
        self._score_end.assign(score_end)
        self._reset(initial)
        for _ in range(max(steps)):
            self._step_control_interval()

    def _delayed_commands(self, trajectory: Trajectory) -> np.ndarray:
        commanded = np.asarray(trajectory.commanded_position, dtype=np.float32)
        sample_indices = np.arange(commanded.shape[0], dtype=np.int64)[:, np.newaxis]
        source_indices = np.maximum(
            0,
            sample_indices - self.delay_steps[np.newaxis, :],
        )
        return np.take_along_axis(commanded, source_indices, axis=0)

    def _after_interval(self, state: newton.State) -> None:
        """Score the finished step, then stage the next step's targets."""

        state_joint_q = state.joint_q
        if state_joint_q is None:
            raise RuntimeError("Newton state does not expose joint positions")
        wp.launch(
            _accumulate_position_error,
            dim=self.world_count,
            inputs=[
                state_joint_q,
                self._joint_q_indices_device,
                self._measured,
                self._score_end,
                self._step,
                self.buffer_steps,
                self.num_envs,
            ],
            outputs=[self._score_sums],
            device=self._device,
        )
        wp.launch(_advance_step, dim=1, inputs=[self._step], device=self._device)
        self._launch_write_targets()

    def _launch_write_targets(self) -> None:
        wp.launch(
            _write_targets,
            dim=self.world_count,
            inputs=[
                self._commands,
                self._last_step,
                self._step,
                self.num_envs,
                self._joint_target_indices_device,
            ],
            outputs=[self._joint_target_q],
            device=self._device,
        )

    def _reset(self, initial_positions: np.ndarray) -> None:
        """Reset every world to its slot's first measured pose at rest."""

        joint_q = np.array(self._base_q, copy=True)
        joint_qd = np.array(self._base_qd, copy=True)
        joint_q[self._joint_q_indices] = np.repeat(
            initial_positions, self.num_envs, axis=0
        )
        joint_qd[self._joint_dof_indices] = 0.0
        context = self._simulation_context
        context.solver.reset(context.state)
        for actuation in self._actuations:
            actuation.reset(context.control)
        for state in (context.state, context.state_next):
            state_joint_q = state.joint_q
            state_joint_qd = state.joint_qd
            if state_joint_q is None or state_joint_qd is None:
                raise RuntimeError("Newton state does not expose joint state arrays")
            state_joint_q.assign(joint_q)
            state_joint_qd.assign(joint_qd)
            newton.eval_fk(
                context.model,
                state_joint_q,
                state_joint_qd,
                state,
            )
        # Joints outside the fit keep their model targets; fitted joints are
        # overwritten from the device command buffer every step.
        self._joint_target_q.assign(self._base_targets)
        self._step.zero_()
        self._launch_write_targets()

    def _step_control_interval(self) -> None:
        self._simulation_context.step(
            dt=self.physics_dt * self.control_substeps,
            substeps=self.control_substeps,
        )


def _joint_names(config: ActuatorSysIdConfig) -> List[str]:
    return [joint for group in config.joint_parameters for joint in group.names]


def _joint_delay_steps(
    config: ActuatorSysIdConfig,
    control_dt: float,
) -> List[int]:
    return [
        round((group.controller.delay_duration_ms / 1000.0) / control_dt)
        for group in config.joint_parameters
        for _ in group.names
    ]


def _matching_label(labels: Sequence[str], requested: str) -> int:
    matches = [
        index
        for index, label in enumerate(labels)
        if label == requested or label.rsplit("/", 1)[-1] == requested
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Newton asset must contain exactly one joint matching {requested!r}"
        )
    return matches[0]


def _builder_joint_indices(
    builder: newton.ModelBuilder,
    joint_names: Sequence[str],
) -> Dict[str, Tuple[int, int]]:
    result: Dict[str, Tuple[int, int]] = {}
    for joint_name in joint_names:
        joint_index = _matching_label(builder.joint_label, joint_name)
        q_start = int(builder.joint_q_start[joint_index])
        dof_start = int(builder.joint_qd_start[joint_index])
        if joint_index + 1 < len(builder.joint_label):
            q_end = int(builder.joint_q_start[joint_index + 1])
            dof_end = int(builder.joint_qd_start[joint_index + 1])
        else:
            q_end = int(builder.joint_coord_count)
            dof_end = int(builder.joint_dof_count)
        if q_end - q_start != 1 or dof_end - dof_start != 1:
            raise ValueError(
                f"actuator_sys_id only supports scalar joints; {joint_name!r} "
                "does not have exactly one coordinate and DOF"
            )
        result[joint_name] = (q_start, dof_start)
    return result


def _model_joint_indices(
    model: newton.Model,
    joint_names: Sequence[str],
    num_envs: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    joint_world = model.joint_world
    joint_q_start = model.joint_q_start
    joint_qd_start = model.joint_qd_start
    joint_target_q_start = model.joint_target_q_start
    if (
        joint_world is None
        or joint_q_start is None
        or joint_qd_start is None
        or joint_target_q_start is None
    ):
        raise RuntimeError("Newton model does not expose joint index arrays")

    worlds = np.asarray(joint_world.numpy(), dtype=np.int64)
    q_starts = np.asarray(joint_q_start.numpy(), dtype=np.int64)
    dof_starts = np.asarray(joint_qd_start.numpy(), dtype=np.int64)
    target_starts = np.asarray(joint_target_q_start.numpy(), dtype=np.int64)
    q_indices = np.empty((num_envs, len(joint_names)), dtype=np.int64)
    dof_indices = np.empty_like(q_indices)
    target_indices = np.empty_like(q_indices)

    for world_index in range(num_envs):
        world_joint_indices = np.flatnonzero(worlds == world_index)
        world_labels = [model.joint_label[index] for index in world_joint_indices]
        for local_index, joint_name in enumerate(joint_names):
            label_index = _matching_label(world_labels, joint_name)
            joint_index = int(world_joint_indices[label_index])
            if q_starts[joint_index + 1] - q_starts[joint_index] != 1:
                raise ValueError(f"joint {joint_name!r} is not scalar")
            if dof_starts[joint_index + 1] - dof_starts[joint_index] != 1:
                raise ValueError(f"joint {joint_name!r} is not scalar")
            if target_starts[joint_index + 1] - target_starts[joint_index] != 1:
                raise ValueError(f"joint {joint_name!r} is not scalar")
            q_indices[world_index, local_index] = q_starts[joint_index]
            dof_indices[world_index, local_index] = dof_starts[joint_index]
            target_indices[world_index, local_index] = target_starts[joint_index]
    return q_indices, dof_indices, target_indices
