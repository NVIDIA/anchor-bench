# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native Newton population evaluator for actuator system identification."""

from __future__ import annotations

import logging
from math import isclose
from pathlib import Path
from typing import ClassVar, Dict, List, Sequence, Tuple

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


@wp.kernel
def _accumulate_position_error(
    # Warp kernels require array constructors in annotations; these are DSL
    # declarations rather than Python typing expressions.
    joint_q: wp.array(dtype=wp.float32),  # pyright: ignore[reportInvalidTypeForm]
    joint_q_indices: wp.array(  # pyright: ignore[reportInvalidTypeForm]
        dtype=wp.int32
    ),
    measured_positions: wp.array(  # pyright: ignore[reportInvalidTypeForm]
        dtype=wp.float64
    ),
    sample_index: int,
    joint_count: int,
    score_sums: wp.array(  # pyright: ignore[reportInvalidTypeForm]
        dtype=wp.float64
    ),
) -> None:
    """Accumulate one sample's squared position error for each world."""

    world_index = wp.tid()
    error_sum = wp.float64(0.0)
    for joint_index in range(joint_count):
        simulated_index = joint_q_indices[
            world_index * joint_count  # pyright: ignore[reportOperatorIssue]
            + joint_index
        ]
        measured_index = sample_index * joint_count + joint_index
        error = wp.float64(joint_q[simulated_index]) - wp.float64(
            measured_positions[measured_index]
        )
        error_sum += error * error
    score_sums[world_index] += error_sum


class NewtonPopulationEvaluator(PopulationEvaluator):
    """Evaluate one candidate per replicated Newton world."""

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
    ) -> None:
        self.config = config
        self.parameter_space = parameter_space
        self.joint_names = _joint_names(config)
        self.num_envs = config.num_envs
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
        self._initialize_score_buffers()
        if all(actuation.is_graphable() for actuation in self._actuations):
            self._simulation_context.capture_step_graph(
                dt=self.physics_dt * self.control_substeps,
                substeps=self.control_substeps,
            )

    def _initialize_simulation(self, context: NewtonSimulationContext) -> None:
        initial = self.parameter_space.decode(self.parameter_space.initial_normalized)
        self._simulation_context = context

        def configure_template(template: newton.ModelBuilder) -> None:
            template_indices = _builder_joint_indices(template, self.joint_names)
            self._configure_template(template, template_indices, initial)

        solver = self.config.simulation.solver
        context.initialize(
            world_count=self.num_envs,
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
            self.num_envs,
        )
        implementation = {
            joint: group.controller.implementation
            for group in self.config.joint_parameters
            for joint in group.names
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
            groups = {
                joint: group
                for group in self.config.joint_parameters
                for joint in group.names
            }
            names = [self.joint_names[column] for column in external]
            self._actuations.append(
                ExternalImplicitPD(
                    model,
                    joint_names=names,
                    q_indices=self._joint_q_indices[:, external],
                    dof_indices=self._joint_dof_indices[:, external],
                    target_indices=self._joint_target_indices[:, external],
                    effort_limits=[
                        float(groups[name].controller.clamps.effort_limit)
                        for name in names
                    ],
                )
            )
        hooks = [
            actuation for actuation in self._actuations if actuation.runs_before_substep
        ]
        if hooks:

            def before_substep(
                state: newton.State, control: newton.Control | None, dt: float
            ) -> None:
                for actuation in hooks:
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

    def _initialize_score_buffers(self) -> None:
        device = self._simulation_context.model.device
        self._device = device
        self._joint_q_indices_device = wp.array(
            self._joint_q_indices.reshape(-1),
            dtype=wp.int32,
            device=device,
        )
        self._score_sums = wp.zeros(
            self.num_envs,
            dtype=wp.float64,
            device=device,
        )
        logger.debug(
            "Initialized device score accumulation device=%s worlds=%d joints=%d",
            device,
            self.num_envs,
            len(self.joint_names),
        )

    def _builder_configuration_key(self, initial: Candidate) -> tuple[object, ...]:
        groups = {
            joint: group
            for group in self.config.joint_parameters
            for joint in group.names
        }
        joint_configuration = tuple(
            (
                joint,
                tuple(initial[joint][name] for name in self._PARAMETERS),
                float(groups[joint].controller.clamps.effort_limit),
                float(groups[joint].controller.clamps.velocity_limit),
                groups[joint].controller.implementation,
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
        group_by_joint = {
            joint: group
            for group in self.config.joint_parameters
            for joint in group.names
        }
        for joint in self.joint_names:
            _, dof_index = indices[joint]
            group = group_by_joint[joint]
            values = initial[joint]
            if group.controller.implementation == "external_implicit":
                # Newton's external actuator supplies effort; disable the native
                # drive with zero gains (VBD ignores joint_target_mode).
                template.joint_target_mode[dof_index] = effort_mode
                template.joint_target_ke[dof_index] = 0.0
                template.joint_target_kd[dof_index] = 0.0
            else:
                template.joint_target_mode[dof_index] = position_mode
                template.joint_target_ke[dof_index] = values["stiffness"]
                template.joint_target_kd[dof_index] = values["damping"]
            template.joint_armature[dof_index] = values["armature"]
            template.joint_friction[dof_index] = values["dry_friction"]
            template.joint_damping[dof_index] = values["viscous_friction"]
            template.joint_effort_limit[dof_index] = float(
                group.controller.clamps.effort_limit
            )
            if group.controller.implementation == "external_implicit":
                template.joint_velocity_limit[dof_index] = float("inf")
            else:
                template.joint_velocity_limit[dof_index] = float(
                    group.controller.clamps.velocity_limit
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
        self._apply_population(population)
        self._score_sums.zero_()
        transition_count = sum(
            self._evaluate_trajectory(trajectory) for trajectory in trajectories
        )
        scores = np.asarray(self._score_sums.numpy(), dtype=np.float64)
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
        for parameter_name, model_array_name in self._PASSIVE_ARRAYS.items():
            model_array = getattr(self._simulation_context.model, model_array_name)
            if model_array is None:
                raise RuntimeError(f"Newton model does not expose {model_array_name}")
            values = np.array(self._base_model_arrays[model_array_name], copy=True)
            for world_index, candidate in enumerate(population):
                for joint_index, joint in enumerate(self.joint_names):
                    values[self._joint_dof_indices[world_index, joint_index]] = (
                        candidate[joint][parameter_name]
                    )
            model_array.assign(values)
        for actuation in self._actuations:
            actuation.apply_population(population)

        flags = (
            newton.ModelFlags.JOINT_DOF_PROPERTIES
            | newton.ModelFlags.ACTUATOR_PROPERTIES
        )
        self._simulation_context.solver.notify_model_changed(flags)

    def _evaluate_trajectory(self, trajectory: Trajectory) -> int:
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
        self._reset(trajectory.measured_position[0])

        commanded = np.asarray(trajectory.commanded_position, dtype=np.float32)
        sample_indices = np.arange(commanded.shape[0], dtype=np.int64)[:, np.newaxis]
        source_indices = np.maximum(
            0,
            sample_indices - self.delay_steps[np.newaxis, :],
        )
        delayed_commands = np.take_along_axis(commanded, source_indices, axis=0)

        measured_positions = np.asarray(
            trajectory.measured_position[1:],
            dtype=np.float64,
        )
        measured_positions_device = wp.array(
            measured_positions.reshape(-1),
            dtype=wp.float64,
            device=self._device,
        )
        for sample_index in range(measured_positions.shape[0]):
            self._set_targets(delayed_commands[sample_index])
            self._step_control_interval()
            if sample_index >= self.buffer_steps:
                self._launch_position_error(measured_positions_device, sample_index)
        return measured_positions.shape[0] - self.buffer_steps

    def _launch_position_error(
        self,
        measured_positions: wp.array,
        sample_index: int,
    ) -> None:
        state_joint_q = self._simulation_context.state.joint_q
        if state_joint_q is None:
            raise RuntimeError("Newton state does not expose joint positions")
        wp.launch(
            _accumulate_position_error,
            dim=self.num_envs,
            inputs=[
                state_joint_q,
                self._joint_q_indices_device,
                measured_positions,
                sample_index,
                len(self.joint_names),
            ],
            outputs=[self._score_sums],
            device=self._device,
        )

    def _reset(self, measured_position: np.ndarray) -> None:
        joint_q = np.array(self._base_q, copy=True)
        joint_qd = np.array(self._base_qd, copy=True)
        joint_q[self._joint_q_indices] = measured_position[np.newaxis, :]
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

    def _set_targets(self, command: np.ndarray) -> None:
        targets = np.array(self._base_targets, copy=True)
        targets[self._joint_target_indices] = command[np.newaxis, :]
        joint_target_q = self._simulation_context.control.joint_target_q
        if joint_target_q is None:
            raise RuntimeError("Newton control does not expose joint position targets")
        joint_target_q.assign(targets)

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
