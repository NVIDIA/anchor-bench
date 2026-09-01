# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# pyright: reportAttributeAccessIssue=false

"""Native Newton population evaluator for actuator system identification."""

from __future__ import annotations

import logging
from math import isclose
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Sequence, Tuple, cast

import newton
import numpy as np
import warp as wp
from pxr import Usd, UsdPhysics

from anchor_bench.assets import resolve_usd_path
from anchor_bench.core.simulation.newton import NewtonSimulationContext
from anchor_bench.core.simulation.registry import create_simulation_context
from anchor_bench.schema.actuator_sys_id import ActuatorSysIdConfig
from anchor_bench.tasks.actuator_sys_id.model import Candidate, Trajectory
from anchor_bench.tasks.actuator_sys_id.parameter_space import ParameterSpace

# The OpenUSD extension modules do not publish complete Python type information.

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


class NewtonPopulationEvaluator:
    """Evaluate one candidate per replicated Newton world."""

    _MODEL_ARRAYS: ClassVar[Dict[str, str]] = {
        "stiffness": "joint_target_ke",
        "damping": "joint_target_kd",
        "armature": "joint_armature",
        "dry_friction": "joint_friction",
        "viscous_friction": "joint_damping",
    }

    def __init__(
        self,
        config: ActuatorSysIdConfig,
        parameter_space: ParameterSpace,
        source_usd_path: Path | None = None,
    ) -> None:
        self.config = config
        self.parameter_space = parameter_space
        self.source_usd_path = source_usd_path or resolve_usd_path(
            config.simulation.usd
        )
        self.joint_names = _joint_names(config)
        self.body_names = [body.name for body in config.rigid_body_parameters]
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
        self._initialize_simulation()
        self._initialize_score_buffers()
        self._simulation_context.capture_step_graph(
            dt=self.physics_dt * self.control_substeps,
            substeps=self.control_substeps,
        )

    def _initialize_simulation(self) -> None:
        initial = self.parameter_space.decode(self.parameter_space.initial_normalized)
        ignored_actuators = _mujoco_actuator_paths(
            self.source_usd_path,
            self.joint_names,
        )
        context = create_simulation_context(
            self.config.simulation.type,
            str(self.source_usd_path),
            fixed_base=True,
        )
        if not isinstance(context, NewtonSimulationContext):
            raise TypeError("actuator_sys_id requires a Newton simulation context")
        self._simulation_context = context

        def configure_template(template: newton.ModelBuilder) -> None:
            template_indices = _builder_joint_indices(template, self.joint_names)
            body_indices = _builder_body_indices(template, self.body_names)
            self._configure_template(
                template,
                template_indices,
                body_indices,
                initial,
            )

        solver_config = self.config.simulation.solver
        context.initialize(
            world_count=self.num_envs,
            configure_builder=configure_template,
            configuration_key=self._builder_configuration_key(initial),
            solver_options={
                "solver": "newton",
                "integrator": solver_config.integrator,
                "iterations": solver_config.solver_iterations,
                "ls_iterations": solver_config.line_search_iterations,
                "njmax": 512,
            },
            usd_ignore_paths=ignored_actuators,
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
        self._body_indices = _model_body_indices(
            model,
            self.body_names,
            self.num_envs,
        )
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
        for model_array_name in self._MODEL_ARRAYS.values():
            model_array = getattr(model, model_array_name)
            if model_array is None:
                raise RuntimeError(f"Newton model does not expose {model_array_name}")
            self._base_model_arrays[model_array_name] = np.array(
                model_array.numpy(),
                dtype=np.float32,
                copy=True,
            )
        self._base_body_arrays: Dict[str, np.ndarray] = {}
        if self.body_names:
            for model_array_name in (
                "body_mass",
                "body_inv_mass",
                "body_inertia",
                "body_inv_inertia",
            ):
                model_array = getattr(model, model_array_name)
                if model_array is None:
                    raise RuntimeError(
                        f"Newton model does not expose {model_array_name}"
                    )
                self._base_body_arrays[model_array_name] = np.array(
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
                tuple(initial[joint][name] for name in self._MODEL_ARRAYS),
                float(groups[joint].controller.clamps.effort_limit),
                float(groups[joint].controller.clamps.velocity_limit),
            )
            for joint in self.joint_names
        )
        body_configuration = tuple(
            (body, initial[body]["mass"]) for body in self.body_names
        )
        return (self.config.type, joint_configuration, body_configuration)

    def _configure_template(
        self,
        template: newton.ModelBuilder,
        indices: Dict[str, Tuple[int, int]],
        body_indices: Dict[str, int],
        initial: Candidate,
    ) -> None:
        position_mode = int(newton.JointTargetMode.POSITION)
        group_by_joint = {
            joint: group
            for group in self.config.joint_parameters
            for joint in group.names
        }
        for joint in self.joint_names:
            _, dof_index = indices[joint]
            group = group_by_joint[joint]
            values = initial[joint]
            template.joint_target_mode[dof_index] = position_mode
            template.joint_target_ke[dof_index] = values["stiffness"]
            template.joint_target_kd[dof_index] = values["damping"]
            template.joint_armature[dof_index] = values["armature"]
            template.joint_friction[dof_index] = values["dry_friction"]
            template.joint_damping[dof_index] = values["viscous_friction"]
            damping_alias = getattr(template, "custom_attributes", {}).get(
                "mujoco:dof_passive_damping"
            )
            if damping_alias is not None:
                alias_values = damping_alias.values
                alias_has_dof = (
                    isinstance(alias_values, dict)
                    and dof_index in alias_values
                    or isinstance(alias_values, list)
                    and dof_index < len(alias_values)
                )
                if alias_has_dof:
                    alias_values[dof_index] = values["viscous_friction"]
            template.joint_effort_limit[dof_index] = float(
                group.controller.clamps.effort_limit
            )
            template.joint_velocity_limit[dof_index] = float(
                group.controller.clamps.velocity_limit
            )
        for body in self.body_names:
            _set_builder_body_mass(
                template,
                body_indices[body],
                initial[body]["mass"],
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
        invalid_worlds = np.flatnonzero(~np.isfinite(scores))
        if invalid_worlds.size:
            finite_scores = scores[np.isfinite(scores)]
            if not finite_scores.size:
                raise RuntimeError(
                    "Newton produced non-finite position errors for every "
                    f"candidate ({self.num_envs} worlds)"
                )
            worst_finite_score = float(np.max(finite_scores))
            penalty_base = max(1.0, worst_finite_score)
            maximum_score = np.finfo(np.float64).max
            penalty = (
                penalty_base * 2.0
                if penalty_base <= maximum_score / 2.0
                else maximum_score
            )
            scores[invalid_worlds] = penalty
            reported_worlds = invalid_worlds[:16].tolist()
            omitted_world_count = max(0, invalid_worlds.size - len(reported_worlds))
            logger.warning(
                "Newton produced non-finite position errors; penalizing "
                "candidates worlds=%s omitted_worlds=%d penalty=%g invalid=%d/%d",
                reported_worlds,
                omitted_world_count,
                penalty,
                invalid_worlds.size,
                self.num_envs,
            )
        return scores

    def _apply_population(self, population: Sequence[Candidate]) -> None:
        for parameter_name, model_array_name in self._MODEL_ARRAYS.items():
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

        flags = (
            newton.ModelFlags.JOINT_DOF_PROPERTIES
            | newton.ModelFlags.ACTUATOR_PROPERTIES
        )
        if self.body_names:
            body_mass = np.array(self._base_body_arrays["body_mass"], copy=True)
            body_inv_mass = np.array(self._base_body_arrays["body_inv_mass"], copy=True)
            body_inertia = np.array(self._base_body_arrays["body_inertia"], copy=True)
            body_inv_inertia = np.array(
                self._base_body_arrays["body_inv_inertia"], copy=True
            )
            for world_index, candidate in enumerate(population):
                for body_index, body in enumerate(self.body_names):
                    model_index = self._body_indices[world_index, body_index]
                    original_mass = self._base_body_arrays["body_mass"][model_index]
                    mass = candidate[body]["mass"]
                    scale = mass / original_mass
                    body_mass[model_index] = mass
                    body_inv_mass[model_index] = 1.0 / mass
                    body_inertia[model_index] *= scale
                    body_inv_inertia[model_index] /= scale
            model = self._simulation_context.model
            model_body_mass = model.body_mass
            model_body_inv_mass = model.body_inv_mass
            model_body_inertia = model.body_inertia
            model_body_inv_inertia = model.body_inv_inertia
            if (
                model_body_mass is None
                or model_body_inv_mass is None
                or model_body_inertia is None
                or model_body_inv_inertia is None
            ):
                raise RuntimeError("Newton model does not expose body inertia arrays")
            model_body_mass.assign(body_mass)
            model_body_inv_mass.assign(body_inv_mass)
            model_body_inertia.assign(body_inertia)
            model_body_inv_inertia.assign(body_inv_inertia)
            flags |= newton.ModelFlags.BODY_INERTIAL_PROPERTIES
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


def _mujoco_actuator_paths(
    source_usd_path: Path,
    joint_names: Sequence[str],
) -> tuple[str, ...]:
    """Return inherited MuJoCo actuators targeting identified joints."""

    stage = Usd.Stage.Open(str(source_usd_path))
    if stage is None:
        raise ValueError(f"could not open source USD: {source_usd_path}")
    joint_paths = set()
    for joint_name in joint_names:
        matches = [
            prim
            for prim in stage.Traverse()
            if prim.IsA(UsdPhysics.Joint)
            and (str(prim.GetPath()) == joint_name or prim.GetName() == joint_name)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Newton asset must contain exactly one joint matching {joint_name!r}"
            )
        joint_paths.add(matches[0].GetPath())

    return tuple(
        str(prim.GetPath())
        for prim in stage.Traverse()
        if str(prim.GetTypeName()) == "MjcActuator"
        and (target := prim.GetRelationship("mjc:target"))
        and joint_paths.intersection(target.GetTargets())
    )


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


def _builder_body_indices(
    builder: newton.ModelBuilder,
    body_names: Sequence[str],
) -> Dict[str, int]:
    """Resolve configured rigid bodies in one unreplicated builder."""

    return {
        body_name: _matching_label(builder.body_label, body_name)
        for body_name in body_names
    }


def _set_builder_body_mass(
    builder: newton.ModelBuilder,
    body_index: int,
    mass: float,
) -> None:
    """Set one builder mass while preserving its inertia shape and COM."""

    original_mass = float(builder.body_mass[body_index])
    if original_mass <= 0.0:
        raise ValueError("identified rigid bodies must have positive source mass")
    scale = mass / original_mass
    builder.body_mass[body_index] = mass
    builder.body_inv_mass[body_index] = 1.0 / mass
    inertia = cast(Any, builder.body_inertia[body_index])
    inverse_inertia = cast(Any, builder.body_inv_inertia[body_index])
    builder.body_inertia[body_index] = inertia * scale
    builder.body_inv_inertia[body_index] = inverse_inertia / scale


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


def _model_body_indices(
    model: newton.Model,
    body_names: Sequence[str],
    num_envs: int,
) -> np.ndarray:
    """Resolve one body index per configured body and replicated world."""

    if not body_names:
        return np.empty((num_envs, 0), dtype=np.int64)
    body_world = model.body_world
    if body_world is None:
        raise RuntimeError("Newton model does not expose body index arrays")
    worlds = np.asarray(body_world.numpy(), dtype=np.int64)
    body_indices = np.empty((num_envs, len(body_names)), dtype=np.int64)
    for world_index in range(num_envs):
        world_body_indices = np.flatnonzero(worlds == world_index)
        world_labels = [model.body_label[index] for index in world_body_indices]
        for local_index, body_name in enumerate(body_names):
            label_index = _matching_label(world_labels, body_name)
            body_indices[world_index, local_index] = world_body_indices[label_index]
    return body_indices
