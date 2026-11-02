# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How candidate PD gains reach a Newton simulation during sysID."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import ClassVar, cast

import newton
import numpy as np
import warp as wp
from newton.actuators import Actuator, ClampingMaxEffort, DrivePD, JointSpaceResponse

from anchor_bench.tasks.actuator_sys_id.model import Candidate


class NewtonActuation(ABC):
    """Apply candidate PD gains for a subset of joints across all worlds."""

    @abstractmethod
    def apply_population(self, population: Sequence[Candidate]) -> None:
        """Write each world's candidate stiffness and damping."""

    def before_substep(
        self, state: newton.State, control: newton.Control | None, dt: float
    ) -> None:
        """Update actuator efforts before a physics substep."""

    def reset(self, control: newton.Control) -> None:
        """Clear per-episode actuator state when a trajectory restarts."""

    def is_graphable(self) -> bool:
        """Return whether before_substep can run inside a captured graph."""
        return True


class NativePD(NewtonActuation):
    """Fit gains through the solver's own joint position drive."""

    _ARRAYS: ClassVar[tuple[tuple[str, str], ...]] = (
        ("stiffness", "joint_target_ke"),
        ("damping", "joint_target_kd"),
    )

    def __init__(
        self,
        model: newton.Model,
        *,
        joint_names: Sequence[str],
        dof_indices: np.ndarray,
    ) -> None:
        self._model = model
        self._joint_names = tuple(joint_names)
        self._dof_indices = dof_indices
        self._base = {
            array_name: np.array(getattr(model, array_name).numpy(), copy=True)
            for _, array_name in self._ARRAYS
        }

    def apply_population(self, population: Sequence[Candidate]) -> None:
        for parameter, array_name in self._ARRAYS:
            values = np.array(self._base[array_name], copy=True)
            for world, candidate in enumerate(population):
                for column, joint in enumerate(self._joint_names):
                    values[self._dof_indices[world, column]] = candidate[joint][
                        parameter
                    ]
            getattr(self._model, array_name).assign(values)


@wp.kernel
# Warp's array subscription is its public kernel annotation syntax.
def _sum_bias(
    gravity: wp.array(dtype=float),  # pyright: ignore[reportInvalidTypeForm]
    coriolis: wp.array(dtype=float),  # pyright: ignore[reportInvalidTypeForm]
    bias: wp.array(dtype=float),  # pyright: ignore[reportInvalidTypeForm]
):
    index = wp.tid()
    bias[index] = gravity[index] + coriolis[index]


class ExternalImplicitPD(NewtonActuation):
    """Newton's implicit PD actuator with effort clamping and bias feedback.

    The PD law is solved against the coupled joint-space response, with the
    passive gravity and Coriolis bias at the current state, before every
    physics substep. Velocity limits are not applied.
    """

    def __init__(
        self,
        model: newton.Model,
        *,
        joint_names: Sequence[str],
        q_indices: np.ndarray,
        dof_indices: np.ndarray,
        target_indices: np.ndarray,
        effort_limits: Sequence[float],
    ) -> None:
        device = model.device
        count = dof_indices.size
        world_count = dof_indices.shape[0]
        drive = DrivePD(
            kp=wp.zeros(count, dtype=float, device=device),
            kd=wp.zeros(count, dtype=float, device=device),
        )
        clamp = ClampingMaxEffort(
            max_effort=wp.array(
                np.tile(np.asarray(effort_limits, dtype=np.float32), world_count),
                dtype=float,
                device=device,
            )
        )
        response = JointSpaceResponse(model)
        actuator = Actuator(
            indices=wp.array(dof_indices.ravel(), dtype=wp.uint32, device=device),
            drive=drive,
            clamping=[clamp],
            pos_indices=wp.array(q_indices.ravel(), dtype=wp.uint32, device=device),
            target_pos_indices=wp.array(
                target_indices.ravel(), dtype=wp.uint32, device=device
            ),
            control_feedforward_attr=None,
        )
        actuator.set_effort_mode_implicit(response)
        self._model = model
        self._joint_names = tuple(joint_names)
        self._response = response
        self._actuator = actuator
        # Binding replaces the constructor arrays with live views; keep those.
        self._kp = drive.kp
        self._kd = drive.kd
        self._bias_state = model.state()
        self._gravity = wp.zeros(model.joint_dof_count, dtype=float, device=device)
        self._coriolis = wp.zeros_like(self._gravity)
        self._bias = wp.zeros_like(self._gravity)

    def apply_population(self, population: Sequence[Candidate]) -> None:
        for parameter, target in (("stiffness", self._kp), ("damping", self._kd)):
            target.assign(
                np.asarray(
                    [
                        candidate[joint][parameter]
                        for candidate in population
                        for joint in self._joint_names
                    ],
                    dtype=np.float32,
                )
            )

    def before_substep(
        self, state: newton.State, control: newton.Control | None, dt: float
    ) -> None:
        self._response.refresh(state)
        # Evaluate bias on a scratch state so solver-owned body poses are untouched.
        # Finalized models always provide joint coordinates and velocities.
        joint_q = cast("wp.array", state.joint_q)
        joint_qd = cast("wp.array", state.joint_qd)
        self._bias_state.joint_q = joint_q
        self._bias_state.joint_qd = joint_qd
        newton.eval_fk(self._model, joint_q, joint_qd, self._bias_state)
        newton.eval_inverse_dynamics_passive(
            self._model,
            self._bias_state,
            gravity_force=self._gravity,
            coriolis_force=self._coriolis,
        )
        wp.launch(
            _sum_bias,
            dim=self._model.joint_dof_count,
            inputs=[self._gravity, self._coriolis, self._bias],
            device=self._model.device,
        )
        # The stepper always provides a control with joint efforts.
        self._actuator.step(
            state, cast("newton.Control", control), dt=dt, bias_force=self._bias
        )

    def reset(self, control: newton.Control) -> None:
        if control.joint_f is not None:
            control.joint_f.zero_()

    def is_graphable(self) -> bool:
        return self._actuator.is_graphable()
