# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import newton
import numpy as np
import warp as wp

from anchor_bench.tasks.actuator_sys_id.newton_actuation import (
    ExternalImplicitPD,
    NativePD,
)


class FakeArray:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)

    def numpy(self):
        return self.values

    def assign(self, values):
        self.values = np.asarray(values, dtype=np.float32)


def test_native_pd_writes_each_world_gains_and_keeps_other_dofs():
    model = SimpleNamespace(
        joint_target_ke=FakeArray([9.0, 0.0, 9.0, 0.0]),
        joint_target_kd=FakeArray([8.0, 0.0, 8.0, 0.0]),
    )
    actuation = NativePD(
        cast(Any, model), joint_names=["hip"], dof_indices=np.asarray([[1], [3]])
    )

    actuation.apply_population(
        [
            {"hip": {"stiffness": 10.0, "damping": 1.0}},
            {"hip": {"stiffness": 20.0, "damping": 2.0}},
        ]
    )

    np.testing.assert_allclose(model.joint_target_ke.values, [9.0, 10.0, 9.0, 20.0])
    np.testing.assert_allclose(model.joint_target_kd.values, [8.0, 1.0, 8.0, 2.0])
    assert actuation.is_graphable()


def _pendulum_model(worlds: int = 2) -> newton.Model:
    template = newton.ModelBuilder()
    link = template.add_link(
        mass=1.0, com=(0.0, 0.0, -0.5), inertia=wp.mat33(np.eye(3) * 0.1)
    )
    joint = template.add_joint_revolute(parent=-1, child=link, axis=(0.0, 1.0, 0.0))
    template.add_articulation([joint])
    builder = newton.ModelBuilder()
    builder.replicate(template, copy_count=worlds)
    return builder.finalize(device="cpu")


def test_external_pd_writes_candidate_gains_to_the_actuator():
    with wp.ScopedDevice("cpu"):
        model = _pendulum_model()
        indices = np.asarray([[0], [1]])
        actuation = ExternalImplicitPD(
            model,
            joint_names=["hinge"],
            q_indices=indices,
            dof_indices=indices,
            target_indices=indices,
            effort_limits=[5.0],
        )
        actuation.apply_population(
            [
                {"hinge": {"stiffness": 10.0, "damping": 1.0}},
                {"hinge": {"stiffness": 20.0, "damping": 2.0}},
            ]
        )
        np.testing.assert_allclose(actuation._kp.numpy(), [10.0, 20.0])
        np.testing.assert_allclose(actuation._kd.numpy(), [1.0, 2.0])


def test_external_pd_reset_clears_joint_effort_and_substep_writes_it():
    with wp.ScopedDevice("cpu"):
        model = _pendulum_model()
        indices = np.asarray([[0], [1]])
        actuation = ExternalImplicitPD(
            model,
            joint_names=["hinge"],
            q_indices=indices,
            dof_indices=indices,
            target_indices=indices,
            effort_limits=[5.0],
        )
        actuation.apply_population([{"hinge": {"stiffness": 50.0, "damping": 1.0}}] * 2)
        state, control = model.state(), model.control()
        assert control.joint_target_q is not None
        control.joint_target_q.assign(np.asarray([0.5, 0.5], dtype=np.float32))
        actuation.before_substep(state, control, 0.01)
        assert control.joint_f is not None
        effort = control.joint_f.numpy()
        assert np.all(np.abs(effort) > 0.0)
        assert np.all(np.abs(effort) <= 5.0 + 1e-5)  # clamped inside the solve

        actuation.reset(control)
        np.testing.assert_array_equal(control.joint_f.numpy(), [0.0, 0.0])
