# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np

from anchor_bench.tasks.actuator_sys_id.newton_actuation import NativePD


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
    assert NativePD.runs_before_substep is False
