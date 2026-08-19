# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import newton
import numpy as np
import pytest
import warp as wp

from anchor_bench.train.utils.mujoco_bias import MuJoCoBiasEvaluator


@pytest.mark.integration
def test_evaluator_preserves_coupled_state_with_real_mujoco(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(wp.config, "kernel_cache_dir", str(tmp_path / "warp-cache"))
    builder = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    inertia = wp.mat33(
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )
    parent = builder.add_link(
        mass=1.0,
        com=(0.0, 0.0, 0.5),
        inertia=inertia,
        label="parent",
    )
    parent_joint = builder.add_joint_revolute(
        -1,
        parent,
        axis=newton.Axis.Y,
        label="parent_joint",
    )
    child = builder.add_link(
        mass=1.0,
        com=(0.0, 0.0, 0.5),
        inertia=inertia,
        label="child",
    )
    child_joint = builder.add_joint_revolute(
        parent,
        child,
        parent_xform=wp.transform(
            (0.0, 0.0, 1.0),
            wp.quat(0.0, 0.0, 0.0, 1.0),
        ),
        axis=newton.Axis.Y,
        label="child_joint",
    )
    builder.add_articulation([parent_joint, child_joint])
    model = builder.finalize(device="cpu")
    evaluator = MuJoCoBiasEvaluator(
        model,
        ("parent_joint", "child_joint"),
        output_entities=("parent_joint",),
    )

    values = evaluator.evaluate(
        np.asarray([[0.2, 0.0], [0.2, 0.7]]),
        np.asarray([[0.1, 0.0], [0.1, 0.8]]),
    )

    assert values.shape == (2, 1)
    assert np.all(np.isfinite(values))
    assert values[0, 0] != pytest.approx(values[1, 0])
