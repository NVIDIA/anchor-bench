# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pandas as pd
import pytest

from anchor_bench.train import (
    friction_torque,
    position_error,
    solver_pd,
    torque_residual,
)


def test_shared_features_support_live_scalar_values() -> None:
    """Calculate live inference features with scalar values."""

    error = position_error(2.0, 1.5)
    solver = solver_pd(error, 0.25, 4.0, 2.0)
    friction = friction_torque(0.25, 0.2, 0.4)

    assert error == 0.5
    assert solver == 1.5
    assert friction == pytest.approx(0.3)
    assert torque_residual(5.0, solver, friction) == pytest.approx(3.2)


def test_shared_features_support_offline_series_values() -> None:
    """Apply identical feature math to vectorized pandas columns."""

    command = pd.Series([2.0, 1.0])
    position = pd.Series([1.0, 1.5])
    velocity = pd.Series([2.0, -2.0])
    effort = pd.Series([10.0, 10.0])

    error = position_error(command, position)
    solver = solver_pd(error, velocity, 2.0, 0.5)
    friction = friction_torque(velocity, 0.2, 0.05)
    residual = torque_residual(effort, solver, friction)

    assert error.tolist() == [1.0, -0.5]
    assert solver.tolist() == [1.0, 0.0]
    assert friction.tolist() == pytest.approx([0.3, -0.3])
    assert residual.tolist() == pytest.approx([8.7, 10.3])
