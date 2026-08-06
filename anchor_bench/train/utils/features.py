"""Shared feature calculations for training and live inference."""

from __future__ import annotations

from typing import Any

import numpy as np


def position_error(command_q: Any, actual_q: Any) -> Any:
    """Return commanded position minus measured position."""

    return command_q - actual_q


def solver_pd(
    error: Any,
    actual_dq: Any,
    kp: float,
    kd: float,
) -> Any:
    """Return proportional-derivative solver effort."""

    return kp * error - kd * actual_dq


def friction_torque(
    actual_dq: Any,
    dry_friction: float,
    viscous_friction: float,
) -> Any:
    """Return dry plus viscous friction effort."""

    return dry_friction * np.sign(actual_dq) + viscous_friction * actual_dq


def torque_residual(
    actual_effort: Any,
    solver_effort: Any,
    friction_effort: Any,
) -> Any:
    """Return effort remaining after solver and friction baselines."""

    return actual_effort - solver_effort - friction_effort
