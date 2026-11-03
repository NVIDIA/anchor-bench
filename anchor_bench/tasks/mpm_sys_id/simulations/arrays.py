# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Require arrays allocated by the particle and rigid-body builders."""

from typing import Optional, TypeVar

T = TypeVar("T")


def require_array(value: Optional[T]) -> T:
    if value is None:
        raise RuntimeError("Simulation is missing a required model/state array")
    return value
