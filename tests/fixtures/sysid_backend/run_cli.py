# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Enter the real CLI with Newton construction forbidden in this process."""

import newton

from anchor_bench.core.simulation.newton import NewtonSimulationContext


def forbid_newton(*args: object, **kwargs: object) -> None:
    raise AssertionError("installed toy backend must not construct Newton resources")


NewtonSimulationContext.__new__ = staticmethod(forbid_newton)  # pyright: ignore[reportAttributeAccessIssue]
newton.ModelBuilder = forbid_newton  # pyright: ignore[reportAssignmentType]

# Import the CLI only after installing the construction guards.
from anchor_bench.cli import main

raise SystemExit(main())
