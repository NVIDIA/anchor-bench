# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Register a custom backend and enter the CLI with Newton resources forbidden."""

import sys

import newton

from anchor_bench.core.simulation.newton import NewtonSimulationContext


def forbid_newton(*args: object, **kwargs: object) -> None:
    raise AssertionError("custom toy backend must not construct Newton resources")


NewtonSimulationContext.__new__ = staticmethod(forbid_newton)  # pyright: ignore[reportAttributeAccessIssue]
newton.ModelBuilder = forbid_newton  # pyright: ignore[reportAssignmentType]

# Import and register the backend only after installing the construction guards.
import toy_sysid_backend

from anchor_bench.cli import main

provider = sys.argv.pop(1)
getattr(toy_sysid_backend, provider)()

raise SystemExit(main())
