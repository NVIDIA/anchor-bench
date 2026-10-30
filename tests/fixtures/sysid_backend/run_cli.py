# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select a test context and enter the CLI with Newton resources forbidden."""

import sys

import newton

from anchor_bench.core.simulation.newton import NewtonSimulationContext


def forbid_newton(*args: object, **kwargs: object) -> None:
    raise AssertionError("custom toy backend must not construct Newton resources")


NewtonSimulationContext.__new__ = staticmethod(forbid_newton)  # pyright: ignore[reportAttributeAccessIssue]
newton.ModelBuilder = forbid_newton  # pyright: ignore[reportAssignmentType]

# Add the test context to the supported implementations after guarding Newton.
import toy_sysid_backend

from anchor_bench.cli import main
from anchor_bench.core.simulation.registry import SIMULATION_CONTEXTS

context_name = sys.argv.pop(1)
SIMULATION_CONTEXTS["toy-sysid"] = getattr(toy_sysid_backend, context_name)

raise SystemExit(main())
