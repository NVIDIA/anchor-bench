# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared material controls for the sand system-identification examples."""

import math

DEFAULTS = {
    "young_modulus": 1.0e7,
    "poisson_ratio": 0.3,
    "damping": 0.0,
    "friction": 0.65,
    "yield_pressure": 1.0e12,
    "tensile_yield_ratio": 0.0,
    "yield_stress": 0.0,
}
# Newton's registered defaults; deliberately excluded from the fit search space.
FROZEN = {
    "hardening": 0.0,
    "hardening_rate": 1.0,
    "softening_rate": 1.0,
    "dilatancy": 0.0,
    "viscosity": 0.0,
}


def apply_material(model, args):
    """Assign every material attribute before constructing the MPM solver."""
    for name, default in {**DEFAULTS, **FROZEN}.items():
        value = getattr(args, name, default)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
        if name in ("young_modulus", "yield_pressure") and value == 0:
            raise ValueError(f"{name} must be positive")
        if name == "poisson_ratio" and value >= 0.5:
            raise ValueError("poisson_ratio must be below 0.5")
        if name == "tensile_yield_ratio" and value > 1:
            raise ValueError("tensile_yield_ratio must not exceed 1")
        getattr(model.mpm, name).fill_(value)
