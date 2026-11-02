# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Construct supported simulations lazily in the CUDA worker."""

from types import SimpleNamespace
from typing import Protocol

from pydantic import TypeAdapter

from anchor_bench.schema.mpm_sys_id import SimulationConfig

_SETTINGS_ADAPTER = TypeAdapter(SimulationConfig)


class Simulation(Protocol):
    def step(self) -> None: ...
    def measure(self) -> dict: ...
    def test_final(self) -> None: ...


def create_simulation(settings: dict, material: dict, cuda_graphs: bool) -> Simulation:
    """Validate settings through the schema before allocating simulation state.

    This boundary also validates direct callers and settings modified after
    config parsing. Constructors consume resolved settings from this factory.
    """
    validated = _SETTINGS_ADAPTER.validate_python(settings)

    from .bottle_drop import Simulation as BottleDrop
    from .sand_pour import Simulation as SandPour

    implementations = {"sand_pour": SandPour, "bottle_drop": BottleDrop}
    options = SimpleNamespace(
        **validated.model_dump(mode="python"), **material, no_cuda_graph=not cuda_graphs
    )
    return implementations[validated.type](options)
