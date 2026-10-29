# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Construct supported simulations lazily in the CUDA worker."""

from types import SimpleNamespace
from typing import Protocol


class Simulation(Protocol):
    def step(self) -> None: ...
    def measure(self) -> dict: ...
    def test_final(self) -> None: ...


def create_simulation(settings: dict, material: dict, cuda_graphs: bool) -> Simulation:
    from .bottle_drop import Simulation as BottleDrop
    from .sand_pour import Simulation as SandPour

    implementations = {"sand_pour": SandPour, "bottle_drop": BottleDrop}
    options = SimpleNamespace(**settings, **material, no_cuda_graph=not cuda_graphs)
    return implementations[settings["type"]](options)
