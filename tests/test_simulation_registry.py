# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from anchor_bench.core import NewtonSimulationContext, SimulationContext
from anchor_bench.core.simulation.registry import create_simulation_context


def test_creates_context_for_registered_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve the configured backend without loading its runtime."""

    monkeypatch.setattr(SimulationContext, "_instance", None)
    usd_path = tmp_path / "robot.usda"
    usd_path.write_text("#usda 1.0", encoding="utf-8")

    context = create_simulation_context("newton", str(usd_path))

    assert isinstance(context, NewtonSimulationContext)
    assert context.usd_path == usd_path.resolve()


def test_rejects_unknown_backend() -> None:
    """Reject backend names missing from the registry."""

    with pytest.raises(ValueError, match="Unknown simulation backend 'missing'"):
        create_simulation_context("missing", "robot.usda")


def test_registry_exposes_typed_sysid_registration() -> None:
    from anchor_bench.core.simulation import registry

    assert hasattr(registry, "SysIdBackend")
    assert hasattr(registry, "register_simulation_backend")
