# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared USD resource identity and lifetime contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from anchor_bench.core.simulation import SimulationContext


class ToyContext(SimulationContext[object, object]):
    """Load resources using the USD assigned by the shared context."""

    load_count = 0

    def _load(self, usd_path: Path) -> tuple[object, object]:
        assert usd_path == self.usd_path
        assert usd_path.read_text() == "#usda 1.0\n"
        self.load_count += 1
        return object(), object()


class OtherToyContext(ToyContext):
    """Represent an incompatible backend using the same USD."""


@pytest.fixture(autouse=True)
def reset_context(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(SimulationContext, "_instance", None)


@pytest.fixture
def usd_path(tmp_path: Path) -> Path:
    path = tmp_path / "scene.usda"
    path.write_text("#usda 1.0\n")
    return path


def test_usd_context_loads_once(usd_path: Path) -> None:
    context = ToyContext(str(usd_path))

    assert context.usd_path == usd_path.resolve()
    assert context.load_count == 0
    model, state = context.load()
    assert context.model is model
    assert context.state is state
    assert context.load() == (model, state)
    assert context.load_count == 1
    assert SimulationContext.current() is context


def test_equivalent_usd_paths_reuse_loaded_context(usd_path: Path) -> None:
    context = ToyContext(str(usd_path))
    model, state = context.load()
    alias = usd_path.with_name("alias.usda")
    alias.symlink_to(usd_path)

    repeated = ToyContext(str(alias))

    assert repeated is context
    assert repeated.model is model
    assert repeated.state is state
    assert repeated.load_count == 1


def test_repeated_usd_conflicts_leave_original_context_usable(usd_path: Path) -> None:
    context = ToyContext(str(usd_path))
    model, state = context.load()
    other = usd_path.with_name("other.usda")
    other.write_text("#usda 1.0\n")

    for _ in range(2):
        with pytest.raises(RuntimeError, match="already uses"):
            ToyContext(str(other))

    assert ToyContext(str(usd_path)) is context
    assert context.model is model
    assert context.state is state
    assert context.load_count == 1


def test_repeated_base_mode_conflicts_preserve_context(usd_path: Path) -> None:
    context = ToyContext(str(usd_path), fixed_base=True)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="different base mode"):
            ToyContext(str(usd_path), fixed_base=False)
    assert ToyContext(str(usd_path), fixed_base=True) is context
    assert context.fixed_base is True


def test_repeated_backend_conflicts_leave_original_context_usable(
    usd_path: Path,
) -> None:
    context = ToyContext(str(usd_path))

    for _ in range(2):
        with pytest.raises(RuntimeError, match="Only one simulation backend"):
            OtherToyContext(str(usd_path))

    assert ToyContext(str(usd_path)) is context
    assert SimulationContext.current() is context


@pytest.mark.parametrize("directory", [False, True])
def test_invalid_usd_path_does_not_claim_context(
    usd_path: Path, directory: bool
) -> None:
    invalid = usd_path.parent if directory else usd_path.with_name("missing.usda")
    with pytest.raises(FileNotFoundError, match="USD file does not exist"):
        ToyContext(str(invalid))
    context = ToyContext(str(usd_path))
    assert SimulationContext.current() is context


def test_usd_path_is_required() -> None:
    context_type: Any = ToyContext
    with pytest.raises(TypeError):
        context_type()
    with pytest.raises(TypeError):
        context_type(resource_key=("toy", 2))
