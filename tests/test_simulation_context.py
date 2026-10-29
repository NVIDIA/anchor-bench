# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend-neutral resource identity and lifetime contracts."""

from __future__ import annotations

from typing import Any

import pytest

from anchor_bench.core.simulation import SimulationContext


class ToyContext(SimulationContext[object, object]):
    """Load in-memory resources without any filesystem dependency."""

    load_count = 0

    def _load(self) -> tuple[object, object]:
        self.load_count += 1
        return object(), object()


class OtherToyContext(ToyContext):
    """Represent an incompatible backend with the same resource key."""


@pytest.fixture(autouse=True)
def reset_context(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(SimulationContext, "_instance", None)


def test_path_free_context_loads_once() -> None:
    context = ToyContext(resource_key=("toy", 2))

    assert context.resource_key == ("toy", 2)
    assert context.load_count == 0
    model, state = context.load()
    assert context.model is model
    assert context.state is state
    assert context.load() == (model, state)
    assert context.load_count == 1
    assert SimulationContext.current() is context


def test_equivalent_resource_keys_reuse_loaded_context() -> None:
    context = ToyContext(resource_key=("toy", 2))
    model, state = context.load()

    resource_parts = ["toy", 2]
    repeated = ToyContext(resource_key=tuple(resource_parts))

    assert repeated is context
    assert repeated.model is model
    assert repeated.state is state
    assert repeated.load_count == 1


def test_repeated_resource_key_conflicts_leave_original_context_usable() -> None:
    context = ToyContext(resource_key=("toy", 2))
    model, state = context.load()

    for _ in range(2):
        with pytest.raises(RuntimeError, match="already uses"):
            ToyContext(resource_key=("toy", 3))

    assert ToyContext(resource_key=("toy", 2)) is context
    assert context.model is model
    assert context.state is state
    assert context.load_count == 1


def test_repeated_backend_conflicts_leave_original_context_usable() -> None:
    context = ToyContext(resource_key=("toy", 2))

    for _ in range(2):
        with pytest.raises(RuntimeError, match="Only one simulation backend"):
            OtherToyContext(resource_key=("toy", 2))

    assert ToyContext(resource_key=("toy", 2)) is context
    assert SimulationContext.current() is context


@pytest.mark.parametrize("resource_key", [["toy", 2], ("toy", [])])
def test_unhashable_resource_key_does_not_claim_context(resource_key: Any) -> None:
    with pytest.raises(TypeError, match="hashable"):
        ToyContext(resource_key=resource_key)

    context = ToyContext(resource_key=("toy", 2))
    assert SimulationContext.current() is context


def test_resource_key_is_required_keyword_only() -> None:
    context_type: Any = ToyContext

    with pytest.raises(TypeError):
        context_type()
    with pytest.raises(TypeError):
        context_type(("toy", 2))
