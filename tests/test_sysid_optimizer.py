"""Profile-only CMA-ES adapter tests for authoritative SysID."""

from __future__ import annotations

import sys
from unittest.mock import patch

import numpy as np
import pytest

from anchor_bench.sysid.optimizer import (
    CMAESOptimizer,
    _MIN_CONVERGENCE_GENERATIONS,
)
from anchor_bench.sysid.schema import SysIdProfile


def _profile() -> SysIdProfile:
    """Build a small mixed per-joint/shared optimizer profile."""

    return SysIdProfile.from_dict(
        {
            "schema_version": 1,
            "data": "inputs/demo",
            "joints": ["a", "b"],
            "simulation": {
                "physics_freq": 500,
                "control_freq": 500,
                "render_freq": 500,
                "integrator": "implicitfast",
                "solver_iterations": 10,
                "line_search_iterations": 10,
            },
            "num_envs": 3,
            "buffer_time": 0.0,
            "seed": 11,
            "optimizer": {
                "type": "cmaes",
                "sigma": 0.5,
                "max_iterations": 20,
                "epsilon": 0.01,
            },
            "parameters": {
                "stiffness": {
                    "mode": "per_joint",
                    "bounds": {"lower": 0.0, "upper": 100.0},
                },
                "damping": {
                    "mode": "shared",
                    "bounds": {"lower": 0.0, "upper": 10.0},
                },
            },
            "loss": {"position": 1.0, "velocity": 0.0},
        },
        "test profile",
    )


def _optimizer(*, seed: int = 11, num_envs: int = 3) -> CMAESOptimizer:
    """Construct a deterministic host optimizer from an explicit parent."""

    profile = _profile()
    return CMAESOptimizer(
        profile,
        num_envs=num_envs,
        cmaes=profile.optimizer,
        seed=seed,
        allow_random_fallback=True,
        parent_parameters={
            "stiffness": {"a": 20.0, "b": 30.0},
            "damping": {"a": 1.0, "b": 3.0},
        },
    )


def test_optimizer_requires_cmaes_for_actual_runs() -> None:
    """Reject an accidental random fallback when cmaes is unavailable."""

    profile = _profile()
    with patch.dict(sys.modules, {"cmaes": None}):
        with pytest.raises(RuntimeError, match="cmaes package is required"):
            CMAESOptimizer(
                profile,
                num_envs=3,
                cmaes=profile.optimizer,
                parent_parameters={
                    "stiffness": {"a": 20.0, "b": 30.0},
                    "damping": {"a": 1.0, "b": 3.0},
                },
            )


def test_optimizer_sampling_is_seed_deterministic() -> None:
    """Produce repeatable populations while changing them across seeds."""

    first = _optimizer(seed=11).sample_population()
    second = _optimizer(seed=11).sample_population()
    different = _optimizer(seed=12).sample_population()
    assert first == second
    assert first != different
    assert all(row["damping"]["a"] == row["damping"]["b"] for row in first)


def test_optimizer_tell_tracks_best_candidate() -> None:
    """Retain the exact candidate with the smallest reported score."""

    optimizer = _optimizer()
    population = optimizer.sample_population()
    assert optimizer.tell([2.0, 0.5, 1.0]) is False
    assert optimizer.best_params == population[1]
    assert optimizer.best_score == 0.5


def test_optimizer_stops_only_after_best_score_stalls() -> None:
    """Use cross-generation best improvement, not population score spread."""

    optimizer = _optimizer()
    optimizer.sample_population()
    assert optimizer.tell([100.0, 100.1, 100.05]) is False
    for _ in range(_MIN_CONVERGENCE_GENERATIONS - 1):
        optimizer.sample_population()
        assert optimizer.tell([100.0, 100.1, 100.05]) is False
    optimizer.sample_population()
    assert optimizer.tell([100.0, 100.1, 100.05]) is True


def test_optimizer_clips_cma_samples_before_denormalizing() -> None:
    """Bound external CMA samples before physical conversion and feedback."""

    class FakeCMA:
        """Return predetermined out-of-range normalized samples."""

        def __init__(self, samples):
            """Store ordered samples and an empty feedback capture."""

            self.samples = list(samples)
            self.told = None

        def ask(self):
            """Return the next normalized sample."""

            return self.samples.pop(0)

        def tell(self, solutions):
            """Capture the bounded solutions reported by the adapter."""

            self.told = solutions

    optimizer = _optimizer(num_envs=2)
    dimension = len(optimizer.param_names)
    fake_cma = FakeCMA(
        [
            np.linspace(-1.5, 1.5, dimension),
            np.linspace(1.5, -1.5, dimension),
        ]
    )
    optimizer._cma = fake_cma
    optimizer.sample_matrix()
    assert np.all(optimizer._last_norm >= -1.0)
    assert np.all(optimizer._last_norm <= 1.0)
    assert np.all(optimizer._last_matrix >= optimizer._lower)
    assert np.all(optimizer._last_matrix <= optimizer._upper)
    optimizer.tell([0.25, 1.0])
    assert np.all(np.asarray(fake_cma.told[0][0]) >= -1.0)
    assert np.all(np.asarray(fake_cma.told[0][0]) <= 1.0)
