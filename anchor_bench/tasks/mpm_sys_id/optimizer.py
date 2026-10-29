# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Random-forest surrogate proposals in normalized material coordinates."""

import numpy as np

try:
    from sklearn.ensemble import RandomForestRegressor
except ImportError:
    RandomForestRegressor = None

from .simulations.material import DEFAULTS


def decode(point, bounds):
    """Convert bounded unit coordinates to SI material values."""
    point = np.asarray(point)
    if (
        point.shape != (len(bounds),)
        or not np.isfinite(point).all()
        or np.any((point < 0) | (point > 1))
    ):
        raise ValueError("Invalid normalized material coordinates")
    result = {}
    for x, (name, (lo, hi, scale, offset)) in zip(point, bounds.items(), strict=True):
        if x == 0 or x == 1:
            result[name] = float(lo if x == 0 else hi)
            continue
        result[name] = float(
            np.clip(
                np.exp(np.log(lo + offset) + x * np.log((hi + offset) / (lo + offset)))
                - offset
                if scale == "log"
                else lo + x * (hi - lo),
                lo,
                hi,
            )
        )
    return result


def initial_point(bounds):
    """Place the known baseline in the search domain, clipping custom intervals."""
    point = []
    for name, (lo, hi, scale, offset) in bounds.items():
        value = np.clip(DEFAULTS[name], lo, hi)
        point.append(
            np.log((value + offset) / (lo + offset))
            / np.log((hi + offset) / (lo + offset))
            if scale == "log"
            else (value - lo) / (hi - lo)
        )
    return point


def propose(
    history, bounds, *, seed=42, initial_samples=16, pool_size=4096, exploration=1.0
):
    """Propose a new material using a forest lower-confidence bound plus random exploration.

    Tree disagreement is a search heuristic, not calibrated physical uncertainty.
    Failed simulations receive a worse training loss than every valid observation.
    Seeded per-evaluation randomness permits exact proposal replay after resuming.
    """
    if initial_samples < 2 or pool_size < 16 or exploration < 0:
        raise ValueError(
            "Require at least two initial samples, 16 candidates, and nonnegative exploration"
        )
    if not history:
        return initial_point(bounds), "baseline"
    rng = np.random.default_rng(np.random.SeedSequence([seed, len(history)]))
    if len(history) < initial_samples and len(history) <= len(bounds):
        # Probe one parameter at a time around the viable baseline before global
        # exploration; random cohesion combinations can otherwise all block pouring.
        point = list(history[0]["point"])
        point[len(history) - 1] = float(rng.uniform(0.1, 0.9))
        return point, "baseline_probe"
    x = np.asarray([trial["point"] for trial in history])
    pool = rng.random((pool_size, len(bounds)))
    valid = [trial for trial in history if trial["valid"]]
    if len(history) < initial_samples or len(valid) < 2 or rng.random() < 0.2:
        # Maximize distance to prior observations during the initial design.
        distance = np.min(
            np.sum((pool[:, None, :] - x[None, :, :]) ** 2, axis=2), axis=1
        )
        return pool[np.argmax(distance)].tolist(), "space_filling"
    if RandomForestRegressor is None:
        raise ImportError("Install the mpm extra: uv sync --extra mpm")

    best = min(valid, key=lambda trial: trial["error"])
    pool[: pool_size // 2] = np.clip(
        rng.normal(best["point"], 0.15, (pool_size // 2, len(bounds))), 0, 1
    )
    pool = pool[
        np.min(np.sum((pool[:, None, :] - x[None, :, :]) ** 2, axis=2), axis=1) > 1e-12
    ]
    if not len(pool):
        raise RuntimeError("No unevaluated candidates remain in the proposal pool")
    losses = [trial["error"] for trial in valid]
    penalty = max(losses) + max(1.0, np.ptp(losses))
    y = np.log1p([trial["error"] if trial["valid"] else penalty for trial in history])
    forest = RandomForestRegressor(
        n_estimators=128, max_features=0.8, random_state=seed + len(history), n_jobs=1
    )
    forest.fit(x, y)
    predictions = np.asarray([tree.predict(pool) for tree in forest.estimators_])
    acquisition = predictions.mean(axis=0) - exploration * predictions.std(axis=0)
    return pool[np.argmin(acquisition)].tolist(), "random_forest"
