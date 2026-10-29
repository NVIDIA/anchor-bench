# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Relative RMS with equal dimensional weighting within each experiment."""

import math


def score(experiments: list[dict], measurements: dict) -> dict:
    results = {}
    for experiment in experiments:
        measurement = measurements.get(experiment["id"], {})
        entry = {
            "measurement": measurement,
            "valid": bool(measurement.get("valid", False)),
            "error": None,
        }
        if entry["valid"]:
            try:
                values = [float(measurement[k]) for k in experiment["dimensions"]]
                if any(not math.isfinite(v) or v < 0 for v in values):
                    raise ValueError("Nonfinite or negative prediction")
                residuals = [(v - t) / t for v, t in zip(values, experiment["target"])]
                entry.update(
                    residuals=residuals,
                    error=math.sqrt(sum(r * r for r in residuals) / len(residuals)),
                )
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                entry.update(valid=False, failure_reason=str(exc))
        results[experiment["id"]] = entry
    valid = bool(results) and all(e["valid"] for e in results.values())
    error = (
        math.sqrt(
            sum(e["weight"] * results[e["id"]]["error"] ** 2 for e in experiments)
            / sum(e["weight"] for e in experiments)
        )
        if valid
        else None
    )
    return {"valid": valid, "error": error, "experiments": results}
