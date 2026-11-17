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
                if not all(math.isfinite(r) for r in residuals):
                    raise ValueError("Nonfinite relative residual")
                # Scale before taking the norm: even hypot can exceed float range
                # when the RMS itself is representable.
                scale = max(abs(r) for r in residuals)
                error = (
                    scale
                    * (
                        math.hypot(*(r / scale for r in residuals))
                        / math.sqrt(len(residuals))
                    )
                    if scale
                    else 0.0
                )
                if not math.isfinite(error):
                    raise ValueError("Nonfinite experiment score")
                entry.update(residuals=residuals, error=error)
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                entry.update(valid=False, failure_reason=str(exc))
        results[experiment["id"]] = entry
    valid = bool(results) and all(e["valid"] for e in results.values())
    error = None
    failure_reason = None
    if valid:
        try:
            weights = [float(e["weight"]) for e in experiments]
            if any(not math.isfinite(w) or w <= 0 for w in weights):
                raise ValueError("Weights must be finite and positive")
            weight_scale = max(weights)
            weights = [w / weight_scale for w in weights]
            errors = [results[e["id"]]["error"] for e in experiments]
            scale = max(errors)
            error = (
                scale
                * math.sqrt(
                    math.fsum(w * (e / scale) ** 2 for w, e in zip(weights, errors))
                    / math.fsum(weights)
                )
                if scale
                else 0.0
            )
            if not math.isfinite(error):
                raise ValueError("Nonfinite combined score")
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            valid, error, failure_reason = False, None, str(exc)
    result = {"valid": valid, "error": error, "experiments": results}
    if failure_reason is not None:
        result["failure_reason"] = failure_reason
    return result
