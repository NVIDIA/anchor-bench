# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One candidate per process; no OpenGL or optimizer imports in the worker."""

import argparse
import gc
import hashlib
import math
from numbers import Real
from pathlib import Path

from filelock import FileLock

from anchor_bench.schema.mpm_sys_id import MpmSysIdConfig

from .artifacts import read_json, write_json
from .simulations.registry import create_simulation


class WorkerFailure(RuntimeError):
    def __init__(self, category: str, message: str):
        super().__init__(message)
        self.category = category


def validate_device(device: str) -> None:
    """Resolve the requested CUDA device before proposing any candidates."""
    import warp as wp

    wp.init()
    wp.set_device(device)
    if not wp.get_device().is_cuda:
        raise RuntimeError("MPM fitting requires CUDA")


def _require_finite_measurements(value, path="measurement") -> None:
    """Reject non-finite numbers before they enter strict JSON artifacts."""
    if isinstance(value, dict):
        for key, item in value.items():
            _require_finite_measurements(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _require_finite_measurements(item, f"{path}[{index}]")
    elif isinstance(value, Real) and not math.isfinite(value):
        raise ValueError(f"Non-finite measurement: {path}")


def evaluate(request: dict, progress=None) -> dict:
    try:
        config = MpmSysIdConfig.model_validate(request["config"])
        material = request["material"]
        if set(material) != set(config.material.fixed) | set(config.material.optimized):
            raise ValueError("Candidate material keys do not match configuration")
        for key, value in config.material.fixed.items():
            if material[key] != value:
                raise ValueError(f"Fixed parameter changed: {key}")
        for key, bound in config.material.optimized.items():
            if not bound.lower <= material[key] <= bound.upper:
                raise ValueError(f"Parameter outside bounds: {key}")
    except Exception as exc:
        raise WorkerFailure("protocol", str(exc)) from exc
    try:
        validate_device(config.execution.device)
    except Exception as exc:
        raise WorkerFailure("initialization", str(exc)) from exc
    results = {}
    for experiment in config.experiments:
        if not experiment.enabled:
            continue
        simulation = None
        try:
            if progress is not None:
                progress("initialization")
            try:
                settings = experiment.simulation.model_dump(mode="json")
                simulation = create_simulation(
                    settings, material, config.execution.cuda_graphs
                )
            except Exception as exc:
                raise WorkerFailure("initialization", str(exc)) from exc
            if progress is not None:
                progress("candidate")
            for _ in range(experiment.simulation.num_frames):
                simulation.step()
            measurement = simulation.measure()
            _require_finite_measurements(measurement)
            simulation.test_final()
            if experiment.simulation.type == "sand_pour":
                measurement["valid"] = (
                    experiment.simulation.num_frames / experiment.simulation.fps
                    >= 8.0 - 1e-8
                    and measurement["retained_fraction"] <= 0.001
                    and measurement["rms_speed_m_s"] < 0.03
                )
            if not measurement.get("valid"):
                measurement["failure_category"] = "unsettled"
                measurement["failure_reason"] = (
                    "Experiment did not reach a valid settled measurement"
                )
            results[experiment.id] = measurement
        except WorkerFailure:
            raise
        except (ValueError, RuntimeError, AssertionError) as exc:
            results[experiment.id] = {
                "valid": False,
                "failure_category": "numerical",
                "failure_reason": f"{type(exc).__name__}: {exc}",
            }
        finally:
            del simulation
            gc.collect()
    if progress is not None:
        progress("protocol")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("response", type=Path)
    args = parser.parse_args()
    digest = hashlib.sha256(args.request.read_bytes()).hexdigest()
    with FileLock(str(args.request.parent.parent.parent / ".worker.lock"), timeout=0):

        def progress(stage):
            write_json(
                args.response.with_name("worker_state.json"),
                {"request_sha256": digest, "stage": stage},
            )

        try:
            try:
                request = read_json(args.request)
            except Exception as exc:
                raise WorkerFailure("protocol", str(exc)) from exc
            measurements = evaluate(request, progress)
            response = {"request_sha256": digest, "measurements": measurements}
        except Exception as exc:
            response = {
                "request_sha256": digest,
                "failure": {
                    "category": exc.category
                    if isinstance(exc, WorkerFailure)
                    else "protocol",
                    "message": f"{type(exc).__name__}: {exc}",
                },
            }
            write_json(args.response, response)
            raise SystemExit(1) from exc
        write_json(args.response, response)


if __name__ == "__main__":
    main()
