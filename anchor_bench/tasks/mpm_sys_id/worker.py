# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One candidate per process; no OpenGL or optimizer imports in the worker."""

import argparse
import gc
import hashlib
from pathlib import Path

from filelock import FileLock

from anchor_bench.schema.mpm_sys_id import MpmSysIdConfig

from .artifacts import read_json, write_json
from .simulations.registry import create_simulation


def evaluate(request: dict) -> dict:
    import warp as wp

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
    wp.init()
    wp.set_device(config.execution.device)
    if not wp.get_device().is_cuda:
        raise RuntimeError("MPM fitting requires CUDA")
    results = {}
    for experiment in config.experiments:
        if not experiment.enabled:
            continue
        simulation = None
        try:
            settings = experiment.simulation.model_dump(mode="json")
            simulation = create_simulation(
                settings, material, config.execution.cuda_graphs
            )
            for _ in range(experiment.simulation.num_frames):
                simulation.step()
            measurement = simulation.measure()
            simulation.test_final()
            if experiment.simulation.type == "sand_pour":
                measurement["valid"] = (
                    experiment.simulation.num_frames / experiment.simulation.fps
                    >= 8.0 - 1e-8
                    and measurement["retained_fraction"] <= 0.001
                    and measurement["rms_speed_m_s"] < 0.03
                )
            if not measurement.get("valid"):
                measurement["failure_reason"] = (
                    "Experiment did not reach a valid settled measurement"
                )
            results[experiment.id] = measurement
        except (ValueError, RuntimeError, AssertionError) as exc:
            results[experiment.id] = {
                "valid": False,
                "failure_reason": f"{type(exc).__name__}: {exc}",
            }
        finally:
            del simulation
            gc.collect()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("response", type=Path)
    args = parser.parse_args()
    digest = hashlib.sha256(args.request.read_bytes()).hexdigest()
    with FileLock(str(args.request.parent.parent.parent / ".worker.lock"), timeout=0):
        measurements = evaluate(read_json(args.request))
        write_json(
            args.response, {"request_sha256": digest, "measurements": measurements}
        )


if __name__ == "__main__":
    main()
