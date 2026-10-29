# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real CUDA coverage for the port and six-case CLI worker boundary."""

import json
from pathlib import Path

import numpy as np
import pytest
import warp as wp
import yaml

from anchor_bench.cli import main
from anchor_bench.schema.mpm_sys_id import FIXED, MpmSysIdConfig
from anchor_bench.tasks.mpm_sys_id.simulations.arrays import require_array
from anchor_bench.tasks.mpm_sys_id.simulations.bottle_drop import (
    Simulation as BottleSimulation,
)
from anchor_bench.tasks.mpm_sys_id.simulations.material import DEFAULTS
from anchor_bench.tasks.mpm_sys_id.simulations.registry import create_simulation
from anchor_bench.tasks.mpm_sys_id.simulations.sand_pour import (
    Simulation as PourSimulation,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not wp.is_cuda_available(), reason="Requires CUDA"),
]
EXAMPLE = Path(__file__).parents[3] / "examples/mpm_sys_id/sand.yaml"


def test_six_experiments_in_isolated_cli_worker(tmp_path):
    data = yaml.safe_load(EXAMPLE.read_text())
    data["optimizer"]["stopping"] = {"mode": "bounded", "max_evaluations": 1}
    for e in data["experiments"]:
        e["input_data"]["source"] = str(EXAMPLE.parent / e["input_data"]["source"])
        # Short runs are deliberately invalid: verify missing settling does not
        # create a best fit while all six physics scenes actually execute.
        e["simulation"]["num_frames"] = 2
        if e["simulation"]["type"] == "bottle_drop":
            e["simulation"].update(
                voxel_size=0.02, cap_radius=0.02, bed_width=0.4, bed_length=0.4
            )
    config = tmp_path / "six.yaml"
    config.write_text(yaml.safe_dump(data))
    output = tmp_path / "run"
    assert main(["mpm_sys_id", "--config", str(config), "--output", str(output)]) == 0
    record = json.loads((output / "trials/00000001/result.json").read_text())
    assert len(record["experiments"]) == 6
    assert record["valid"] is False and record["error"] is None
    assert not (output / "best.json").exists()
    assert all(record["material"][k] == v for k, v in FIXED.items())
    for result in record["experiments"].values():
        assert "rms_speed_m_s" in result["measurement"]
        assert np.isfinite(result["measurement"]["rms_speed_m_s"])
    assert (output / ".completed").exists()


@pytest.mark.parametrize("kind", ["sand_pour", "bottle_drop"])
def test_cuda_graph_matches_eager(kind):
    raw = yaml.safe_load(EXAMPLE.read_text())
    config = MpmSysIdConfig.model_validate(raw)
    settings = next(
        e.simulation.model_dump(mode="json")
        for e in config.experiments
        if e.simulation.type == kind
    )
    if kind == "bottle_drop":
        settings.update(
            bed_width=0.4,
            bed_length=0.4,
            voxel_size=0.02,
            cap_radius=0.02,
            pre_settle_seconds=0.05,
            impact_settle_seconds=1.0,
            lift_speed=1.0,
        )
    results = []
    with wp.ScopedDevice("cuda:0"):
        for graphs in (False, True):
            simulation = create_simulation(settings, {**DEFAULTS, **FIXED}, graphs)
            for _ in range(180 if kind == "bottle_drop" else 12):
                simulation.step()
            simulation.test_final()
            assert isinstance(simulation, (BottleSimulation, PourSimulation))
            positions = require_array(simulation.state_0.particle_q).numpy()
            if isinstance(simulation, BottleSimulation):
                assert simulation.phase == "settling crater"
                assert len(simulation.graphs) == (4 if graphs else 0)
                results.append(
                    (positions, require_array(simulation.state_0.body_q).numpy())
                )
            else:
                assert len(simulation.graphs) == (2 if graphs else 0)
                results.append((positions,))
            del simulation
    for eager, captured in zip(results[0], results[1]):
        # Preserve the source regression's state-based tolerance. Atomic contact
        # reductions perturb trajectories; crater bin thresholds amplify tiny
        # state differences and are not suitable for near-bitwise comparison.
        np.testing.assert_allclose(
            captured, eager, atol=0.001 if kind == "bottle_drop" else 1e-5, rtol=0
        )


def test_sigterm_preserves_resume_checkpoint(tmp_path):
    import signal
    import subprocess
    import sys
    import time

    from filelock import FileLock

    data = yaml.safe_load(EXAMPLE.read_text())
    data["optimizer"]["stopping"] = {"mode": "continuous"}
    for index, experiment in enumerate(data["experiments"]):
        experiment["enabled"] = index == 0
        experiment["input_data"]["source"] = str(
            EXAMPLE.parent / experiment["input_data"]["source"]
        )
    config = tmp_path / "continuous.yaml"
    config.write_text(yaml.safe_dump(data))
    output = tmp_path / "continuous"
    with (tmp_path / "cli.log").open("w") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "anchor_bench",
                "mpm_sys_id",
                "--config",
                str(config),
                "--output",
                str(output),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 60
            while not (output / "trials/00000001/worker.log").exists():
                if process.poll() is not None or time.monotonic() > deadline:
                    pytest.fail((tmp_path / "cli.log").read_text())
                time.sleep(0.1)
            process.send_signal(signal.SIGTERM)
            assert process.wait(timeout=15) == 130
            assert not (output / ".completed").exists()
            assert (
                json.loads((output / "status.json").read_text())["state"]
                == "interrupted"
            )
            assert (
                json.loads((output / "checkpoint.json").read_text())["next_evaluation"]
                == 1
            )
            with (
                FileLock(str(output / ".run.lock"), timeout=0),
                FileLock(str(output / ".worker.lock"), timeout=0),
            ):
                assert (output / "inputs.json").exists()
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=15)
