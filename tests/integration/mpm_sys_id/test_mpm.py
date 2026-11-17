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


def _simulation_settings(kind, lift_distance):
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
            lift_distance=lift_distance,
        )
    return settings


@pytest.mark.parametrize(
    ("kind", "lift_distance"),
    [("sand_pour", None), ("bottle_drop", 0.4), ("bottle_drop", 1e-6)],
)
def test_cuda_graph_and_eager_complete_physical_scenarios(kind, lift_distance):
    settings = _simulation_settings(kind, lift_distance)
    results = []
    with wp.ScopedDevice("cuda:0"):
        for graphs in (False, True):
            simulation = create_simulation(settings, {**DEFAULTS, **FIXED}, graphs)
            host_min_cap_z = float("inf")
            for _ in range(180 if kind == "bottle_drop" else 12):
                simulation.step()
                if isinstance(simulation, BottleSimulation):
                    host_min_cap_z = min(
                        host_min_cap_z,
                        float(require_array(simulation.state_0.body_q).numpy()[0, 2]),
                    )
            if isinstance(simulation, BottleSimulation):
                assert simulation.measure()["min_cap_z_m"] == host_min_cap_z
            if isinstance(simulation, BottleSimulation) and lift_distance == 1e-6:
                assert simulation.measure()["valid"] is False
                assert simulation.measure()["bottle_clearance_m"] < 0
                with pytest.raises(ValueError, match="Bottle has not cleared"):
                    simulation.validate_final_state()
            else:
                simulation.validate_final_state()
            assert isinstance(simulation, (BottleSimulation, PourSimulation))
            positions = require_array(simulation.state_0.particle_q).numpy()
            if isinstance(simulation, BottleSimulation):
                assert simulation.phase == "settling crater"
                # A one-substep lift only warms removal; it never captures it.
                expected_graphs = {
                    "settling bed",
                    "falling / impact",
                    "settling crater",
                }
                if lift_distance != 1e-6:
                    expected_graphs.add("removing bottle")
                assert set(simulation.graphs) == (expected_graphs if graphs else set())
                results.append((require_array(simulation.state_0.body_q).numpy(),))
            else:
                assert len(simulation.graphs) == (2 if graphs else 0)
                results.append((positions,))
            del simulation
    for eager, captured in zip(results[0], results[1]):
        # Check the rigid trajectory and uncoupled pour here. For sand in the
        # coupled bottle scene, contact roundoff can amplify at individual
        # particles over 1,440 substeps (especially with an embedded bottle).
        # Compare that graph from identical states in the replay test below.
        np.testing.assert_allclose(
            captured, eager, atol=0.001 if kind == "bottle_drop" else 1e-5, rtol=0
        )


@pytest.mark.parametrize(
    ("frames", "phase", "lift_distance"),
    [
        (1, "settling bed", 0.4),
        (40, "falling / impact", 0.4),
        (64, "removing bottle", 0.4),
        (100, "settling crater", 0.4),
        (64, "settling crater", 1e-6),
    ],
)
def test_bottle_graph_replay_matches_eager_from_same_state(
    frames, phase, lift_distance
):
    """Compare execution modes without accumulating independent contact trajectories."""
    settings = _simulation_settings("bottle_drop", lift_distance)
    with wp.ScopedDevice("cuda:0"):
        simulation = create_simulation(settings, {**DEFAULTS, **FIXED}, True)
        assert isinstance(simulation, BottleSimulation)
        for _ in range(frames):
            simulation.step()
        assert simulation.phase == phase
        graph = simulation.graphs[phase]
        # Include material history, velocities, and the previous contact force.
        # Newton adds arrays such as collider_ids after stepping, so a fresh
        # model.state().assign(...) cannot snapshot this evolved state.
        arrays = [
            array
            for owner in (
                simulation.state_0,
                vars(simulation.state_0)["mpm"],
                simulation.rigid_state,
                vars(simulation.rigid_state)["mpm"],
            )
            for array in vars(owner).values()
            if isinstance(array, wp.array)
        ]
        tolerances = {
            "particle_q": 1e-5,
            "particle_qd": 1e-4,
            "body_q": 1e-5,
            "body_qd": 1e-4,
            "body_f": 1e-3,
        }
        for substep in range(settings["substeps"]):
            snapshot = [(array, wp.clone(array)) for array in arrays]
            captured = {}
            for use_graph in (True, False):
                for array, saved in snapshot:
                    array.assign(saved)
                # Compare each replay from identical evolving inputs. Letting two
                # contact trajectories diverge across substeps tests chaotic
                # amplification of reduction roundoff, not graph replay parity.
                simulation.solver.reset(simulation.state_0, flags=0)
                if use_graph:
                    wp.capture_launch(graph)
                else:
                    simulation._substep()
                values = {
                    name: require_array(getattr(simulation.state_0, name)).numpy()
                    for name in tolerances
                }
                if use_graph:
                    captured = values
                else:
                    for name, atol in tolerances.items():
                        np.testing.assert_allclose(
                            values[name],
                            captured[name],
                            atol=atol,
                            rtol=0,
                            err_msg=f"{phase}, substep {substep}: {name}",
                        )
                simulation.solver.check_sparse_grid_rebuild_status()


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
