# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration, measurement, search, and durable run lifecycle coverage."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from filelock import FileLock
from pydantic import ValidationError

from anchor_bench.cli import main
from anchor_bench.core.task_runner import TaskRunner
from anchor_bench.errors import AnchorBenchError, DatasetLoadError
from anchor_bench.schema.mpm_sys_id import FIXED, MpmSysIdConfig
from anchor_bench.schema.parse import parse
from anchor_bench.tasks.mpm_sys_id.artifacts import read_json, write_json
from anchor_bench.tasks.mpm_sys_id.dataset import load_experiments
from anchor_bench.tasks.mpm_sys_id.objective import score
from anchor_bench.tasks.mpm_sys_id.optimizer import decode, propose
from anchor_bench.tasks.mpm_sys_id.task import MpmSysIdTask

EXAMPLE = Path(__file__).parents[1] / "examples/mpm_sys_id/sand.yaml"


@pytest.fixture
def config_path(tmp_path):
    data = yaml.safe_load(EXAMPLE.read_text())
    for index, e in enumerate(data["experiments"]):
        e["enabled"] = index == 0
        e["input_data"]["source"] = "measurements.csv" if index == 0 else "missing.csv"
    data["optimizer"]["stopping"] = {"mode": "bounded", "max_evaluations": 2}
    (tmp_path / "measurements.csv").write_text(
        "width_m,length_m,height_m\n1,2,0.5\n2,4,1\n"
    )
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_csv_selection_and_validation(config_path):
    config = parse(config_path)
    assert isinstance(config, MpmSysIdConfig)
    cases = load_experiments(config)
    assert len(cases) == 1
    assert cases[0]["target"] == [1.5, 3, 0.75]
    assert cases[0]["sample_std_m"][0] == pytest.approx(2**-0.5)
    assert Path(cases[0]["input_data"]["source"]).is_absolute()
    for content in (
        "width_m,length_m,depth_m\n1,2,3\n",
        "width_m,length_m,height_m\n1,nan,3\n",
        "width_m,length_m,height_m\n1,2,\n",
        "width_m,length_m,height_m\n1,2,3,4\n",
        "width_m,length_m,height_m\n",
    ):
        (config_path.parent / "measurements.csv").write_text(content)
        with pytest.raises(DatasetLoadError):
            load_experiments(config)


def test_fixed_parameters_and_experiment_schema(config_path):
    raw = yaml.safe_load(config_path.read_text())
    for name in FIXED:
        modified = copy.deepcopy(raw)
        modified["material"]["fixed"][name] += 0.01
        with pytest.raises(ValidationError):
            MpmSysIdConfig.model_validate(modified)
    modified = copy.deepcopy(raw)
    for e in modified["experiments"]:
        e["enabled"] = False
    with pytest.raises(ValidationError):
        MpmSysIdConfig.model_validate(modified)
    modified = copy.deepcopy(raw)
    modified["experiments"][1]["id"] = modified["experiments"][0]["id"]
    with pytest.raises(ValidationError):
        MpmSysIdConfig.model_validate(modified)
    modified = copy.deepcopy(raw)
    modified["experiments"][0]["simulation"]["friction"] = 0.1
    with pytest.raises(ValidationError):
        MpmSysIdConfig.model_validate(modified)
    modified = copy.deepcopy(raw)
    modified["material"]["optimized"]["young_modulus"] = {"lower": 1.0, "upper": 2.0}
    with pytest.raises(ValidationError):
        MpmSysIdConfig.model_validate(modified)


def test_objective_requires_all_enabled_experiments():
    cases = [
        {"id": "a", "dimensions": ["width_m"], "target": [2.0], "weight": 1},
        {"id": "b", "dimensions": ["depth_m"], "target": [1.0], "weight": 3},
    ]
    result = score(
        cases,
        {"a": {"valid": True, "width_m": 2.4}, "b": {"valid": True, "depth_m": 1.0}},
    )
    assert result["error"] == pytest.approx(0.1)
    assert not score(cases, {"a": {"valid": True, "width_m": 2.4}})["valid"]
    assert not score(
        cases,
        {
            "a": {"valid": True, "width_m": float("nan")},
            "b": {"valid": True, "depth_m": 1},
        },
    )["valid"]


def test_forest_replays_and_preserves_parameter_bounds(config_path):
    pytest.importorskip("sklearn", reason="Install the mpm extra: uv sync --extra mpm")
    config = parse(config_path)
    assert isinstance(config, MpmSysIdConfig)
    bounds = {
        k: [b.lower, b.upper, b.scale, b.offset]
        for k, b in config.material.optimized.items()
    }
    history = []
    for _ in range(20):
        point, strategy = propose(
            history, bounds, seed=48, initial_samples=3, pool_size=32
        )
        assert propose(
            json.loads(json.dumps(history)),
            bounds,
            seed=48,
            initial_samples=3,
            pool_size=32,
        ) == (point, strategy)
        material = {**decode(point, bounds), **config.material.fixed}
        assert all(material[k] == v for k, v in FIXED.items())
        history.append(
            {
                "point": point,
                "valid": True,
                "error": sum((v - 0.25) ** 2 for v in point),
            }
        )
    assert any(h["error"] < history[0]["error"] for h in history[1:])
    assert decode([0] * 5, bounds)["damping"] == 0


def test_run_resume_lock_and_data_fingerprint(config_path, tmp_path, monkeypatch):
    pytest.importorskip("sklearn", reason="Install the mpm extra: uv sync --extra mpm")
    output = tmp_path / "run"
    calls = []

    def evaluate(self, request, directory):
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / "request.json", request)
        calls.append(request)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return {
            e["id"]: {"valid": True, **dict(zip(e["dimensions"], e["target"]))}
            for e in self.experiments
        }

    monkeypatch.setattr(MpmSysIdTask, "_evaluate", evaluate)
    with pytest.raises(KeyboardInterrupt):
        TaskRunner().run(str(config_path), output=output)
    assert not (output / ".completed").exists()
    assert read_json(output / "status.json")["state"] == "interrupted"
    assert read_json(output / "checkpoint.json")["next_evaluation"] == 2
    with pytest.raises(AnchorBenchError, match="--resume"):
        TaskRunner().run(str(config_path), output=output)
    with (
        FileLock(str(output / ".run.lock")),
        pytest.raises(AnchorBenchError, match="Another MPM"),
    ):
        TaskRunner().run(str(config_path), output=output, resume=True)
    updated = yaml.safe_load(config_path.read_text())
    updated["description"] = "Updated run label; physics is unchanged"
    config_path.write_text(yaml.safe_dump(updated))
    # A damaged derived index is reconstructed from atomic trial result records.
    (output / "history.jsonl").write_text("{truncated")
    assert (
        main(
            [
                "mpm_sys_id",
                "--config",
                str(config_path),
                "--output",
                str(output),
                "--resume",
            ]
        )
        == 0
    )
    assert calls[1]["material"] == calls[2]["material"]
    assert (output / ".completed").exists()
    assert len((output / "history.jsonl").read_text().splitlines()) == 2
    assert read_json(output / "best.json")["error"] == 0
    assert TaskRunner().run(str(config_path), output=output) == output
    assert len(calls) == 3
    original = (output / "config.yaml").read_text()
    (config_path.parent / "measurements.csv").write_text(
        "width_m,length_m,height_m\n1,2,3\n"
    )
    with pytest.raises(AnchorBenchError, match="inputs"):
        TaskRunner().run(str(config_path), output=output)
    assert (output / "config.yaml").read_text() == original


def test_cli_rejects_wrong_task_type(config_path, tmp_path):
    with pytest.raises(AnchorBenchError, match="Expected task"):
        TaskRunner().run(
            str(config_path), expected_type="training", output=tmp_path / "unused"
        )
    assert not (tmp_path / "unused").exists()


def test_measurement_geometry():
    from anchor_bench.tasks.mpm_sys_id.simulations.bottle_drop import (
        bottle_mesh,
        measure_crater,
    )
    from anchor_bench.tasks.mpm_sys_id.simulations.sand_pour import measure_pile

    reference = np.full((40, 40), 0.12)
    current = reference.copy()
    current[15:25, 16:24] -= 0.04
    metrics = measure_crater(reference, current, cell_size=0.01, threshold=0.02)
    np.testing.assert_allclose(
        [metrics["width_m"], metrics["length_m"], metrics["depth_m"]], [0.1, 0.08, 0.04]
    )
    vertices, indices, com, inertia = bottle_mesh(0.04, 0.25, 0.015, 0.015)
    a, b, c = (vertices[indices.reshape(-1, 3)[:, i]] for i in range(3))
    assert np.sum(a * np.cross(b, c)) / 6 > 0
    assert 0 < com < 0.25 and min(inertia) > 0
    pile = measure_pile(
        np.array([[-1.0, -2.0, 0], [1, 2, 0.4]]),
        np.zeros((2, 3)),
        np.full(2, 0.01),
        outlet_height=1,
    )
    np.testing.assert_allclose(
        [pile["width_m"], pile["length_m"], pile["height_m"]], [2.0, 3.98, 0.408]
    )


def test_completed_worker_response_is_reused(config_path, tmp_path):
    import hashlib

    config = parse(config_path)
    assert isinstance(config, MpmSysIdConfig)
    task = MpmSysIdTask(config)
    directory = tmp_path / "cached"
    directory.mkdir()
    request = {"material": config.material.fixed, "schema_version": 1}
    write_json(directory / "request.json", request)
    digest = hashlib.sha256((directory / "request.json").read_bytes()).hexdigest()
    expected = {"pour": {"valid": False, "failure_reason": "not settled"}}
    write_json(
        directory / "response.json",
        {"request_sha256": digest, "measurements": expected},
    )
    assert task._evaluate(dict(reversed(list(request.items()))), directory) == expected
    write_json(
        directory / "response.json",
        {"request_sha256": "wrong", "measurements": expected},
    )
    with pytest.raises(AnchorBenchError, match="does not match"):
        task._evaluate(request, directory)


def test_mpm_tracking_records_csv_and_scores(config_path, tmp_path, monkeypatch):
    pytest.importorskip("sklearn", reason="Install the mpm extra: uv sync --extra mpm")
    import mlflow
    from mlflow.tracking import MlflowClient

    data = yaml.safe_load(config_path.read_text())
    uri = f"sqlite:///{tmp_path / 'tracking.db'}"
    data["mlops"] = {
        "backend": "mlflow",
        "experiment_name": "mpm-test",
        "upload_dataset": True,
        "options": {"tracking_uri": uri},
    }
    config_path.write_text(yaml.safe_dump(data))

    def evaluate(self, request, directory):
        directory.mkdir(parents=True, exist_ok=True)
        return {
            e["id"]: {"valid": True, **dict(zip(e["dimensions"], e["target"]))}
            for e in self.experiments
        }

    monkeypatch.setattr(MpmSysIdTask, "_evaluate", evaluate)
    previous_uri = mlflow.get_tracking_uri()
    output = tmp_path / "tracked"
    TaskRunner().run(str(config_path), output=output)
    assert mlflow.get_tracking_uri() == previous_uri
    client = MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name("mpm-test")
    assert experiment is not None
    run = client.search_runs([experiment.experiment_id])[0]
    assert run.info.status == "FINISHED"
    assert run.data.metrics["optimization/best_relative_rms"] == 0
    assert run.data.metrics["material/best/young_modulus"] == 9576000.0
    assert len(run.inputs.dataset_inputs) == 1
    assert run.data.params["datasets/1m-5gal-sand-pour/uploaded"] == "True"


def test_pile_excludes_retained_particles_but_keeps_global_diagnostics():
    from anchor_bench.tasks.mpm_sys_id.simulations.sand_pour import measure_pile

    positions = np.tile([0.0, 0.0, 0.09], (2000, 1))
    # Include the outlet boundary, and keep retention within the valid-run limit.
    positions[-2:] = [[10, 20, 1.0], [-10, -20, 1.2]]
    velocities = np.zeros_like(positions)
    velocities[-1, 2] = 0.5
    radii = np.full(len(positions), 0.01)
    radii[-2:] = 0.1
    pile = measure_pile(positions, velocities, radii, outlet_height=1.0)

    assert pile["width_m"] == pytest.approx(0.02)
    assert pile["length_m"] == pytest.approx(0.02)
    assert pile["height_m"] == pytest.approx(0.1)
    assert pile["retained_fraction"] == pytest.approx(0.001)
    assert pile["rms_speed_m_s"] == pytest.approx(0.5 / np.sqrt(2000))


def test_pile_quantiles_resist_airborne_and_scattered_particles():
    from anchor_bench.tasks.mpm_sys_id.simulations.sand_pour import measure_pile

    positions = np.tile([0.0, 0.0, 0.09], (2000, 1))
    positions[-2:] = [[10, 20, 0.9], [-10, -20, 0.09]]
    velocities = np.zeros_like(positions)
    radii = np.full(len(positions), 0.01)
    pile = measure_pile(positions, velocities, radii, outlet_height=1.0)

    assert pile["width_m"] == pytest.approx(0.02)
    assert pile["length_m"] == pytest.approx(0.02)
    assert pile["height_m"] == pytest.approx(0.1)
    full = measure_pile(
        positions, velocities, radii, outlet_height=1.0, footprint_fraction=1.0
    )
    assert full["width_m"] == pytest.approx(20.02)
    assert full["length_m"] == pytest.approx(40.02)


def test_pile_without_particles_below_outlet_is_invalid():
    from anchor_bench.tasks.mpm_sys_id.simulations.sand_pour import measure_pile

    pile = measure_pile(
        np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.2]]),
        np.zeros((2, 3)),
        np.full(2, 0.01),
        outlet_height=1.0,
    )

    assert pile["valid"] is False
    assert pile["failure_reason"] == "No particles below the funnel outlet"
    assert pile["retained_fraction"] == 1.0
    assert pile["rms_speed_m_s"] == 0.0
    assert [pile[key] for key in ("width_m", "length_m", "height_m")] == [0, 0, 0]


@pytest.mark.parametrize("field", ["height_m", "rms_speed_m_s", "nested"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_worker_rejects_nonfinite_measurements(config_path, monkeypatch, field, value):
    from types import SimpleNamespace
    from unittest.mock import Mock

    import warp as wp

    from anchor_bench.tasks.mpm_sys_id import worker

    config = parse(config_path)
    assert isinstance(config, MpmSysIdConfig)
    measurement = {"valid": True, field: [value] if field == "nested" else value}
    simulation = Mock()
    simulation.measure.return_value = measurement
    monkeypatch.setattr(worker, "create_simulation", lambda *args: simulation)
    monkeypatch.setattr(wp, "init", lambda: None)
    monkeypatch.setattr(wp, "set_device", lambda device: None)
    monkeypatch.setattr(wp, "get_device", lambda: SimpleNamespace(is_cuda=True))
    results = worker.evaluate(
        {
            "config": config.model_dump(mode="json"),
            "material": {
                **config.material.fixed,
                **{k: b.lower for k, b in config.material.optimized.items()},
            },
        }
    )
    assert len(results) == 1
    result = next(iter(results.values()))
    assert result["valid"] is False
    assert f"measurement.{field}" in result["failure_reason"]
    json.dumps(results, allow_nan=False)


@pytest.mark.parametrize("failure", ["exit", "timeout", "interrupt"])
def test_worker_failure_persistence(config_path, tmp_path, monkeypatch, failure):
    import subprocess
    from unittest.mock import Mock

    config = parse(config_path)
    assert isinstance(config, MpmSysIdConfig)
    task = MpmSysIdTask(config)
    task.experiments = [{"id": "pour"}]
    process = Mock()
    if failure == "exit":
        process.wait.return_value = -11
    elif failure == "timeout":
        process.wait.side_effect = [subprocess.TimeoutExpired("worker", 1), -9]
    else:
        process.wait.side_effect = [KeyboardInterrupt, 0]
        process.poll.return_value = None
    launch = Mock(return_value=process)
    monkeypatch.setattr("anchor_bench.tasks.mpm_sys_id.task.subprocess.Popen", launch)
    directory = tmp_path / "trial"
    request = {"evaluation": 1}
    if failure == "interrupt":
        with pytest.raises(KeyboardInterrupt):
            task._evaluate(request, directory)
        process.terminate.assert_called_once()
        assert not (directory / "response.json").exists()
        return
    results = task._evaluate(request, directory)
    assert results["pour"]["valid"] is False
    reason = results["pour"]["failure_reason"]
    assert ("exited -11" if failure == "exit" else "timed out") in reason
    assert "worker.log" in reason
    if failure == "timeout":
        process.kill.assert_called_once()
    assert read_json(directory / "response.json")["measurements"] == results
    assert task._evaluate(request, directory) == results
    launch.assert_called_once()


def test_resume_after_worker_crash_advances_candidate(
    config_path, tmp_path, monkeypatch
):
    from unittest.mock import Mock

    from anchor_bench.tasks.mpm_sys_id import task as task_module

    output = tmp_path / "run"
    process = Mock()
    process.wait.return_value = -11
    launch = Mock(return_value=process)
    monkeypatch.setattr(task_module.subprocess, "Popen", launch)
    original_write = task_module.write_json

    def interrupt_before_trial_record(path, value):
        if path.name == "result.json":
            raise KeyboardInterrupt
        original_write(path, value)

    monkeypatch.setattr(task_module, "write_json", interrupt_before_trial_record)
    with pytest.raises(KeyboardInterrupt):
        TaskRunner().run(str(config_path), output=output)
    first = output / "trials/00000001"
    assert (first / "response.json").exists()
    assert not (first / "result.json").exists()
    monkeypatch.setattr(task_module, "write_json", original_write)
    TaskRunner().run(str(config_path), output=output, resume=True)
    # First crash is reused, so only the second candidate launches on resume.
    assert launch.call_count == 2
    records = [read_json(p) for p in sorted((output / "trials").glob("*/result.json"))]
    assert len(records) == 2
    assert all(not r["valid"] and r["error"] is None for r in records)
    assert records[0]["point"] != records[1]["point"]
    assert (output / ".completed").exists()
    assert not (output / "best.json").exists()


@pytest.mark.parametrize("stage", ["setup", "body"])
@pytest.mark.parametrize("kind", ["timeout", "missing_package"])
def test_output_context_preserves_unrelated_errors_and_releases_lock(
    config_path, tmp_path, monkeypatch, stage, kind
):
    from importlib.metadata import PackageNotFoundError

    from filelock import Timeout

    from anchor_bench.tasks.mpm_sys_id import task as task_module

    config = parse(config_path)
    assert isinstance(config, MpmSysIdConfig)
    task = MpmSysIdTask(config)
    output = tmp_path / "run"
    output.mkdir()
    failure = (
        Timeout("plugin.lock")
        if kind == "timeout"
        else PackageNotFoundError("unrelated-plugin")
    )

    def fail_setup(config):
        raise failure

    if stage == "setup":
        monkeypatch.setattr(task_module, "load_experiments", fail_setup)
    with pytest.raises(type(failure)) as caught, task.output_context(output):
        assert stage == "body"
        # The lock remains held throughout the context body.
        with pytest.raises(Timeout):
            FileLock(str(output / ".run.lock"), timeout=0).acquire()
        raise failure
    assert caught.value is failure
    with FileLock(str(output / ".run.lock"), timeout=0):
        pass


def test_output_context_translates_missing_mpm_dependency_and_releases_lock(
    config_path, tmp_path, monkeypatch
):
    from importlib.metadata import PackageNotFoundError

    from anchor_bench.tasks.mpm_sys_id import task as task_module

    config = parse(config_path)
    assert isinstance(config, MpmSysIdConfig)
    task = MpmSysIdTask(config)
    output = tmp_path / "run"
    output.mkdir()
    failure = PackageNotFoundError("newton")

    def missing_version(name):
        raise failure

    monkeypatch.setattr(task_module.importlib.metadata, "version", missing_version)
    with (
        pytest.raises(AnchorBenchError, match="uv sync --extra mpm") as caught,
        task.output_context(output),
    ):
        pytest.fail("Missing dependency must prevent entry")
    assert caught.value.__cause__ is failure
    with FileLock(str(output / ".run.lock"), timeout=0):
        pass


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("existing", [False, True])
def test_mpm_resume_without_checkpoint_does_not_mutate_output(
    config_path, tmp_path, explicit, existing
):
    from anchor_bench.core.artifacts import resolve_output_folder

    data = yaml.safe_load(config_path.read_text())
    data["output_dir"] = str(tmp_path / "output")
    config_path.write_text(yaml.safe_dump(data))
    config = parse(config_path)
    assert isinstance(config, MpmSysIdConfig)
    requested = tmp_path / "explicit" if explicit else None
    resolved = resolve_output_folder(
        config, output_root=config.output_dir, output_folder=requested
    )
    if existing:
        resolved.mkdir(parents=True)
        (resolved / "config.yaml").write_text("original config")
        (resolved / "sentinel").write_text("keep")
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises(AnchorBenchError, match="Cannot resume: inputs.json"):
        TaskRunner().run(str(config_path), resume=True, output=requested)
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before
    assert resolved.exists() is existing
    if not existing:
        assert not (tmp_path / "output").exists()


def test_mpm_resume_uses_default_output_path(config_path, tmp_path, monkeypatch):
    from anchor_bench.core.artifacts import resolve_output_folder

    data = yaml.safe_load(config_path.read_text())
    data["output_dir"] = str(tmp_path / "output")
    config_path.write_text(yaml.safe_dump(data))
    config = parse(config_path)
    assert isinstance(config, MpmSysIdConfig)
    resolved = resolve_output_folder(config, output_root=config.output_dir)
    calls = 0

    def run(self, output_directory):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise KeyboardInterrupt
        return output_directory

    monkeypatch.setattr(MpmSysIdTask, "run", run)
    with pytest.raises(KeyboardInterrupt):
        TaskRunner().run(str(config_path))
    assert (resolved / "inputs.json").is_file()
    assert TaskRunner().run(str(config_path), resume=True) == resolved
    assert calls == 2
    assert (resolved / ".completed").is_file()


@pytest.mark.parametrize("chunk_size", [1, 3, 256])
def test_chunked_distances_match_dense_reference(chunk_size):
    from anchor_bench.tasks.mpm_sys_id.optimizer import _minimum_squared_distances

    rng = np.random.default_rng(48)
    history = rng.random((17, 5))
    pool = rng.random((11, 5))
    pool[0] = history[-1]
    pool[1] = history[0] + 1e-7
    pool[2] = history[0] + 1e-5
    expected = np.min(
        np.sum((pool[:, None, :] - history[None, :, :]) ** 2, axis=2), axis=1
    )
    actual = _minimum_squared_distances(pool, history, chunk_size=chunk_size)
    np.testing.assert_array_equal(actual, expected)
    assert actual[0] == 0
    np.testing.assert_array_equal(actual > 1e-12, expected > 1e-12)
    assert np.argmax(actual) == np.argmax(expected)


@pytest.mark.parametrize("valid", [False, True])
def test_chunked_proposals_match_dense_reference(config_path, monkeypatch, valid):
    from anchor_bench.tasks.mpm_sys_id import optimizer

    config = parse(config_path)
    assert isinstance(config, MpmSysIdConfig)
    bounds = {
        k: [b.lower, b.upper, b.scale, b.offset]
        for k, b in config.material.optimized.items()
    }
    rng = np.random.default_rng(48)
    history = [
        {"point": p.tolist(), "valid": valid, "error": float(np.sum(p**2))}
        for p in rng.random((257, 5))
    ]
    actual = optimizer.propose(history, bounds, seed=48, pool_size=272)

    def dense_distances(pool, observations):
        return np.min(
            np.sum((pool[:, None, :] - observations[None, :, :]) ** 2, axis=2),
            axis=1,
        )

    monkeypatch.setattr(optimizer, "_minimum_squared_distances", dense_distances)
    assert optimizer.propose(history, bounds, seed=48, pool_size=272) == actual
    assert actual[1] == ("random_forest" if valid else "space_filling")
