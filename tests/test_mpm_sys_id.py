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
        [pile["width_m"], pile["length_m"], pile["height_m"]], [2.02, 4.02, 0.41]
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
