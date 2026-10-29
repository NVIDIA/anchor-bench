# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fresh-process installed-plugin discovery through the complete sysID CLI."""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

pytestmark = pytest.mark.integration

REPOSITORY = Path(__file__).resolve().parents[2]
FIXTURE = REPOSITORY / "tests" / "fixtures" / "sysid_backend"
COMMON_OUTPUTS = (
    ".completed",
    "config.yaml",
    "best_params.yaml",
    "optimization_log.csv",
    "run_summary.yaml",
)


@pytest.fixture
def installed_backend(tmp_path: Path) -> Path:
    """Install an importable package and distribution metadata on a private path."""
    installed = tmp_path / "site-packages"
    installed.mkdir()
    shutil.copytree(FIXTURE / "toy_sysid_backend", installed / "toy_sysid_backend")
    distribution = installed / "anchor_bench_toy_sysid-1.0.dist-info"
    distribution.mkdir()
    (distribution / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: anchor-bench-toy-sysid\nVersion: 1.0\n",
        encoding="utf-8",
    )
    return installed


def _write_config(tmp_path: Path) -> Path:
    rows = [
        {
            "time_ns": timestamp,
            "experiment": "toy-wave",
            "field": field,
            "value": value,
        }
        for field, values in (
            ("hip/command_q", [0.0, 1.0, -0.5, 0.75]),
            ("hip/actual_q", [0.0, 0.65, -0.325, 0.4875]),
        )
        for timestamp, value in zip((0, 10_000_000, 20_000_000, 30_000_000), values)
    ]
    parquet = tmp_path / "recording.parquet"
    pd.DataFrame(rows).to_parquet(parquet, index=False)
    usd = tmp_path / "response.usda"
    usd.write_text(
        '#usda 1.0\ndef Xform "response" {\n    custom double gain_scale = 2.0\n}\n',
        encoding="utf-8",
    )
    document = {
        "type": "actuator_sys_id",
        "joint_parameters": [
            {
                "names": ["hip"],
                "controller": {
                    "type": "pd",
                    "clamps": {"effort_limit": 10.0, "velocity_limit": 10.0},
                    "stiffness": {"lower": 0.5, "upper": 3.0},
                    "damping": {"lower": 0.0, "upper": 1.0},
                },
                "parameters": {
                    "dry_friction": {"lower": 0.0, "upper": 0.1},
                    "viscous_friction": {"lower": 0.0, "upper": 0.1},
                    "armature": {"lower": 0.0, "upper": 0.1},
                },
            }
        ],
        "optimizer": {
            "type": "python-cmaes",
            "sigma": 0.4,
            "max_iterations": 3,
            "epsilon": 0.0001,
        },
        "simulation": {
            "type": "toy-sysid",
            "usd_path": str(usd),
            "physics_freq": 200,
            "control_freq": 100,
            "render_freq": 50,
            "gain": 2.5,
            "model_name": "usd-response",
        },
        "num_envs": 6,
        "seed": 17,
        "dataset": {
            "type": "parquet",
            "source": str(parquet),
            "resampling": {"type": "uniform", "dt": 0.01},
        },
    }
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(document), encoding="utf-8")
    return config


@pytest.mark.parametrize("export_json", [True, False], ids=["json-export", "no-export"])
def test_cli_discovers_installed_sysid_backend(
    tmp_path: Path,
    installed_backend: Path,
    export_json: bool,
) -> None:
    provider = "provide_json" if export_json else "provide_no_export"
    distribution = installed_backend / "anchor_bench_toy_sysid-1.0.dist-info"
    (distribution / "entry_points.txt").write_text(
        f"[anchor_bench.sysid_backends]\ntoy-sysid = toy_sysid_backend:{provider}\n",
        encoding="utf-8",
    )
    config = _write_config(tmp_path)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(installed_backend), str(REPOSITORY), environment.get("PYTHONPATH", "")]
    )
    outputs = [tmp_path / "first-run", tmp_path / "repeat-run"]
    for output in outputs:
        result = subprocess.run(
            [
                sys.executable,
                str(FIXTURE / "run_cli.py"),
                "--config",
                str(config),
                "--output",
                str(output),
            ],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert all((output / filename).is_file() for filename in COMMON_OUTPUTS)
        assert not (output / "identified.usda").exists()
        assert (output / "identified.json").exists() is export_json

        resolved = yaml.safe_load((output / "config.yaml").read_text())
        assert resolved["simulation"] == {
            "type": "toy-sysid",
            "usd_path": str(tmp_path / "response.usda"),
            "physics_freq": 200,
            "control_freq": 100,
            "render_freq": 50,
            "gain": 2.5,
            "model_name": "usd-response",
        }
        best = yaml.safe_load((output / "best_params.yaml").read_text())
        summary = yaml.safe_load((output / "run_summary.yaml").read_text())
        assert summary["generations"] == 3
        assert summary["best_mse"] == best["best_mse"]
        assert math.isfinite(best["best_mse"])
        # Mean command square is (1 + 0.25 + 0.5625) / 4 for this recording.
        # The USD scale 2.0 and configured gain 2.5 give an effective gain of 5.0.
        stiffness = best["parameters"]["hip"]["stiffness"]
        assert best["best_mse"] == pytest.approx(
            0.453125 * (stiffness / 5.0 - 0.65) ** 2
        )
        with (output / "optimization_log.csv").open(newline="") as stream:
            history = list(csv.DictReader(stream))
        assert len(history) == 3
        assert [int(row["generation"]) for row in history] == [0, 1, 2]
        assert all(
            math.isfinite(float(row[column]))
            for row in history
            for column in ("best_mse", "population_mean_mse")
        )
        assert float(history[-1]["best_mse"]) == best["best_mse"]
        if export_json:
            model = json.loads((output / "identified.json").read_text())
            assert model["parameters"] == best["parameters"]
            assert model["simulation"] == resolved["simulation"]
            assert model["gain_scale"] == 2.0

    for filename in ("best_params.yaml", "optimization_log.csv"):
        assert (outputs[0] / filename).read_text() == (
            outputs[1] / filename
        ).read_text()
