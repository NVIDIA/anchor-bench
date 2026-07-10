from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

import yaml


RUNTIME_E2E_ENV = "ANCHOR_BENCH_RUN_RUNTIME_E2E"
RUNTIME_OUTPUT_ENV = "ANCHOR_BENCH_RUNTIME_OUTPUT_DIR"
MOTION_NAME = "runtime_smoke_motion"
DATASET_NAME = "runtime_smoke_dataset"
ACTUATOR_MODEL = "fitted_implicit"
REPO = Path(__file__).resolve().parents[1]


def _runtime_enabled() -> bool:
    return os.environ.get(RUNTIME_E2E_ENV) == "1"


def _missing_modules(names: tuple[str, ...]) -> list[str]:
    return [name for name in names if importlib.util.find_spec(name) is None]


def _write_sage_motion(folder: Path, joint_names: list[str]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    first = [0.0 for _ in joint_names]
    second = [round(0.01 * (index + 1), 6) for index, _ in enumerate(joint_names)]
    zeros = [0.0 for _ in joint_names]
    (folder / "joint_list.txt").write_text("\n".join(joint_names) + "\n", encoding="utf-8")
    (folder / "metadata.yaml").write_text(
        "schema: sage\n"
        "time_unit: seconds\n"
        "position_unit: rad\n"
        "velocity_unit: rad/s\n"
        "torque_unit: Nm\n",
        encoding="utf-8",
    )
    (folder / "control.csv").write_text(
        "type,timestamp,positions\n"
        f'CONTROL,0.0,"{first}"\n'
        f'CONTROL,0.002,"{second}"\n',
        encoding="utf-8",
    )
    (folder / "state_motor.csv").write_text(
        "type,timestamp,positions,velocities,torques\n"
        f'STATE_MOTOR,0.0,"{first}","{zeros}","{zeros}"\n'
        f'STATE_MOTOR,0.002,"{second}","{zeros}","{zeros}"\n',
        encoding="utf-8",
    )


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "anchor_bench.cli", *args],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )


class RuntimePipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        if not _runtime_enabled():
            self.skipTest(f"set {RUNTIME_E2E_ENV}=1 to run the full Isaac/Newton runtime pipeline smoke test")

        missing = _missing_modules(("isaaclab", "isaaclab_newton", "newton", "torch", "cmaes"))
        if missing:
            self.skipTest(f"full runtime modules are missing: {', '.join(missing)}")

        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available inside the full runtime container")

        from anchor_bench.robots.registry import get_robot
        from anchor_bench.runtime.assets import AssetNotFoundError, resolve_runtime_asset

        try:
            resolve_runtime_asset(get_robot("so101"))
        except AssetNotFoundError as exc:
            self.skipTest(str(exc))

    def _run_in_workspace(self, root: Path) -> None:
        from anchor_bench.robots.registry import get_robot

        motion_dir = root / "inputs" / MOTION_NAME
        joint_names = list(get_robot("so101").spec.joint_names)
        _write_sage_motion(motion_dir, joint_names)

        sysid_output = root / "sysid"
        benchmark_output = root / "benchmark"
        analysis_output = root / "analysis"

        _run_cli(
            "sysid",
            "--robot",
            "so101",
            "--data",
            str(motion_dir),
            "--output",
            str(sysid_output),
            "--headless",
            "--num-envs",
            "2",
            "--max-iter",
            "1",
            "--max-trajectory-len",
            "2",
            "--seed",
            "42",
        )

        best_params = sysid_output / "best_params.yaml"
        self.assertTrue(best_params.exists())
        self.assertTrue((sysid_output / "optimization_log.csv").exists())
        self.assertTrue((sysid_output / "run_summary.yaml").exists())
        best_params_data = yaml.safe_load(best_params.read_text(encoding="utf-8"))
        self.assertEqual("so101", best_params_data["metadata"]["robot_name"])
        self.assertIn("best_mse", best_params_data["metadata"])

        _run_cli(
            "benchmark",
            "--robot",
            "so101",
            "--motion",
            str(motion_dir),
            "--output",
            str(benchmark_output),
            "--headless",
            "--motion-name",
            MOTION_NAME,
            "--actuator-params",
            str(best_params),
            "--dataset",
            DATASET_NAME,
            "--actuator-model",
            ACTUATOR_MODEL,
            "--seed",
            "42",
        )

        real_folder = benchmark_output / "real" / "so101" / DATASET_NAME / MOTION_NAME
        sim_folder = benchmark_output / "sim" / "so101" / DATASET_NAME / ACTUATOR_MODEL / MOTION_NAME
        for folder in (real_folder, sim_folder):
            with self.subTest(folder=folder):
                self.assertTrue((folder / "control.csv").exists())
                self.assertTrue((folder / "state_motor.csv").exists())
                self.assertTrue((folder / "joint_list.txt").exists())
        self.assertTrue((real_folder / "metadata.yaml").exists())
        self.assertTrue((sim_folder / "run_summary.yaml").exists())

        _run_cli(
            "analyze",
            "--robot",
            "so101",
            "--result-folder",
            str(benchmark_output),
            "--output",
            str(analysis_output),
            "--require-end-effector",
            "--motion",
            MOTION_NAME,
        )

        metrics = analysis_output / "metrics" / "so101_metrics.csv"
        ee_metrics = analysis_output / "metrics" / "so101_end_effector_metrics.csv"
        self.assertTrue(metrics.exists())
        self.assertTrue(ee_metrics.exists())
        metrics_text = metrics.read_text(encoding="utf-8")
        self.assertIn(f"{DATASET_NAME},{MOTION_NAME},{ACTUATOR_MODEL},", metrics_text)
        self.assertTrue(
            (analysis_output / "plots" / DATASET_NAME / MOTION_NAME / "joints" / "Rotation.png").exists()
        )
        self.assertTrue((analysis_output / "plots" / DATASET_NAME / MOTION_NAME / "end_effector.png").exists())

    def test_so101_sysid_benchmark_and_analysis_run_in_full_runtime(self):
        output_root = os.environ.get(RUNTIME_OUTPUT_ENV)
        if output_root:
            root = Path(output_root).expanduser() / f"runtime_pipeline_{uuid4().hex}"
            root.mkdir(parents=True, exist_ok=False)
            self._run_in_workspace(root)
        else:
            with tempfile.TemporaryDirectory(prefix="anchor-bench-runtime-pipeline-") as tmp:
                self._run_in_workspace(Path(tmp))


if __name__ == "__main__":
    unittest.main()
