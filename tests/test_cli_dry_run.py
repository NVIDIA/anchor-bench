
import subprocess
import sys
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "anchor_bench.cli", *args],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )


class CliDryRunTests(unittest.TestCase):
    def test_robot_list_and_describe(self):
        result = run_cli("robots", "list")
        self.assertIn("so101", result.stdout)
        result = run_cli("robots", "describe", "so101")
        self.assertIn("SO-101", result.stdout)

    def test_sysid_benchmark_and_analysis_dry_runs(self):
        data = "inputs/datasets/so101/custom_figure8_with_camera"
        result = run_cli("sysid", "--robot", "so101", "--data", data, "--output", "/tmp/anchor-bench-sysid", "--dry-run")
        self.assertIn("dry_run: true", result.stdout)
        self.assertIn("newton 1.2", result.stdout)

        result = run_cli("benchmark", "--robot", "so101", "--motion", data, "--dry-run")
        self.assertIn("benchmark", result.stdout)

        result = run_cli("analyze", "--robot", "so101", "--result-folder", "examples/results/so101", "--dry-run")
        self.assertIn("analysis", result.stdout)

        result = run_cli(
            "analyze",
            "--robot",
            "so101",
            "--result-folder",
            "examples/results/so101",
            "--require-end-effector",
            "--dry-run",
        )
        self.assertIn("--require-end-effector", result.stdout)

    def test_run_yaml_defaults_expand_in_dry_runs(self):
        result = run_cli("sysid", "--robot", "so101", "--dry-run")
        self.assertIn("inputs/datasets/so101/custom_figure8_with_camera", result.stdout)
        self.assertIn("outputs/sysid/so101/full", result.stdout)
        self.assertIn("--max-iter", result.stdout)
        self.assertIn("'100'", result.stdout)

        result = run_cli("benchmark", "--robot", "so101", "--dry-run")
        self.assertIn("inputs/datasets/so101/custom_figure8_with_camera", result.stdout)
        self.assertIn("outputs/sim2real_benchmark", result.stdout)

        result = run_cli("analyze", "--robot", "so101", "--dry-run")
        self.assertIn("outputs/sim2real_benchmark", result.stdout)
        self.assertIn("outputs/sim2real_analysis", result.stdout)

        result = run_cli("analyze", "--robot", "so101", "--motion", "custom_figure8_with_camera", "--dry-run")
        self.assertIn("--motion", result.stdout)
        self.assertIn("custom_figure8_with_camera", result.stdout)


    def test_train_actuator_dry_runs_for_pure_and_hybrid_gru(self):
        data = "inputs/datasets/so101/custom_figure8_with_camera"

        result = run_cli(
            "train-actuator",
            "--robot",
            "so101",
            "--model",
            "gru",
            "--data",
            data,
            "--joint-type",
            "Rotation",
            "--output",
            "/tmp/anchor-bench-training",
            "--input-columns",
            "position,position_error,velocity",
            "--target-column",
            "torque",
            "--skip-optuna",
            "--dry-run",
        )
        self.assertIn("workflow: train-actuator", result.stdout)
        self.assertIn("--model", result.stdout)
        self.assertIn("gru", result.stdout)
        self.assertIn("--input-columns", result.stdout)
        self.assertIn("position,position_error,velocity", result.stdout)
        self.assertIn("--target-column", result.stdout)
        self.assertIn("outputs/training", result.stdout)

        result = run_cli(
            "train-actuator",
            "--robot",
            "so101",
            "--model",
            "hybrid_residual",
            "--data",
            data,
            "--joint-type",
            "Rotation",
            "--output",
            "/tmp/anchor-bench-training",
            "--skip-optuna",
            "--dry-run",
        )
        self.assertIn("hybrid_residual", result.stdout)
        self.assertIn("residual torque", result.stdout)


if __name__ == "__main__":
    unittest.main()
