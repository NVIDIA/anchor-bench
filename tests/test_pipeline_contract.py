from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml


JOINT_NAMES = ["Rotation", "Pitch"]


def _write_sage_motion(
    folder: Path,
    *,
    position_offset: float = 0.0,
    velocity_offset: float = 0.0,
    torque_offset: float = 0.0,
) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "joint_list.txt").write_text("\n".join(JOINT_NAMES) + "\n", encoding="utf-8")
    (folder / "metadata.yaml").write_text(
        "time_unit: microseconds\n"
        "position_unit: rad\n"
        "velocity_unit: rad/s\n"
        "torque_unit: Nm\n",
        encoding="utf-8",
    )
    with (folder / "control.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["type", "timestamp", "positions"])
        for index, timestamp in enumerate((0, 5000, 10000)):
            values = [0.1 * index, 0.2 + 0.1 * index]
            writer.writerow(["CONTROL", timestamp, values])
    with (folder / "state_motor.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["type", "timestamp", "positions", "velocities", "torques"])
        for index, timestamp in enumerate((0, 5000, 10000)):
            positions = [position_offset + 0.1 * index, position_offset + 0.2 + 0.1 * index]
            velocities = [velocity_offset + 0.01 * index, velocity_offset + 0.02 * index]
            torques = [torque_offset + 0.5 * index, torque_offset + 0.25 * index]
            writer.writerow(["STATE_MOTOR", timestamp, positions, velocities, torques])


def _write_sysid_best_params(path: Path) -> Path:
    from anchor_bench.robots.registry import get_robot
    from anchor_bench.runtime.robot_config import build_fitted_actuator_document

    path.parent.mkdir(parents=True, exist_ok=True)
    document = build_fitted_actuator_document(
        get_robot("so101"),
        {
            "stiffness": {"Rotation": 12.5},
            "damping": {"Pitch": 1.75},
        },
        metadata={
            "source": "sysid",
            "best_mse": 0.001,
            "robot_name": "so101",
            "data_source": "synthetic",
        },
    )
    path.write_text(
        yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
    )
    return path


def _write_experiment_config(root: Path) -> Path:
    config = root / "experiment.yaml"
    config.write_text(
        f"""
schema_version: 1
robot: so101
outputs:
  benchmark: {root / "benchmark"}
  analysis: {root / "analysis"}
seeds: [42, 7]
benchmark:
  num_envs: 1
  headless: true
datasets:
  synthetic:
    real_root: {root / "datasets"}
    motions: [custom_motion]
actuator_models:
  fitted_implicit:
    type: implicit
    initial_params: {root / "initial.yaml"}
    fitted_params: {root / "sysid"}/seed_{{seed}}/best_params.yaml
    benchmark_params: fitted
analysis:
  seeds: [42]
  datasets: [synthetic]
  motions: "*"
  actuator_models: "*"
  plots:
    joints: false
    end_effector: false
    require_end_effector: false
""",
        encoding="utf-8",
    )
    (root / "initial.yaml").write_text("actuators: {}\n", encoding="utf-8")
    return config


def _write_benchmark_result(
    result_root: Path,
    *,
    seed: int,
    dataset: str,
    actuator_model: str,
    motion: str,
) -> None:
    seed_root = result_root / f"seed_{seed}"
    real_folder = seed_root / "real" / "so101" / dataset / motion
    sim_folder = seed_root / "sim" / "so101" / dataset / actuator_model / motion
    _write_sage_motion(real_folder)
    _write_sage_motion(
        sim_folder,
        position_offset=0.05,
        velocity_offset=0.01,
        torque_offset=0.02,
    )


class PipelineContractTests(unittest.TestCase):
    def test_sysid_fitted_params_flow_through_benchmark_and_analysis(self):
        from anchor_bench.experiments.runner import run_experiment_analysis, run_experiment_benchmark
        from anchor_bench.robots.registry import get_robot
        from anchor_bench.runtime.robot_config import load_implicit_actuator_params

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_sage_motion(root / "datasets" / "custom_motion")
            params_42 = _write_sysid_best_params(root / "sysid" / "seed_42" / "best_params.yaml")
            _write_sysid_best_params(root / "sysid" / "seed_7" / "best_params.yaml")
            config = _write_experiment_config(root)

            loaded_params = load_implicit_actuator_params(get_robot("so101"), params_42)
            self.assertEqual(12.5, loaded_params["stiffness"]["Rotation"])
            self.assertEqual(47.782635, loaded_params["stiffness"]["Pitch"])
            self.assertEqual(1.75, loaded_params["damping"]["Pitch"])
            self.assertEqual(0.0, loaded_params["motor_lag_ms"])
            self.assertNotIn("best_mse", loaded_params)

            with patch("anchor_bench.benchmark.newton_runner.run_benchmark_batch") as run_batch:
                run_experiment_benchmark(config, dry_run=False)

            seed_42_call = next(call for call in run_batch.call_args_list if call.kwargs["seed"] == 42)
            robot_arg, motions_arg, output_arg = seed_42_call.args
            self.assertEqual("so101", robot_arg)
            self.assertEqual("custom_motion", motions_arg[0].motion_name)
            self.assertTrue(motions_arg[0].motion.endswith("datasets/custom_motion"))
            self.assertTrue(output_arg.endswith("benchmark/seed_42"))
            self.assertEqual(str(params_42), seed_42_call.kwargs["actuator_params"])
            self.assertEqual("synthetic", seed_42_call.kwargs["dataset"])
            self.assertEqual("fitted_implicit", seed_42_call.kwargs["actuator_model"])
            self.assertTrue(seed_42_call.kwargs["headless"])
            self.assertEqual(1, seed_42_call.kwargs["num_envs"])

            _write_benchmark_result(
                root / "benchmark",
                seed=42,
                dataset="synthetic",
                actuator_model="fitted_implicit",
                motion="custom_motion",
            )

            with patch("anchor_bench.analysis.runtime._load_pyplot") as load_pyplot:
                metrics_plan = run_experiment_analysis(config, dry_run=False)
                load_pyplot.assert_not_called()

            metrics_path = root / "analysis" / "metrics" / "so101_metrics.csv"
            with metrics_path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))

        self.assertEqual("experiment analyze", metrics_plan.workflow)
        self.assertEqual(2, len(rows))
        self.assertEqual({"Rotation", "Pitch"}, {row["joint"] for row in rows})
        for row in rows:
            self.assertEqual("synthetic", row["dataset"])
            self.assertEqual("custom_motion", row["motion"])
            self.assertEqual("fitted_implicit", row["actuator_model"])
            self.assertEqual("42", row["seed"])
            self.assertGreater(float(row["position_rmse"]), 0.0)
            self.assertGreaterEqual(float(row["velocity_rmse"]), 0.0)
            self.assertGreaterEqual(float(row["torque_rmse"]), 0.0)


if __name__ == "__main__":
    unittest.main()
