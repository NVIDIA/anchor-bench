from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[1]


def _write_sage(folder: Path) -> None:
    folder.mkdir(parents=True)
    (folder / "joint_list.txt").write_text("Rotation\nPitch\n", encoding="utf-8")
    (folder / "control.csv").write_text(
        'type,timestamp,positions\nCONTROL,0.0,"[0.0, 0.0]"\n',
        encoding="utf-8",
    )
    (folder / "state_motor.csv").write_text(
        'type,timestamp,positions,velocities,torques\n'
        'STATE_MOTOR,0.0,"[0.0, 0.0]","[0.0, 0.0]","[0.0, 0.0]"\n',
        encoding="utf-8",
    )
    (folder / "metadata.yaml").write_text(
        "time_unit: seconds\n"
        "position_unit: rad\n"
        "velocity_unit: rad/s\n"
        "torque_unit: Nm\n",
        encoding="utf-8",
    )


def _write_experiment_config(root: Path, text: str) -> Path:
    config = root / "experiment.yaml"
    config.write_text(text, encoding="utf-8")
    return config


class ExperimentConfigTests(unittest.TestCase):
    def test_discovery_ignores_sage_folder_with_invalid_metadata(self):
        from anchor_bench.experiments.runner import ExperimentConfigError, expand_benchmark_runs, load_experiment_config

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "datasets" / "joints_all"
            invalid_motion = dataset_root / "invalid_motion"
            _write_sage(invalid_motion)
            (invalid_motion / "metadata.yaml").write_text(
                "time_unit: milliseconds\n"
                "position_unit: rad\n"
                "velocity_unit: rad/s\n"
                "torque_unit: Nm\n",
                encoding="utf-8",
            )
            initial_params = root / "initial.yaml"
            initial_params.write_text("actuators: {}\n", encoding="utf-8")
            config = _write_experiment_config(
                root,
                f"""
schema_version: 1
robot: so101
datasets:
  joints_all:
    real_root: {dataset_root}
actuator_models:
  implicit:
    type: implicit
    initial_params: {initial_params}
    benchmark_params: initial
""",
            )

            with self.assertRaisesRegex(ExperimentConfigError, "has no valid SAGE child folders"):
                expand_benchmark_runs(load_experiment_config(config))

    def test_explicit_motion_reports_invalid_sage_metadata(self):
        from anchor_bench.experiments.runner import ExperimentConfigError, expand_benchmark_runs, load_experiment_config

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "datasets" / "joints_all"
            invalid_motion = dataset_root / "invalid_motion"
            _write_sage(invalid_motion)
            (invalid_motion / "metadata.yaml").write_text(
                "time_unit: seconds\n"
                "position_unit: degrees\n"
                "velocity_unit: rad/s\n"
                "torque_unit: Nm\n",
                encoding="utf-8",
            )
            initial_params = root / "initial.yaml"
            initial_params.write_text("actuators: {}\n", encoding="utf-8")
            config = _write_experiment_config(
                root,
                f"""
schema_version: 1
robot: so101
datasets:
  joints_all:
    real_root: {dataset_root}
    motions: [invalid_motion]
actuator_models:
  implicit:
    type: implicit
    initial_params: {initial_params}
    benchmark_params: initial
""",
            )

            with self.assertRaisesRegex(ExperimentConfigError, "position_unit"):
                expand_benchmark_runs(load_experiment_config(config))

    def test_benchmark_num_envs_defaults_and_validates(self):
        from anchor_bench.experiments.runner import ExperimentConfigError, load_experiment_config

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "datasets" / "joints_all"
            _write_sage(dataset_root / "coupled_joints")
            config = _write_experiment_config(
                root,
                f"""
schema_version: 1
robot: so101
datasets:
  joints_all:
    real_root: {dataset_root}
actuator_models:
  implicit:
    type: implicit
    initial_params: {root / "initial.yaml"}
    benchmark_params: initial
""",
            )
            (root / "initial.yaml").write_text("stiffness: {{}}\n", encoding="utf-8")

            loaded = load_experiment_config(config)

            self.assertEqual(1, loaded.benchmark.num_envs)
            self.assertEqual("outputs/sim2real_benchmark", loaded.outputs.benchmark)
            self.assertEqual("outputs/sim2real_analysis", loaded.outputs.analysis)

            config.write_text(
                f"""
schema_version: 1
robot: so101
benchmark:
  num_envs: 2
datasets:
  joints_all:
    real_root: {dataset_root}
actuator_models:
  implicit:
    type: implicit
    initial_params: {root / "initial.yaml"}
    benchmark_params: initial
""",
                encoding="utf-8",
            )

            loaded = load_experiment_config(config)

            self.assertEqual(2, loaded.benchmark.num_envs)

            config.write_text(
                f"""
schema_version: 1
robot: so101
benchmark:
  num_envs: 0
datasets:
  joints_all:
    real_root: {dataset_root}
actuator_models:
  implicit:
    type: implicit
    initial_params: {root / "initial.yaml"}
    benchmark_params: initial
""",
                encoding="utf-8",
            )

            with self.assertRaises(ExperimentConfigError):
                load_experiment_config(config)

    def test_expands_default_fitted_params_and_discovers_valid_motions(self):
        from anchor_bench.experiments.runner import expand_benchmark_runs, load_experiment_config

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "datasets" / "joints_all"
            _write_sage(dataset_root / "coupled_joints")
            _write_sage(dataset_root / "elbow_only")
            (dataset_root / "notes").mkdir()
            (root / "initial.yaml").write_text("actuators: {}\n", encoding="utf-8")
            for seed in (42, 7):
                params = root / "sysid" / f"seed_{seed}" / "best_params.yaml"
                params.parent.mkdir(parents=True)
                params.write_text("stiffness: {}\n", encoding="utf-8")

            config = _write_experiment_config(
                root,
                f"""
schema_version: 1
robot: so101
outputs:
  benchmark: {root / "benchmark"}
  analysis: {root / "analysis"}
seeds: [42, 7]
datasets:
  joints_all:
    real_root: {dataset_root}
actuator_models:
  implicit:
    type: implicit
    initial_params: {root / "initial.yaml"}
    fitted_params: {root / "sysid"}/seed_{{seed}}/best_params.yaml
analysis:
  plots:
    joints: true
    end_effector: true
""",
            )

            runs = expand_benchmark_runs(load_experiment_config(config))

        self.assertEqual(
            [
                ("joints_all", "coupled_joints", "implicit", 42),
                ("joints_all", "elbow_only", "implicit", 42),
                ("joints_all", "coupled_joints", "implicit", 7),
                ("joints_all", "elbow_only", "implicit", 7),
            ],
            [(run.dataset, run.motion_name, run.actuator_model, run.seed) for run in runs],
        )
        self.assertTrue(all(run.actuator_params.endswith("best_params.yaml") for run in runs))

    def test_expands_benchmark_batches_by_num_envs_and_compatible_keys(self):
        from anchor_bench.experiments.runner import expand_benchmark_batches, load_experiment_config

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "datasets" / "joints_all"
            for motion in ("custom_motion", "diagonal_sweep", "frequency_sweep"):
                _write_sage(dataset_root / motion)
            params = root / "sysid" / "seed_42" / "best_params.yaml"
            params.parent.mkdir(parents=True)
            params.write_text("stiffness: {}\n", encoding="utf-8")
            config = _write_experiment_config(
                root,
                f"""
schema_version: 1
robot: so101
benchmark:
  num_envs: 2
outputs:
  benchmark: {root / "benchmark"}
datasets:
  joints_all:
    real_root: {dataset_root}
actuator_models:
  implicit:
    type: implicit
    fitted_params: {params}
""",
            )

            batches = expand_benchmark_batches(load_experiment_config(config))

        self.assertEqual([2, 1], [len(batch.runs) for batch in batches])
        self.assertEqual([2, 1], [batch.batch_size for batch in batches])
        self.assertEqual([2, 2], [batch.configured_num_envs for batch in batches])
        self.assertEqual(
            [["custom_motion", "diagonal_sweep"], ["frequency_sweep"]],
            [[run.motion_name for run in batch.runs] for batch in batches],
        )

    def test_benchmark_batches_do_not_mix_seeds(self):
        from anchor_bench.experiments.runner import expand_benchmark_batches, load_experiment_config

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "datasets" / "joints_all"
            _write_sage(dataset_root / "custom_motion")
            _write_sage(dataset_root / "diagonal_sweep")
            for seed in (42, 7):
                params = root / "sysid" / f"seed_{seed}" / "best_params.yaml"
                params.parent.mkdir(parents=True)
                params.write_text("stiffness: {}\n", encoding="utf-8")
            config = _write_experiment_config(
                root,
                f"""
schema_version: 1
robot: so101
benchmark:
  num_envs: 4
outputs:
  benchmark: {root / "benchmark"}
seeds: [42, 7]
datasets:
  joints_all:
    real_root: {dataset_root}
actuator_models:
  implicit:
    type: implicit
    fitted_params: {root / "sysid"}/seed_{{seed}}/best_params.yaml
""",
            )

            batches = expand_benchmark_batches(load_experiment_config(config))

        self.assertEqual(2, len(batches))
        self.assertEqual([{42}, {7}], [{run.seed for run in batch.runs} for batch in batches])

    def test_missing_fitted_params_mentions_running_sysid_first(self):
        from anchor_bench.experiments.runner import ExperimentConfigError, expand_benchmark_runs, load_experiment_config

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "datasets" / "joints_all"
            _write_sage(dataset_root / "coupled_joints")
            (root / "initial.yaml").write_text("actuators: {}\n", encoding="utf-8")
            config = _write_experiment_config(
                root,
                f"""
schema_version: 1
robot: so101
seeds: [42]
datasets:
  joints_all:
    real_root: {dataset_root}
actuator_models:
  implicit:
    type: implicit
    initial_params: {root / "initial.yaml"}
    fitted_params: {root / "sysid"}/seed_{{seed}}/best_params.yaml
""",
            )

            with self.assertRaises(ExperimentConfigError) as ctx:
                expand_benchmark_runs(load_experiment_config(config))

        message = str(ctx.exception)
        self.assertIn("best_params.yaml does not exist", message)
        self.assertIn("Run sysid first or set benchmark_params: initial", message)

    def test_experiment_benchmark_dry_run_warns_for_missing_fitted_params(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "datasets" / "joints_all"
            _write_sage(dataset_root / "coupled_joints")
            config = _write_experiment_config(
                root,
                f"""
schema_version: 1
robot: so101
outputs:
  benchmark: {root / "benchmark"}
seeds: [42]
datasets:
  joints_all:
    real_root: {dataset_root}
actuator_models:
  implicit:
    type: implicit
    fitted_params: {root / "sysid"}/seed_{{seed}}/best_params.yaml
""",
            )

            result = subprocess.run(
                [sys.executable, "-m", "anchor_bench.cli", "experiment", "benchmark", str(config), "--dry-run"],
                cwd=REPO,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )

        self.assertIn("workflow: experiment benchmark", result.stdout)
        self.assertIn("warnings:", result.stdout)
        self.assertIn("best_params.yaml does not exist", result.stdout)
        self.assertIn("warning: best_params.yaml does not exist", result.stderr)

    def test_experiment_benchmark_runtime_uses_actual_batch_size_as_num_envs(self):
        from anchor_bench.experiments.runner import run_experiment_benchmark

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "datasets" / "joints_all"
            for motion in ("custom_motion", "diagonal_sweep", "frequency_sweep"):
                _write_sage(dataset_root / motion)
            params = root / "sysid" / "seed_42" / "best_params.yaml"
            params.parent.mkdir(parents=True)
            params.write_text("stiffness: {}\n", encoding="utf-8")
            config = _write_experiment_config(
                root,
                f"""
schema_version: 1
robot: so101
benchmark:
  num_envs: 2
  headless: true
outputs:
  benchmark: {root / "benchmark"}
datasets:
  joints_all:
    real_root: {dataset_root}
actuator_models:
  implicit:
    type: implicit
    fitted_params: {params}
""",
            )

            with patch("anchor_bench.benchmark.newton_runner.run_benchmark_batch") as run_batch:
                run_experiment_benchmark(config, dry_run=False)

        self.assertEqual([2, 1], [call.kwargs["num_envs"] for call in run_batch.call_args_list])
        self.assertTrue(all(call.kwargs["headless"] for call in run_batch.call_args_list))

    def test_initial_benchmark_params_do_not_require_fitted_params(self):
        from anchor_bench.experiments.runner import expand_benchmark_runs, load_experiment_config

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "datasets" / "joints_all"
            _write_sage(dataset_root / "coupled_joints")
            initial_params = root / "initial.yaml"
            initial_params.write_text("actuators: {}\n", encoding="utf-8")
            config = _write_experiment_config(
                root,
                f"""
schema_version: 1
robot: so101
seeds: [42]
datasets:
  joints_all:
    real_root: {dataset_root}
actuator_models:
  implicit:
    type: implicit
    initial_params: {initial_params}
    fitted_params: {root / "sysid"}/seed_{{seed}}/best_params.yaml
    benchmark_params: initial
""",
            )

            runs = expand_benchmark_runs(load_experiment_config(config))

        self.assertEqual(str(initial_params), runs[0].actuator_params)

    def test_analysis_seed_subset_keeps_multi_seed_output_layout(self):
        from anchor_bench.experiments.runner import expand_analysis_runs, load_experiment_config

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "datasets" / "joints_all"
            _write_sage(dataset_root / "coupled_joints")
            config = _write_experiment_config(
                root,
                f"""
schema_version: 1
robot: so101
outputs:
  benchmark: {root / "benchmark"}
  analysis: {root / "analysis"}
seeds: [42, 7]
datasets:
  joints_all:
    real_root: {dataset_root}
actuator_models:
  implicit:
    type: implicit
    initial_params: {root / "initial.yaml"}
analysis:
  seeds: [42]
  datasets: [joints_all]
  motions: "*"
  actuator_models: "*"
""",
            )

            runs = expand_analysis_runs(load_experiment_config(config))

        self.assertEqual(1, len(runs))
        self.assertEqual([42], runs[0].seeds)
        self.assertTrue(runs[0].result_folder.endswith("benchmark"))
        self.assertTrue(runs[0].output.endswith("analysis"))

    def test_analysis_require_end_effector_expands_to_flag(self):
        from anchor_bench.experiments.runner import build_experiment_analysis_plan, expand_analysis_runs, load_experiment_config

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "datasets" / "joints_all"
            _write_sage(dataset_root / "coupled_joints")
            config = _write_experiment_config(
                root,
                f"""
schema_version: 1
robot: so101
outputs:
  benchmark: {root / "benchmark"}
  analysis: {root / "analysis"}
datasets:
  joints_all:
    real_root: {dataset_root}
actuator_models:
  implicit:
    type: implicit
    initial_params: {root / "initial.yaml"}
analysis:
  plots:
    joints: true
    end_effector: true
    require_end_effector: true
""",
            )

            loaded = load_experiment_config(config)
            runs = expand_analysis_runs(loaded)
            plan = build_experiment_analysis_plan(config)

        self.assertTrue(runs[0].plots.require_end_effector)
        self.assertIn("--require-end-effector", plan.runs[0])

    def test_experiment_benchmark_cli_dry_run_lists_expanded_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "datasets" / "joints_all"
            _write_sage(dataset_root / "coupled_joints")
            params = root / "sysid" / "seed_42" / "best_params.yaml"
            params.parent.mkdir(parents=True)
            params.write_text("stiffness: {}\n", encoding="utf-8")
            (root / "initial.yaml").write_text("actuators: {}\n", encoding="utf-8")
            config = _write_experiment_config(
                root,
                f"""
schema_version: 1
benchmark:
  headless: true
robot: so101
outputs:
  benchmark: {root / "benchmark"}
  analysis: {root / "analysis"}
seeds: [42]
datasets:
  joints_all:
    real_root: {dataset_root}
actuator_models:
  implicit:
    type: implicit
    initial_params: {root / "initial.yaml"}
    fitted_params: {root / "sysid"}/seed_{{seed}}/best_params.yaml
""",
            )

            result = subprocess.run(
                [sys.executable, "-m", "anchor_bench.cli", "experiment", "benchmark", str(config), "--dry-run"],
                cwd=REPO,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            analyze_result = subprocess.run(
                [sys.executable, "-m", "anchor_bench.cli", "experiment", "analyze", str(config), "--dry-run"],
                cwd=REPO,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )

        self.assertIn("workflow: experiment benchmark", result.stdout)
        self.assertIn("runs:", result.stdout)
        self.assertIn("--dataset", result.stdout)
        self.assertIn("joints_all", result.stdout)
        self.assertIn("--actuator-model", result.stdout)
        self.assertIn("implicit", result.stdout)
        self.assertIn("--headless", result.stdout)
        self.assertIn("--seed", result.stdout)
        self.assertIn("best_params.yaml", result.stdout)
        self.assertIn("benchmark_num_envs", result.stdout)
        self.assertIn("batch_count", result.stdout)
        self.assertIn("total_runs", result.stdout)
        self.assertIn("workflow: experiment analyze", analyze_result.stdout)
        self.assertIn("runs:", analyze_result.stdout)
        self.assertIn("anchor-bench", analyze_result.stdout)
        self.assertIn("analyze", analyze_result.stdout)


if __name__ == "__main__":
    unittest.main()
