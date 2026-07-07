
from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


def _write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _demo_rows(length: int = 8) -> list[dict[str, float]]:
    rows = []
    for index in range(length):
        position = 0.1 * index
        error = 1.0 - position
        velocity = 0.05 * index
        torque = 2.0 * error - 0.25 * velocity
        rows.append(
            {
                "position": position,
                "position_error": error,
                "velocity": velocity,
                "torque": torque,
            }
        )
    return rows


def _write_sage_training_episode(root: Path, *, offset: float = 0.0) -> None:
    root.mkdir(parents=True)
    (root / "joint_list.txt").write_text("Rotation\nPitch\n", encoding="utf-8")
    (root / "metadata.yaml").write_text(
        "schema: sage\n"
        "time_unit: microseconds\n"
        "position_unit: rad\n"
        "velocity_unit: rad/s\n"
        "torque_unit: N*m\n",
        encoding="utf-8",
    )
    control = ["type,timestamp,positions"]
    state = ["type,timestamp,positions,velocities,torques"]
    for index in range(3):
        control.append(
            f'CONTROL,{index * 2000},"[{offset + 0.1 + index}, {offset + 0.2 + index}]"'
        )
        state.append(
            f'STATE_MOTOR,{index * 2000},'
            f'"[{offset + 0.3 + index}, {offset + 0.4 + index}]",'
            f'"[{0.5 + 0.1 * index}, {1.5 + 0.1 * index}]",'
            f'"[{10.0 + index}, {20.0 + index}]"'
        )
    (root / "control.csv").write_text("\n".join(control) + "\n", encoding="utf-8")
    (root / "state_motor.csv").write_text("\n".join(state) + "\n", encoding="utf-8")


class TrainingDataTests(unittest.TestCase):
    def test_csv_discovery_split_and_stats_are_deterministic(self):
        from anchor_bench.training.data import (
            compute_normalization_stats,
            discover_csv_files,
            split_csv_files,
            write_stats_json,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index in range(5):
                _write_csv(root / f"sample_{index}.csv", _demo_rows())

            csvs = discover_csv_files(root)
            self.assertEqual([f"sample_{i}.csv" for i in range(5)], [path.name for path in csvs])

            train_a, val_a = split_csv_files(csvs, seed=123, val_fraction=0.4)
            train_b, val_b = split_csv_files(csvs, seed=123, val_fraction=0.4)
            self.assertEqual(train_a, train_b)
            self.assertEqual(val_a, val_b)
            self.assertEqual(3, len(train_a))
            self.assertEqual(2, len(val_a))

            stats = compute_normalization_stats(
                train_a,
                input_columns=["position", "position_error", "velocity"],
                target_columns=["torque"],
                metadata={"model": "gru", "joint_type": "Rotation"},
            )
            self.assertEqual(1, stats["schema_version"])
            self.assertEqual("gru", stats["model"])
            self.assertEqual(["position", "position_error", "velocity"], stats["input_columns"])
            self.assertEqual(["torque"], stats["target_columns"])
            self.assertIn("position", stats["normalization"]["inputs"]["mean"])
            self.assertIn("torque", stats["normalization"]["targets"]["std"])

            stats_path = write_stats_json(root / "stats.json", stats)
            loaded = json.loads(stats_path.read_text(encoding="utf-8"))
            self.assertEqual(stats, loaded)

    def test_sage_discovery_keeps_whole_episodes(self):
        from anchor_bench.training.data import discover_sage_dirs, split_csv_files

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_sage_training_episode(root / "episode_b")
            _write_sage_training_episode(root / "nested" / "episode_a", offset=1.0)
            (root / "unrelated").mkdir()

            episodes = discover_sage_dirs(root)
            self.assertEqual(
                {"episode_a", "episode_b"},
                {path.name for path in episodes},
            )
            train_a, val_a = split_csv_files(episodes, seed=17)
            train_b, val_b = split_csv_files(episodes, seed=17)
            self.assertEqual((train_a, val_a), (train_b, val_b))
            self.assertEqual(1, len(train_a))
            self.assertEqual(1, len(val_a))

    def test_column_resolution_uses_model_defaults_and_validates_headers(self):
        from anchor_bench.training.data import resolve_training_columns

        header = ["position", "position_error", "velocity", "torque"]
        pure = resolve_training_columns(header, model="gru")
        self.assertEqual(["position", "position_error", "velocity"], pure.input_columns)
        self.assertEqual(["torque"], pure.target_columns)

        hybrid = resolve_training_columns(header, model="hybrid_residual")
        self.assertEqual(["position", "position_error", "velocity"], hybrid.input_columns)
        self.assertEqual(["torque_residual"], hybrid.target_columns)

        with self.assertRaisesRegex(ValueError, "missing required CSV columns: velocity"):
            resolve_training_columns(["position", "position_error", "torque"], model="gru")

    def test_column_resolution_accepts_custom_columns_and_rejects_unsupported_inputs(self):
        from anchor_bench.training.data import resolve_training_columns

        header = ["position", "position_error", "velocity", "solver_pd", "measured_torque"]
        columns = resolve_training_columns(
            header,
            model="gru",
            input_columns=["position", "solver_pd"],
            target_column="measured_torque",
        )

        self.assertEqual(["position", "solver_pd"], columns.input_columns)
        self.assertEqual(["measured_torque"], columns.target_columns)
        self.assertEqual(["measured_torque"], columns.source_target_columns)

        hybrid = resolve_training_columns(
            ["position", "position_error", "velocity", "torque"],
            model="hybrid_residual",
            input_columns=["position", "position_error", "velocity", "solver_pd"],
            derived_columns=["solver_pd"],
        )
        self.assertEqual(["torque_residual"], hybrid.target_columns)

        dynamic = resolve_training_columns(
            ["position", "position_error", "velocity", "torque"],
            model="gru",
            input_columns=["position", "dynamic_bias"],
            derived_columns=["dynamic_bias"],
        )
        self.assertEqual(["position", "dynamic_bias"], dynamic.input_columns)

        with self.assertRaisesRegex(ValueError, "duplicate GRU input columns: position"):
            resolve_training_columns(
                ["position", "torque"],
                model="gru",
                input_columns=["position", "position"],
            )

        with self.assertRaisesRegex(ValueError, "unsupported GRU input columns: qfrc_bias"):
            resolve_training_columns(
                ["position", "position_error", "velocity", "torque"],
                model="gru",
                input_columns=["position", "qfrc_bias"],
            )


class TrainingRunnerTests(unittest.TestCase):
    def test_training_config_resolves_run_yaml_defaults(self):
        from anchor_bench.core.schemas import TrainActuatorRunConfig
        from anchor_bench.training.runner import resolve_train_actuator_run_config

        resolved = resolve_train_actuator_run_config(
            TrainActuatorRunConfig(robot="so101", dry_run=True)
        )

        self.assertEqual("so101", resolved.robot)
        self.assertEqual("gru", resolved.model)
        self.assertTrue(resolved.output.endswith("outputs/training/so101"))
        self.assertEqual("full", resolved.joint_type)
        self.assertTrue(resolved.skip_optuna)
        self.assertEqual(42, resolved.seed)
        self.assertEqual(["position", "position_error", "velocity"], resolved.input_columns)
        self.assertEqual("torque", resolved.target_column)

    def test_training_config_resolves_cli_column_overrides(self):
        from anchor_bench.core.schemas import TrainActuatorRunConfig
        from anchor_bench.training.runner import resolve_train_actuator_run_config

        resolved = resolve_train_actuator_run_config(
            TrainActuatorRunConfig(
                robot="so101",
                data="inputs/datasets/so101/custom_figure8_with_camera",
                input_columns=["position", "solver_pd"],
                target_column="measured_torque",
                dry_run=True,
            )
        )

        self.assertEqual(["position", "solver_pd"], resolved.input_columns)
        self.assertEqual("measured_torque", resolved.target_column)
        with self.assertRaisesRegex(ValueError, "duplicate GRU input columns: position"):
            resolve_train_actuator_run_config(
                TrainActuatorRunConfig(
                    robot="so101",
                    input_columns=["position", "position"],
                    dry_run=True,
                )
            )

    def test_sage_sequence_uses_full_state_for_scalar_dynamic_bias(self):
        from dataclasses import replace

        import numpy as np

        from anchor_bench.training.trainer import (
            TrainingJob, _load_sequences, _runtime_metadata,
        )

        class FakeBiasProvider:
            def __init__(self):
                self.calls = []

            def compute(self, *, joint_names, positions, velocities):
                self.calls.append((list(joint_names), positions.copy(), velocities.copy()))
                return positions + 10.0 * velocities

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            episode = root / "episode"
            _write_sage_training_episode(episode)
            provider = FakeBiasProvider()
            job = TrainingJob(
                model="gru",
                joint_type="Pitch",
                train_csvs=[],
                val_csvs=[],
                train_sage_dirs=[episode],
                val_sage_dirs=[],
                output_dir=root / "out",
                input_columns=["position", "position_error", "velocity", "dynamic_bias"],
                target_columns=["torque"],
                source_target_columns=["torque"],
                seed=1,
                final_epochs=1,
                trial_epochs=1,
                trials=1,
                skip_optuna=True,
                n_streams=1,
                noise_std=0.0,
                dynamic_bias_provider=provider,
                metadata={
                    "sample_dt_s": 0.002,
                    "delay": {"handling": "none", "training_delay_s": 0.0},
                    "dynamic_bias_derivation": {
                        "source": "newton.solvers.SolverMuJoCo.qfrc_bias",
                        "asset_sha256": "demo-hash",
                    },
                },
            )

            train, val = _load_sequences(job)
            inconsistent = root / "inconsistent"
            _write_sage_training_episode(inconsistent)
            (inconsistent / "joint_list.txt").write_text(
                "Rotation\nElbow\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "common SAGE joint set"):
                _load_sequences(replace(job, train_sage_dirs=[episode, inconsistent]))

        self.assertEqual([], val)
        self.assertEqual(1, len(train))
        self.assertEqual(["Rotation", "Pitch"], provider.calls[0][0])
        self.assertEqual((3, 2), provider.calls[0][1].shape)
        np.testing.assert_allclose(provider.calls[0][1][0], [0.3, 0.4])
        features, targets = train[0]
        np.testing.assert_allclose(
            features,
            [
                [0.4, -0.2, 1.5, 15.4],
                [1.4, -0.2, 1.6, 17.4],
                [2.4, -0.2, 1.7, 19.4],
            ],
            atol=1.0e-6,
        )
        np.testing.assert_allclose(targets[:, 0], [20.0, 21.0, 22.0])

        stats = {
            "normalization": {
                "inputs": {
                    "mean": {column: 0.0 for column in job.input_columns},
                    "std": {column: 1.0 for column in job.input_columns},
                },
                "targets": {
                    "mean": {"torque": 0.0},
                    "std": {"torque": 1.0},
                },
            },
        }
        runtime_metadata = _runtime_metadata(job, stats)
        self.assertEqual(
            "demo-hash",
            runtime_metadata["dynamic_bias_derivation"]["asset_sha256"],
        )

        self.assertEqual(
            ["Pitch", "Rotation"],
            runtime_metadata["dynamic_bias_derivation"]["trajectory_joint_names"],
        )

    def test_scalar_csv_ignores_unselected_text_columns(self):
        from unittest.mock import patch

        import numpy as np

        from anchor_bench.training.trainer import TrainingJob, _load_sequences

        class FakeSeries:
            def __init__(self, values):
                self._values = np.asarray(values)

            def to_numpy(self):
                return self._values

        class FakeFrame:
            def __init__(self):
                self._columns = {
                    "episode": ["run-a", "run-a"],
                    "position": [0.1, 0.2],
                    "position_error": [0.3, 0.4],
                    "velocity": [0.5, 0.6],
                    "torque": [0.7, 0.8],
                }
                self.columns = list(self._columns)

            def __getitem__(self, name):
                return FakeSeries(self._columns[name])

        class FakePandas:
            @staticmethod
            def read_csv(_path):
                return FakeFrame()

        job = TrainingJob(
            model="gru",
            joint_type="Rotation",
            train_csvs=[Path("trajectory.csv")],
            val_csvs=[],
            output_dir=Path("unused"),
            input_columns=["position", "position_error", "velocity"],
            target_columns=["torque"],
            source_target_columns=["torque"],
            seed=1,
            final_epochs=1,
            trial_epochs=1,
            trials=1,
            skip_optuna=True,
            n_streams=1,
            noise_std=0.0,
        )

        with patch("anchor_bench.training.trainer._import_pandas", return_value=FakePandas()):
            train, val = _load_sequences(job)

        self.assertEqual([], val)
        np.testing.assert_allclose(
            train[0][0],
            [[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]],
        )
        np.testing.assert_allclose(train[0][1], [[0.7], [0.8]])

    def test_g1_dynamic_bias_dry_run_does_not_require_newton_or_assets(self):
        from anchor_bench.core.schemas import TrainActuatorRunConfig
        from anchor_bench.training.runner import run_train_actuator

        plan = run_train_actuator(
            TrainActuatorRunConfig(
                robot="g1",
                data="inputs/datasets/g1/g1_arm_multi_joint",
                input_columns=[
                    "position", "position_error", "velocity", "dynamic_bias",
                ],
                dry_run=True,
            )
        )

        self.assertTrue(plan.dry_run)
        self.assertEqual("sage", plan.metadata["data_format"])
        self.assertIn("dynamic_bias", plan.metadata["input_columns"])

    def test_grouped_residual_model_requires_shared_physics_baseline(self):
        from anchor_bench.training.runner import _residual_physics_for_joint

        with self.assertRaisesRegex(ValueError, "one shared stiffness"):
            _residual_physics_for_joint("so101", "full")

    def test_grouped_model_requires_complete_parameter_map(self):
        from types import SimpleNamespace

        from anchor_bench.training.runner import _param_for_actuator

        actuator = SimpleNamespace(name="arm", joints=["shoulder", "elbow"])
        with self.assertRaisesRegex(ValueError, "missing: elbow"):
            _param_for_actuator(
                {"shoulder": 20.0},
                joint_type="full",
                actuator=actuator,
                parameter_name="stiffness",
            )

    @unittest.skipUnless(
        importlib.util.find_spec("torch") and importlib.util.find_spec("pandas"),
        "training smoke test requires torch and pandas",
    )
    def test_tiny_gru_training_smoke_writes_checkpoint_and_stats(self):
        from anchor_bench.core.schemas import TrainActuatorRunConfig
        from anchor_bench.training.runner import run_train_actuator

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            output_dir = root / "out"
            data_dir.mkdir()
            _write_csv(data_dir / "a.csv", _demo_rows(length=12))
            _write_csv(data_dir / "b.csv", _demo_rows(length=12))

            plan = run_train_actuator(
                TrainActuatorRunConfig(
                    robot="so101",
                    model="gru",
                    data=str(data_dir),
                    output=str(output_dir),
                    joint_type="Rotation",
                    skip_optuna=True,
                    final_epochs=1,
                    n_streams=1,
                    seed=5,
                    dry_run=False,
                )
            )

            self.assertFalse(plan.dry_run)
            self.assertTrue((output_dir / "gru_Rotation_best.pt").exists())
            self.assertTrue((output_dir / "gru_Rotation_network.pt").exists())
            stats = json.loads((output_dir / "gru_Rotation_stats.json").read_text(encoding="utf-8"))
            self.assertEqual("gru", stats["model"])
            self.assertEqual("Rotation", stats["joint_type"])
            self.assertIn("network", plan.metadata["artifacts"])

            import torch

            extra_files = {"metadata.json": ""}
            torch.jit.load(str(output_dir / "gru_Rotation_network.pt"), _extra_files=extra_files)
            metadata = json.loads(extra_files["metadata.json"].decode("utf-8"))
            self.assertEqual("gru", metadata["model_type"])
            self.assertEqual(["torque"], metadata["target_columns"])
            self.assertEqual(0.002, metadata["sample_dt_s"])
            self.assertEqual("none", metadata["delay"]["handling"])

    @unittest.skipUnless(
        importlib.util.find_spec("torch") and importlib.util.find_spec("pandas"),
        "training smoke test requires torch and pandas",
    )
    def test_residual_and_solver_pd_exports_embed_physics_baselines(self):
        from anchor_bench.core.schemas import TrainActuatorRunConfig
        from anchor_bench.training.runner import run_train_actuator

        import torch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            data_dir.mkdir()
            _write_csv(data_dir / "a.csv", _demo_rows(length=12))
            _write_csv(data_dir / "b.csv", _demo_rows(length=12))

            cases = [
                {
                    "model": "hybrid_residual",
                    "input_columns": ["position", "position_error", "velocity"],
                    "network_name": "hybrid_residual_Rotation_network.pt",
                    "target": "torque_residual",
                    "has_friction": True,
                },
                {
                    "model": "gru",
                    "input_columns": ["position", "position_error", "velocity", "solver_pd"],
                    "network_name": "gru_Rotation_network.pt",
                    "target": "torque",
                    "has_friction": False,
                },
            ]
            for index, case in enumerate(cases):
                with self.subTest(model=case["model"]):
                    output_dir = root / f"out_{index}"
                    run_train_actuator(
                        TrainActuatorRunConfig(
                            robot="so101",
                            model=case["model"],
                            data=str(data_dir),
                            output=str(output_dir),
                            joint_type="Rotation",
                            input_columns=case["input_columns"],
                            skip_optuna=True,
                            final_epochs=1,
                            n_streams=1,
                            seed=5,
                            dry_run=False,
                        )
                    )

                    extra_files = {"metadata.json": ""}
                    torch.jit.load(
                        str(output_dir / case["network_name"]),
                        _extra_files=extra_files,
                    )
                    metadata = json.loads(extra_files["metadata.json"].decode("utf-8"))
                    self.assertEqual([case["target"]], metadata["target_columns"])
                    self.assertEqual(
                        {"kp": 48.419899, "kd": 3.334595, "velocity_target": "zero"},
                        metadata["pd_baseline"],
                    )
                    if case["has_friction"]:
                        self.assertEqual(
                            {"dry": 0.347432, "viscous": 1.590709},
                            metadata["friction_baseline"],
                        )
                    else:
                        self.assertNotIn("friction_baseline", metadata)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "TorchScript export test requires torch")
    def test_gru_checkpoint_export_writes_scripted_network_with_gru_attribute(self):
        from anchor_bench.training.export import build_runtime_gru, export_gru_checkpoint

        import torch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint_path = root / "gru_demo_best.pt"
            model = build_runtime_gru(input_size=3, output_size=1, hidden_size=4, num_layers=1, dropout=0.0)
            torch.save(
                {
                    "model": "gru",
                    "joint_type": "demo",
                    "model_state_dict": model.state_dict(),
                    "input_columns": ["position", "position_error", "velocity"],
                    "target_columns": ["torque"],
                    "runtime_metadata": {
                        "schema_version": 1,
                        "model_type": "gru",
                        "input_columns": ["position", "position_error", "velocity"],
                        "target_columns": ["torque"],
                        "sample_dt_s": 0.002,
                        "normalization": {
                            "inputs": {
                                "mean": {"position": 0.0, "position_error": 0.0, "velocity": 0.0},
                                "std": {"position": 1.0, "position_error": 1.0, "velocity": 1.0},
                            },
                            "targets": {
                                "mean": {"torque": 0.0},
                                "std": {"torque": 1.0},
                            },
                        },
                        "delay": {"handling": "none", "training_delay_s": 0.0},
                    },
                    "hyperparameters": {"hidden_size": 4, "num_layers": 1, "dropout": 0.0},
                },
                checkpoint_path,
            )

            network_path = export_gru_checkpoint(checkpoint_path)
            extra_files = {"metadata.json": ""}
            network = torch.jit.load(str(network_path), _extra_files=extra_files)
            self.assertTrue(hasattr(network, "gru"))
            metadata = json.loads(extra_files["metadata.json"].decode("utf-8"))
            self.assertEqual("gru", metadata["model_type"])
            self.assertEqual(["torque"], metadata["target_columns"])
            self.assertEqual(0.002, metadata["sample_dt_s"])
            output, hidden = network(torch.zeros(2, 1, 3), torch.zeros(1, 2, 4))
            self.assertEqual((2, 1, 1), tuple(output.shape))
            self.assertEqual((1, 2, 4), tuple(hidden.shape))


if __name__ == "__main__":
    unittest.main()
