
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
                    "hyperparameters": {"hidden_size": 4, "num_layers": 1, "dropout": 0.0},
                },
                checkpoint_path,
            )

            network_path = export_gru_checkpoint(checkpoint_path)
            extra_files = {"metadata.json": ""}
            network = torch.jit.load(str(network_path), _extra_files=extra_files)
            self.assertTrue(hasattr(network, "gru"))
            self.assertFalse(extra_files["metadata.json"])
            output, hidden = network(torch.zeros(2, 1, 3), torch.zeros(1, 2, 4))
            self.assertEqual((2, 1, 1), tuple(output.shape))
            self.assertEqual((1, 2, 4), tuple(hidden.shape))


if __name__ == "__main__":
    unittest.main()
