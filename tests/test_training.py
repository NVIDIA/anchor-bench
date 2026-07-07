from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


def _write_sage_motion(root: Path, length: int = 8) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "joint_list.txt").write_text("Rotation\n", encoding="utf-8")
    (root / "metadata.yaml").write_text(
        "schema: sage\ntime_unit: seconds\nposition_unit: rad\n"
        "velocity_unit: rad/s\ntorque_unit: N*m\n",
        encoding="utf-8",
    )
    control = ["type,timestamp,positions"]
    state = ["type,timestamp,positions,velocities,torques"]
    for index in range(length):
        timestamp = index * 0.002
        position = 0.1 * index
        command = 1.0
        velocity = 0.05 * index
        torque = 2.0 * (command - position) - 0.25 * velocity
        control.append(f'CONTROL,{timestamp},"[{command}]"')
        state.append(
            f'STATE_MOTOR,{timestamp},"[{position}]","[{velocity}]","[{torque}]"'
        )
    (root / "control.csv").write_text("\n".join(control) + "\n", encoding="utf-8")
    (root / "state_motor.csv").write_text("\n".join(state) + "\n", encoding="utf-8")


class TrainingDataTests(unittest.TestCase):
    def test_sage_discovery_and_split_are_deterministic(self):
        from anchor_bench.training.data import (
            discover_training_sources,
            expand_joint_sources,
            split_training_sources,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index in range(5):
                _write_sage_motion(root / f"motion_{index}")

            sources = discover_training_sources(root)
            expanded = expand_joint_sources(sources, ["Rotation"])
            train_a, val_a = split_training_sources(
                expanded, seed=123, val_fraction=0.4
            )
            train_b, val_b = split_training_sources(
                expanded, seed=123, val_fraction=0.4
            )

        self.assertEqual(5, len(sources))
        self.assertEqual(train_a, train_b)
        self.assertEqual(val_a, val_b)
        self.assertEqual(3, len(train_a))
        self.assertEqual(2, len(val_a))

    def test_sage_motion_is_one_training_source_with_derived_joint_columns(self):
        import numpy as np

        from anchor_bench.training.data import (
            discover_training_sources,
            expand_joint_sources,
            load_training_table,
            read_training_header,
            split_training_sources,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "joint_list.txt").write_text("Rotation\nPitch\n", encoding="utf-8")
            (root / "metadata.yaml").write_text(
                "schema: sage\ntime_unit: seconds\nposition_unit: rad\nvelocity_unit: rad/s\ntorque_unit: N*m\n",
                encoding="utf-8",
            )
            (root / "control.csv").write_text(
                'type,timestamp,positions\nCONTROL,0.0,"[1.0, 2.0]"\nCONTROL,1.0,"[1.5, 2.5]"\n',
                encoding="utf-8",
            )
            (root / "state_motor.csv").write_text(
                "type,timestamp,positions,velocities,torques\n"
                'STATE_MOTOR,0.0,"[0.5, 1.5]","[0.1, 0.2]","[3.0, 4.0]"\n'
                'STATE_MOTOR,1.0,"[1.0, 2.0]","[0.3, 0.4]","[5.0, 6.0]"\n',
                encoding="utf-8",
            )

            sources = discover_training_sources(root)
            expanded = expand_joint_sources(sources, ["Rotation", "Pitch"])
            train_sources, val_sources = split_training_sources(expanded, seed=7)
            header = read_training_header(expanded[1], "full")
            table = load_training_table(
                sources[0],
                joint="Pitch",
                columns=["position", "position_error", "velocity", "torque"],
            )

        self.assertEqual(1, len(sources))
        self.assertEqual(["Rotation", "Pitch"], [source.joint for source in expanded])
        self.assertEqual(2, len(train_sources))
        self.assertEqual([], val_sources)
        self.assertIn("position_error", header)
        np.testing.assert_allclose(table["position"], [1.5, 2.0])
        np.testing.assert_allclose(table["position_error"], [0.5, 0.5])
        np.testing.assert_allclose(table["velocity"], [0.2, 0.4])
        np.testing.assert_allclose(table["torque"], [4.0, 6.0])

    def test_column_resolution_uses_model_defaults_and_validates_headers(self):
        from anchor_bench.training.data import resolve_training_columns

        header = ["position", "position_error", "velocity", "torque"]
        pure = resolve_training_columns(header, model="gru")
        self.assertEqual(["position", "position_error", "velocity"], pure.input_columns)
        self.assertEqual(["torque"], pure.target_columns)

        hybrid = resolve_training_columns(header, model="hybrid_residual")
        self.assertEqual(["position", "position_error", "velocity"], hybrid.input_columns)
        self.assertEqual(["torque_residual"], hybrid.target_columns)

        with self.assertRaisesRegex(ValueError, "missing required training signals: velocity"):
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

        resolved = resolve_train_actuator_run_config(TrainActuatorRunConfig(robot="so101", dry_run=True))

        self.assertEqual("so101", resolved.robot)
        self.assertEqual("gru", resolved.model)
        self.assertTrue(resolved.data.endswith("so101_arm_50motion/train"))
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
                joint_type="Rotation",
                data="inputs/datasets/so101/custom_figure8_with_camera",
                input_columns=["position", "solver_pd"],
                target_column="torque",
                dry_run=True,
            )
        )

        self.assertEqual(["position", "solver_pd"], resolved.input_columns)
        self.assertEqual("torque", resolved.target_column)

    @unittest.skipUnless(
        importlib.util.find_spec("torch"),
        "training smoke test requires torch",
    )
    def test_tiny_gru_training_smoke_writes_checkpoint_and_stats(self):
        from anchor_bench.core.schemas import TrainActuatorRunConfig
        from anchor_bench.training.runner import resolve_train_actuator_run_config, run_resolved_train_actuator

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            output_dir = root / "out"
            _write_sage_motion(data_dir / "a", length=12)
            _write_sage_motion(data_dir / "b", length=12)

            resolved = resolve_train_actuator_run_config(
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
            plan = run_resolved_train_actuator(resolved)

            self.assertFalse(plan.dry_run)
            self.assertTrue((output_dir / "gru_Rotation_best.pt").exists())
            self.assertTrue((output_dir / "gru_Rotation_network.pt").exists())
            stats = json.loads((output_dir / "gru_Rotation_stats.json").read_text(encoding="utf-8"))
            self.assertEqual("gru", stats["model"])
            self.assertEqual("Rotation", stats["joint_type"])
            self.assertEqual(0.002, stats["sample_dt_s"])
            self.assertIn("network", plan.metadata["artifacts"])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "TorchScript export test requires torch")
    def test_gru_checkpoint_export_writes_scripted_network_with_gru_attribute(self):
        from anchor_bench.training.export import build_runtime_gru, export_gru_checkpoint

        import torch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint_path = root / "gru_demo_best.pt"
            model = build_runtime_gru(input_size=3, output_size=1, hidden_size=4, num_layers=1, dropout=0.0)
            runtime_metadata = {
                "schema_version": 1,
                "model_type": "gru",
                "joint_names": ["demo"],
                "output_joint_names": ["demo"],
                "input_columns": ["position", "position_error", "velocity"],
                "target_columns": ["torque"],
                "sample_dt_s": 0.002,
                "normalization": {
                    "inputs": {
                        "mean": {
                            "position": 0.0,
                            "position_error": 0.0,
                            "velocity": 0.0,
                        },
                        "std": {
                            "position": 1.0,
                            "position_error": 1.0,
                            "velocity": 1.0,
                        },
                    },
                    "targets": {
                        "mean": {"torque": 0.0},
                        "std": {"torque": 1.0},
                    },
                },
                "delay": {
                    "handling": "learned",
                    "training_delay_s": 0.0,
                    "external_delay_s": 0.0,
                },
            }
            torch.save(
                {
                    "model": "gru",
                    "joint_type": "demo",
                    "model_state_dict": model.state_dict(),
                    "input_columns": ["position", "position_error", "velocity"],
                    "target_columns": ["torque"],
                    "input_size": 3,
                    "output_size": 1,
                    "normalization": runtime_metadata["normalization"],
                    "runtime_metadata": runtime_metadata,
                    "hyperparameters": {"hidden_size": 4, "num_layers": 1, "dropout": 0.0},
                },
                checkpoint_path,
            )

            network_path = export_gru_checkpoint(checkpoint_path)
            extra_files = {"metadata.json": ""}
            network = torch.jit.load(str(network_path), _extra_files=extra_files)
            self.assertTrue(hasattr(network, "gru"))
            embedded = json.loads(extra_files["metadata.json"].decode("utf-8"))
            self.assertEqual(runtime_metadata, embedded)
            output, hidden = network(torch.zeros(2, 1, 3), torch.zeros(1, 2, 4))
            self.assertEqual((2, 1, 1), tuple(output.shape))
            self.assertEqual((1, 2, 4), tuple(hidden.shape))


if __name__ == "__main__":
    unittest.main()
