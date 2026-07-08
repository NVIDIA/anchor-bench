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



def _write_two_joint_sage_motion(root: Path, length: int = 4) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "joint_list.txt").write_text("Rotation\nPitch\n", encoding="utf-8")
    (root / "metadata.yaml").write_text(
        "schema: sage\ntime_unit: seconds\nposition_unit: rad\n"
        "velocity_unit: rad/s\ntorque_unit: N*m\n",
        encoding="utf-8",
    )
    control = ["type,timestamp,positions"]
    state = ["type,timestamp,positions,velocities,torques"]
    for index in range(length):
        timestamp = index * 0.002
        control.append(f'CONTROL,{timestamp},"[{1.0 + index}, {2.0 + index}]"')
        state.append(
            f'STATE_MOTOR,{timestamp},"[{0.1 + index}, {0.2 + index}]",'
            f'"[{0.3 + index}, {0.4 + index}]","[{10.0 + index}, {20.0 + index}]"'
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
            header = read_training_header(expanded[1], "Rotation")
            table = load_training_table(
                sources[0],
                joint="Pitch",
                columns=["position", "position_error", "velocity", "torque"],
            )

        self.assertEqual(1, len(sources))
        self.assertEqual(["Rotation", "Pitch"], [source.joint for source in expanded])
        self.assertEqual(
            ["Rotation", "Pitch"],
            [source.joint for source in train_sources],
        )
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


class TrainingFeatureBlockTests(unittest.TestCase):
    @staticmethod
    def _job(
        input_columns,
        *,
        model="gru",
        residual_physics=None,
        input_joints=("j0", "j1"),
        output_joints=("j1",),
    ):
        from anchor_bench.training.trainer import TrainingJob

        return TrainingJob(
            model=model,
            train_sources=[],
            val_sources=[],
            output_dir=Path("out"),
            input_columns=list(input_columns),
            target_columns=[
                "torque_residual" if model == "hybrid_residual" else "torque"
            ],
            source_target_columns=["torque"],
            seed=1,
            final_epochs=1,
            trial_epochs=1,
            trials=1,
            skip_optuna=True,
            n_streams=1,
            noise_std=0.0,
            residual_physics=residual_physics,
            input_joints=input_joints,
            output_joints=output_joints,
            sample_dt_s=0.01,
        )

    @staticmethod
    def _episode(feature_blocks, target):
        import numpy as np

        from anchor_bench.training.data import TrainingEpisode

        return TrainingEpisode(
            timestamps=np.arange(len(target), dtype=np.float64) * 0.01,
            input_joints=("j0", "j1"),
            output_joints=("j1",),
            feature_blocks=feature_blocks,
            target_torque=np.asarray(target, dtype=np.float32),
        )

    def test_prepare_flattens_vector_joint_and_global_features(self):
        import numpy as np

        from anchor_bench.training.data import FeatureBlock
        from anchor_bench.training.trainer import _feature_widths, _prepare_episode

        job = self._job(["velocity", "position", "previous_torque"])
        episode = self._episode(
            {
                "velocity": FeatureBlock(
                    np.array(
                        [
                            [[1, 2, 3], [4, 5, 6]],
                            [[11, 12, 13], [14, 15, 16]],
                        ],
                        dtype=np.float32,
                    ),
                    domain="input_joints",
                    channels=("vx", "vy", "vz"),
                ),
                "position": FeatureBlock(
                    np.array(
                        [
                            [[7, 8, 9]],
                            [[17, 18, 19]],
                        ],
                        dtype=np.float32,
                    ),
                    domain="global",
                    channels=("x", "y", "z"),
                ),
            },
            [[10], [20]],
        )

        features, target = _prepare_episode(job, episode)

        np.testing.assert_allclose(
            features,
            [
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
                [11, 12, 13, 14, 15, 16, 17, 18, 19, 10],
            ],
        )
        np.testing.assert_allclose(target, [[10], [20]])
        self.assertEqual((6, 3, 1), _feature_widths(job))
        self.assertEqual(
            [
                {
                    "name": "velocity",
                    "domain": "input_joints",
                    "channels": ["vx", "vy", "vz"],
                },
                {
                    "name": "position",
                    "domain": "global",
                    "channels": ["x", "y", "z"],
                },
                {
                    "name": "previous_torque",
                    "domain": "output_joints",
                    "channels": ["value"],
                },
            ],
            job.metadata["input_feature_specs"],
        )

    def test_scalar_feature_blocks_preserve_feature_major_layout(self):
        import numpy as np

        from anchor_bench.training.data import FeatureBlock
        from anchor_bench.training.trainer import _prepare_episode

        job = self._job(["position"])
        episode = self._episode(
            {
                "position": FeatureBlock(
                    np.array([[[1], [2]], [[3], [4]]], dtype=np.float32),
                    domain="input_joints",
                    channels=("value",),
                )
            },
            [[10], [20]],
        )

        features, _ = _prepare_episode(job, episode)

        np.testing.assert_allclose(features, [[1, 2], [3, 4]])
        self.assertEqual(
            [
                {
                    "name": "position",
                    "domain": "input_joints",
                    "channels": ["value"],
                }
            ],
            job.metadata["input_feature_specs"],
        )

    def test_prepare_rejects_feature_layout_drift_between_episodes(self):
        import numpy as np

        from anchor_bench.training.data import FeatureBlock
        from anchor_bench.training.trainer import _prepare_episode

        job = self._job(["velocity"])
        first = self._episode(
            {
                "velocity": FeatureBlock(
                    np.zeros((2, 2, 2), dtype=np.float32),
                    domain="input_joints",
                    channels=("vx", "vy"),
                )
            },
            [[10], [20]],
        )
        second = self._episode(
            {
                "velocity": FeatureBlock(
                    np.zeros((2, 2, 2), dtype=np.float32),
                    domain="input_joints",
                    channels=("vy", "vx"),
                )
            },
            [[30], [40]],
        )

        _prepare_episode(job, first, source_name="first")
        with self.assertRaisesRegex(ValueError, "layout differs"):
            _prepare_episode(job, second, source_name="second")

    def test_vector_feature_normalization_and_metadata_use_expanded_widths(self):
        import numpy as np

        from anchor_bench.training.data import FeatureBlock
        from anchor_bench.training.trainer import (
            _feature_widths,
            _normalization_stats,
            _normalize_sequences,
            _prepare_episode,
            _runtime_metadata,
            _training_stats_metadata,
        )

        job = self._job(["velocity", "position", "previous_torque"])
        episode = self._episode(
            {
                "velocity": FeatureBlock(
                    np.array(
                        [
                            [[1, 2, 3], [4, 5, 6]],
                            [[11, 12, 13], [14, 15, 16]],
                        ],
                        dtype=np.float32,
                    ),
                    domain="input_joints",
                    channels=("vx", "vy", "vz"),
                ),
                "position": FeatureBlock(
                    np.array([[[7, 8, 9]], [[17, 18, 19]]], dtype=np.float32),
                    domain="global",
                    channels=("x", "y", "z"),
                ),
            },
            [[10], [20]],
        )
        sequence = _prepare_episode(job, episode)
        widths = _feature_widths(job)
        stats = _normalization_stats(
            [sequence],
            input_columns=job.input_columns,
            target_columns=job.target_columns,
            metadata=_training_stats_metadata(job),
            input_widths=widths,
            target_width=1,
        )
        normalized = _normalize_sequences(
            [sequence],
            stats,
            input_columns=job.input_columns,
            target_columns=job.target_columns,
            input_widths=widths,
            target_width=1,
        )
        runtime = _runtime_metadata(job, stats)

        self.assertEqual(
            [6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
            stats["normalization"]["inputs"]["mean"]["velocity"],
        )
        self.assertEqual(
            [12.0, 13.0, 14.0],
            stats["normalization"]["inputs"]["mean"]["position"],
        )
        self.assertEqual(
            5.0,
            stats["normalization"]["inputs"]["mean"]["previous_torque"],
        )
        np.testing.assert_allclose(
            normalized[0][0],
            np.array([[-1.0] * 10, [1.0] * 10], dtype=np.float32),
        )
        np.testing.assert_allclose(normalized[0][1], [[-1.0], [1.0]])
        self.assertEqual(
            job.metadata["input_feature_specs"],
            stats["input_feature_specs"],
        )
        self.assertEqual(
            job.metadata["input_feature_specs"],
            runtime["input_feature_specs"],
        )

    def test_runtime_metadata_requires_current_stats_feature_specs(self):
        from anchor_bench.training.trainer import _runtime_metadata

        job = self._job(["position"])
        job.metadata["input_feature_specs"] = [
            {
                "name": "position",
                "domain": "input_joints",
                "channels": ["value"],
            }
        ]
        normalization = {
            "inputs": {
                "mean": {"position": [0.0, 0.0]},
                "std": {"position": [1.0, 1.0]},
            },
            "targets": {
                "mean": {"torque": 0.0},
                "std": {"torque": 1.0},
            },
        }
        cases = [
            (
                "missing",
                {"normalization": normalization},
                "missing input_feature_specs",
            ),
            (
                "null",
                {
                    "normalization": normalization,
                    "input_feature_specs": None,
                },
                "must be a list",
            ),
            (
                "mismatched",
                {
                    "normalization": normalization,
                    "input_feature_specs": [
                        {
                            "name": "position",
                            "domain": "global",
                            "channels": ["value"],
                        }
                    ],
                },
                "do not match",
            ),
        ]
        for label, stats, message in cases:
            with self.subTest(label=label), self.assertRaisesRegex(
                ValueError,
                message,
            ):
                _runtime_metadata(job, stats)

    def test_feature_specs_reject_null_and_invalid_previous_torque_layout(self):
        from anchor_bench.training.trainer import _feature_widths

        null_job = self._job(["position"])
        null_job.metadata["input_feature_specs"] = None
        with self.assertRaisesRegex(ValueError, "must be a list"):
            _feature_widths(null_job)

        previous_job = self._job(["position", "previous_torque"])
        previous_job.metadata["input_feature_specs"] = [
            {
                "name": "position",
                "domain": "input_joints",
                "channels": ["value"],
            },
            {
                "name": "previous_torque",
                "domain": "input_joints",
                "channels": ["value"],
            },
        ]
        with self.assertRaisesRegex(ValueError, "output_joints"):
            _feature_widths(previous_job)

    def test_physics_derivation_rejects_multichannel_inputs(self):
        import numpy as np

        from anchor_bench.training.data import FeatureBlock
        from anchor_bench.training.trainer import (
            ResidualPhysics,
            _prepare_episode,
        )

        job = self._job(
            ["solver_pd"],
            model="hybrid_residual",
            residual_physics=ResidualPhysics(stiffness=2.0, damping=0.5),
        )
        episode = self._episode(
            {
                "position_error": FeatureBlock(
                    np.zeros((2, 2, 3), dtype=np.float32),
                    domain="input_joints",
                    channels=("x", "y", "z"),
                ),
                "velocity": FeatureBlock(
                    np.zeros((2, 2, 1), dtype=np.float32),
                    domain="input_joints",
                    channels=("value",),
                ),
            },
            [[10], [20]],
        )

        with self.assertRaisesRegex(ValueError, "one scalar channel"):
            _prepare_episode(job, episode)


class TrainingRunnerTests(unittest.TestCase):
    def test_training_config_resolves_run_yaml_defaults(self):
        from anchor_bench.core.schemas import TrainActuatorRunConfig
        from anchor_bench.training.runner import resolve_train_actuator_run_config

        resolved = resolve_train_actuator_run_config(TrainActuatorRunConfig(robot="so101", dry_run=True))

        self.assertEqual("so101", resolved.robot)
        self.assertEqual("gru", resolved.model)
        self.assertTrue(resolved.data.endswith("so101_arm_50motion/train"))
        self.assertTrue(resolved.output.endswith("outputs/training/so101"))
        self.assertEqual(
            ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"],
            resolved.shared_siso_joints,
        )
        self.assertIsNone(resolved.input_joints)
        self.assertIsNone(resolved.output_joints)
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
                input_joints=["Rotation"],
                data="inputs/datasets/so101/custom_figure8_with_camera",
                input_columns=["position", "solver_pd"],
                target_column="torque",
                dry_run=True,
            )
        )

        self.assertEqual(["position", "solver_pd"], resolved.input_columns)
        self.assertEqual("torque", resolved.target_column)

    def test_shared_dynamic_bias_is_evaluated_once_then_sliced_in_source_order(self):
        import numpy as np

        from anchor_bench.training.data import TrainingSource
        from anchor_bench.training.dynamic_bias import DynamicBiasResult
        from anchor_bench.training.trainer import TrainingJob, _load_sequences

        class FakeBiasProvider:
            def __init__(self):
                self.calls = []

            def compute(
                self,
                *,
                joint_names,
                positions,
                velocities,
                requested_joint_names,
            ):
                requested = list(requested_joint_names)
                self.calls.append(
                    {
                        "joint_names": list(joint_names),
                        "requested_joint_names": requested,
                        "positions": np.array(positions, copy=True),
                        "velocities": np.array(velocities, copy=True),
                    }
                )
                values = np.column_stack(
                    [
                        np.full(len(positions), 20.0 if joint == "Pitch" else 10.0)
                        for joint in requested
                    ]
                )
                return DynamicBiasResult(
                    values,
                    {
                        "source": "fake",
                        "trajectory_joint_names": list(joint_names),
                        "requested_joint_names": requested,
                        "mapped_joint_names": list(joint_names),
                        "unmapped_trajectory_joint_names": [],
                        "defaulted_joint_names": [],
                    },
                )

        with tempfile.TemporaryDirectory() as tmp:
            episode_path = Path(tmp) / "episode"
            _write_two_joint_sage_motion(episode_path)
            provider = FakeBiasProvider()
            sources = [
                TrainingSource(episode_path, "Rotation"),
                TrainingSource(episode_path, "Pitch"),
            ]
            job = TrainingJob(
                model="gru",
                train_sources=sources,
                val_sources=[],
                output_dir=Path(tmp) / "out",
                input_columns=["dynamic_bias"],
                target_columns=["torque"],
                source_target_columns=["torque"],
                seed=1,
                final_epochs=1,
                trial_epochs=1,
                trials=1,
                skip_optuna=True,
                n_streams=1,
                noise_std=0.0,
                shared_siso_joints=["Rotation", "Pitch"],
                sample_dt_s=0.002,
                dynamic_bias_provider=provider,
            )
            train, val = _load_sequences(job)

        self.assertEqual([], val)
        self.assertEqual(1, len(provider.calls))
        self.assertEqual(
            ["Rotation", "Pitch"],
            provider.calls[0]["requested_joint_names"],
        )
        self.assertEqual(2, len(train))
        np.testing.assert_allclose(train[0][0][:, 0], 10.0)
        np.testing.assert_allclose(train[1][0][:, 0], 20.0)
        np.testing.assert_allclose(train[0][1][:, 0], [10.0, 11.0, 12.0, 13.0])
        np.testing.assert_allclose(train[1][1][:, 0], [20.0, 21.0, 22.0, 23.0])
        self.assertEqual(
            ["Rotation", "Pitch"],
            job.metadata["dynamic_bias_derivation"]["requested_joint_names"],
        )

    def test_artifact_labels_distinguish_topologies_and_joint_domains(self):
        from anchor_bench.training.data import TrainingSource
        from anchor_bench.training.trainer import TrainingJob, _artifact_label

        def job(**selection):
            return TrainingJob(
                model="gru",
                train_sources=[TrainingSource(Path("episode"))],
                val_sources=[],
                output_dir=Path("out"),
                input_columns=["position"],
                target_columns=["torque"],
                source_target_columns=["torque"],
                seed=1,
                final_epochs=1,
                trial_epochs=1,
                trials=1,
                skip_optuna=True,
                n_streams=1,
                noise_std=0.0,
                sample_dt_s=0.01,
                **selection,
            )

        jobs = [
            job(input_joints=["joint_a"]),
            job(input_joints=["joint_a", "joint_b"], output_joints=["joint_a"]),
            job(input_joints=["joint_a", "joint_b"]),
            job(shared_siso_joints=["joint_a", "joint_b"]),
            job(shared_siso_joints=["joint_a", "joint_c"]),
            job(input_joints=["joint_" + "x" * 200]),
        ]
        labels = [_artifact_label(item) for item in jobs]

        self.assertEqual(len(labels), len(set(labels)))
        self.assertEqual(labels[0], _artifact_label(jobs[0]))
        self.assertIn("siso", labels[0])
        self.assertIn("miso", labels[1])
        self.assertIn("mimo", labels[2])
        self.assertIn("shared_siso", labels[3])
        self.assertTrue(all(len(label) <= 96 for label in labels))

    def test_stats_metadata_uses_resolved_runtime_axes(self):
        from anchor_bench.training.data import TrainingSource
        from anchor_bench.training.trainer import (
            TrainingJob,
            _training_stats_metadata,
        )

        base = {
            "model": "gru",
            "train_sources": [TrainingSource(Path("episode"))],
            "val_sources": [],
            "output_dir": Path("out"),
            "input_columns": ["position"],
            "target_columns": ["torque"],
            "source_target_columns": ["torque"],
            "seed": 1,
            "final_epochs": 1,
            "trial_epochs": 1,
            "trials": 1,
            "skip_optuna": True,
            "n_streams": 1,
            "noise_std": 0.0,
            "sample_dt_s": 0.01,
            "metadata": {
                "input_feature_specs": [
                    {
                        "name": "position",
                        "domain": "input_joints",
                        "channels": ["value"],
                    }
                ]
            },
        }
        explicit = TrainingJob(**base, input_joints=["joint_a", "joint_b"])
        shared = TrainingJob(**base, shared_siso_joints=["joint_b", "joint_a"])

        explicit_metadata = _training_stats_metadata(explicit)
        shared_metadata = _training_stats_metadata(shared)
        self.assertEqual(["joint_a", "joint_b"], explicit_metadata["input_joints"])
        self.assertEqual(["joint_a", "joint_b"], explicit_metadata["output_joints"])
        self.assertEqual(["actuator"], shared_metadata["input_joints"])
        self.assertEqual(["actuator"], shared_metadata["output_joints"])
        self.assertEqual(
            ["joint_b", "joint_a"],
            shared_metadata["shared_siso_joints"],
        )

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
                    input_joints=["Rotation"],
                    skip_optuna=True,
                    final_epochs=1,
                    n_streams=1,
                    seed=5,
                    dry_run=False,
                )
            )
            plan = run_resolved_train_actuator(resolved)

            self.assertFalse(plan.dry_run)
            artifacts = plan.metadata["artifacts"]
            self.assertTrue(Path(artifacts["checkpoint"]).exists())
            self.assertTrue(Path(artifacts["network"]).exists())
            self.assertRegex(
                Path(artifacts["stats"]).name,
                r"^gru_siso__in__Rotation__out__Rotation__[0-9a-f]{10}_stats.json$",
            )
            stats = json.loads(Path(artifacts["stats"]).read_text(encoding="utf-8"))
            self.assertEqual("gru", stats["model"])
            self.assertEqual("siso", stats["training_topology"])
            self.assertIsNone(stats["shared_siso_joints"])
            self.assertEqual(["Rotation"], stats["input_joints"])
            self.assertEqual(["Rotation"], stats["output_joints"])
            self.assertNotIn("joint_type", stats)
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
