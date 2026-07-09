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
        "schema: sage\ntime_unit: seconds\nposition_unit: rad\nvelocity_unit: rad/s\ntorque_unit: N*m\n",
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
        state.append(f'STATE_MOTOR,{timestamp},"[{position}]","[{velocity}]","[{torque}]"')
    (root / "control.csv").write_text("\n".join(control) + "\n", encoding="utf-8")
    (root / "state_motor.csv").write_text("\n".join(state) + "\n", encoding="utf-8")


def _write_two_joint_sage_motion(root: Path, length: int = 4) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "joint_list.txt").write_text("Rotation\nPitch\n", encoding="utf-8")
    (root / "metadata.yaml").write_text(
        "schema: sage\ntime_unit: seconds\nposition_unit: rad\nvelocity_unit: rad/s\ntorque_unit: N*m\n",
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
            split_training_sources,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index in range(5):
                _write_sage_motion(root / f"motion_{index}")

            sources = discover_training_sources(root)
            train_a, val_a = split_training_sources(sources, seed=123, val_fraction=0.4)
            train_b, val_b = split_training_sources(sources, seed=123, val_fraction=0.4)

        self.assertEqual(5, len(sources))
        self.assertEqual(train_a, train_b)
        self.assertEqual(val_a, val_b)
        self.assertEqual(3, len(train_a))
        self.assertEqual(2, len(val_a))


class TrainingRecurrenceTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch unavailable")
    def test_hidden_state_resets_between_sequences_but_not_chunks(self):
        import numpy as np
        import torch

        from anchor_bench.training.trainer import (
            TrainingHyperParameters,
            _run_epoch,
        )

        class RecordingModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.hidden_was_none = []

            def forward(self, values, hidden):
                self.hidden_was_none.append(hidden is None)
                next_hidden = torch.ones(
                    1,
                    values.shape[0],
                    1,
                    dtype=values.dtype,
                    device=values.device,
                )
                return values[..., :1], next_hidden

        model = RecordingModel()
        sequences = [
            (
                np.ones((3, 1), dtype=np.float32),
                np.ones((3, 1), dtype=np.float32),
            ),
            (
                np.ones((3, 1), dtype=np.float32),
                np.ones((3, 1), dtype=np.float32),
            ),
        ]
        loss = _run_epoch(
            torch,
            model,
            optimizer=None,
            loss_fn=torch.nn.MSELoss(),
            sequences=sequences,
            hparams=TrainingHyperParameters(chunk_length=2),
            train=False,
            device=torch.device("cpu"),
            noise_std=0.0,
        )

        self.assertEqual(0.0, loss)
        self.assertEqual([True, False, True, False], model.hidden_was_none)


class TrainingFeatureBlockTests(unittest.TestCase):
    @staticmethod
    def _job(
        input_columns,
        *,
        model="gru",
        residual_physics=None,
        input_joints=("j0", "j1"),
        output_joints=("j1",),
        mappings=None,
    ):
        from anchor_bench.training.data import TrainingMapping
        from anchor_bench.training.trainer import TrainingJob

        return TrainingJob(
            model=model,
            train_sources=[],
            val_sources=[],
            output_dir=Path("out"),
            input_columns=list(input_columns),
            seed=1,
            final_epochs=1,
            trial_epochs=1,
            trials=1,
            skip_optuna=True,
            n_streams=1,
            noise_std=0.0,
            residual_physics=residual_physics,
            group_name="test_group",
            mappings=(
                tuple(mappings)
                if mappings is not None
                else (TrainingMapping(tuple(input_joints), tuple(output_joints)),)
            ),
            sample_dt_s=0.01,
        )

    @staticmethod
    def _episode(feature_blocks, target):
        import numpy as np

        from anchor_bench.training.data import TrainingEpisode, TrainingMapping

        return TrainingEpisode(
            timestamps=np.arange(len(target), dtype=np.float64) * 0.01,
            mapping=TrainingMapping(("j0", "j1"), ("j1",)),
            feature_blocks=feature_blocks,
            target_torque=np.asarray(target, dtype=np.float32),
        )

    def test_training_job_rejects_mixed_width_and_duplicate_mappings(self):
        from anchor_bench.training.data import TrainingMapping

        first = TrainingMapping(("j0",), ("j0",))
        with self.assertRaisesRegex(ValueError, "mapping 1 has dimensions"):
            self._job(
                ["position"],
                mappings=(first, TrainingMapping(("j1", "j2"), ("j1",))),
            )
        with self.assertRaisesRegex(ValueError, "must not contain duplicates"):
            self._job(["position"], mappings=(first, first))

    def test_prepare_flattens_vector_input_features_and_output_previous_torque(self):
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
                            [[7, 8, 9], [10, 11, 12]],
                            [[17, 18, 19], [20, 21, 22]],
                        ],
                        dtype=np.float32,
                    ),
                    domain="input_joints",
                    channels=("x", "y", "z"),
                ),
            },
            [[10], [20]],
        )

        features, target = _prepare_episode(job, episode)

        np.testing.assert_allclose(
            features,
            [
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0],
                [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 10],
            ],
        )
        np.testing.assert_allclose(target, [[10], [20]])
        self.assertEqual((6, 6, 1), _feature_widths(job))
        self.assertEqual(
            [
                {
                    "name": "velocity",
                    "domain": "input_joints",
                    "channels": ["vx", "vy", "vz"],
                },
                {
                    "name": "position",
                    "domain": "input_joints",
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

    def test_prepare_rejects_output_joint_domain_for_non_previous_feature(self):
        import numpy as np

        from anchor_bench.training.data import FeatureBlock
        from anchor_bench.training.trainer import _prepare_episode

        job = self._job(["position"])
        episode = self._episode(
            {
                "position": FeatureBlock(
                    np.zeros((2, 1, 3), dtype=np.float32),
                    domain="output_joints",
                    channels=("x", "y", "z"),
                )
            },
            [[10], [20]],
        )

        with self.assertRaisesRegex(
            ValueError,
            r"feature 'position' must use domain 'input_joints'",
        ):
            _prepare_episode(job, episode)

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
                    np.array(
                        [
                            [[7, 8, 9], [10, 11, 12]],
                            [[17, 18, 19], [20, 21, 22]],
                        ],
                        dtype=np.float32,
                    ),
                    domain="input_joints",
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
            target_column="torque",
            metadata=_training_stats_metadata(job),
            input_widths=widths,
            target_width=1,
        )
        normalized = _normalize_sequences(
            [sequence],
            stats,
            input_columns=job.input_columns,
            target_column="torque",
            input_widths=widths,
            target_width=1,
        )
        runtime = _runtime_metadata(job, stats)

        self.assertEqual(2, stats["schema_version"])
        self.assertEqual(2, runtime["schema_version"])
        self.assertEqual("test_group", runtime["group_name"])
        self.assertEqual(
            [{"input_joints": ["j0", "j1"], "output_joints": ["j1"]}],
            runtime["joint_mappings"],
        )
        self.assertNotIn("joint_names", runtime)
        self.assertNotIn("output_joint_names", runtime)
        self.assertEqual(
            ["recorded_episode", "joint_mapping"],
            runtime["previous_torque_derivation"]["reset_boundaries"],
        )

        self.assertEqual(
            [6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
            stats["normalization"]["inputs"]["mean"]["velocity"],
        )
        self.assertEqual(
            [12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
            stats["normalization"]["inputs"]["mean"]["position"],
        )
        self.assertEqual(
            5.0,
            stats["normalization"]["inputs"]["mean"]["previous_torque"],
        )
        np.testing.assert_allclose(
            normalized[0][0],
            np.array([[-1.0] * 13, [1.0] * 13], dtype=np.float32),
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
        self.assertEqual(1, len(resolved.groups))
        group = resolved.groups[0]
        self.assertEqual("so101_actuator", group.name)
        self.assertEqual(
            ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"],
            [mapping.input_joints[0] for mapping in group.mappings],
        )
        self.assertTrue(all(mapping.input_joints == mapping.output_joints for mapping in group.mappings))
        self.assertTrue(resolved.skip_optuna)
        self.assertEqual(42, resolved.seed)
        self.assertEqual(["position", "position_error", "velocity"], resolved.input_columns)

    def test_training_config_resolves_cli_column_overrides(self):
        from anchor_bench.core.schemas import TrainActuatorRunConfig
        from anchor_bench.training.runner import resolve_train_actuator_run_config

        resolved = resolve_train_actuator_run_config(
            TrainActuatorRunConfig(
                robot="so101",
                input_joints=["Rotation"],
                data="inputs/datasets/so101/custom_figure8_with_camera",
                input_columns=["position", "solver_pd"],
                dry_run=True,
            )
        )

        self.assertEqual(["position", "solver_pd"], resolved.input_columns)

    def test_shared_siso_expands_mappings_and_records_dynamic_bias_requests(self):
        import numpy as np

        from anchor_bench.training.data import TrainingMapping, TrainingSource
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
                    [np.full(len(positions), 20.0 if joint == "Pitch" else 10.0) for joint in requested]
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
            sources = [TrainingSource(episode_path)]
            job = TrainingJob(
                model="gru",
                train_sources=sources,
                val_sources=[],
                output_dir=Path(tmp) / "out",
                input_columns=["dynamic_bias"],
                seed=1,
                final_epochs=1,
                trial_epochs=1,
                trials=1,
                skip_optuna=True,
                n_streams=1,
                noise_std=0.0,
                group_name="shared_siso",
                mappings=(
                    TrainingMapping(("Rotation",), ("Rotation",)),
                    TrainingMapping(("Pitch",), ("Pitch",)),
                ),
                sample_dt_s=0.002,
                dynamic_bias_provider=provider,
            )
            train, val = _load_sequences(job)

        self.assertEqual([], val)
        self.assertEqual(2, len(provider.calls))
        self.assertEqual(
            [["Rotation"], ["Pitch"]],
            [call["requested_joint_names"] for call in provider.calls],
        )
        self.assertEqual(2, len(train))
        np.testing.assert_allclose(train[0][0][:, 0], 10.0)
        np.testing.assert_allclose(train[1][0][:, 0], 20.0)
        np.testing.assert_allclose(train[0][1][:, 0], [10.0, 11.0, 12.0, 13.0])
        np.testing.assert_allclose(train[1][1][:, 0], [20.0, 21.0, 22.0, 23.0])
        self.assertEqual(
            [
                {"mapping_index": 0, "requested_joint_names": ["Rotation"]},
                {"mapping_index": 1, "requested_joint_names": ["Pitch"]},
            ],
            job.metadata["dynamic_bias_derivation"]["mapping_requests"],
        )

    def test_shared_siso_resets_previous_torque_and_pools_normalization(self):
        import numpy as np

        from anchor_bench.training.data import TrainingMapping, TrainingSource
        from anchor_bench.training.trainer import (
            TrainingJob,
            _feature_widths,
            _load_sequences,
            _normalization_stats,
            _training_stats_metadata,
        )

        with tempfile.TemporaryDirectory() as tmp:
            episode_path = Path(tmp) / "episode"
            _write_two_joint_sage_motion(episode_path)
            job = TrainingJob(
                model="gru",
                train_sources=[TrainingSource(episode_path)],
                val_sources=[],
                output_dir=Path(tmp) / "out",
                input_columns=["position", "previous_torque"],
                group_name="shared_siso",
                mappings=(
                    TrainingMapping(("Rotation",), ("Rotation",)),
                    TrainingMapping(("Pitch",), ("Pitch",)),
                ),
                sample_dt_s=0.002,
                seed=1,
                final_epochs=1,
                trial_epochs=1,
                trials=1,
                skip_optuna=True,
                n_streams=1,
                noise_std=0.0,
            )
            train, _ = _load_sequences(job)

        self.assertEqual(2, len(train))
        np.testing.assert_allclose(train[0][0][:, 1], [0.0, 10.0, 11.0, 12.0])
        np.testing.assert_allclose(train[1][0][:, 1], [0.0, 20.0, 21.0, 22.0])
        stats = _normalization_stats(
            train,
            input_columns=job.input_columns,
            target_column="torque",
            metadata=_training_stats_metadata(job),
            input_widths=_feature_widths(job),
            target_width=1,
        )
        self.assertAlmostEqual(1.65, stats["normalization"]["inputs"]["mean"]["position"], places=6)
        self.assertAlmostEqual(12.0, stats["normalization"]["inputs"]["mean"]["previous_torque"])
        self.assertAlmostEqual(16.5, stats["normalization"]["targets"]["mean"]["torque"])

    def test_artifact_labels_include_group_and_mapping_digest(self):
        from anchor_bench.training.data import TrainingMapping, TrainingSource
        from anchor_bench.training.trainer import TrainingJob, _artifact_label

        def job(*, group_name="group", input_joints, output_joints=None):
            inputs = tuple(input_joints)
            outputs = tuple(output_joints or input_joints)
            return TrainingJob(
                model="gru",
                train_sources=[TrainingSource(Path("episode"))],
                val_sources=[],
                output_dir=Path("out"),
                input_columns=["position"],
                seed=1,
                group_name=group_name,
                mappings=(TrainingMapping(inputs, outputs),),
                final_epochs=1,
                trial_epochs=1,
                trials=1,
                skip_optuna=True,
                n_streams=1,
                noise_std=0.0,
                sample_dt_s=0.01,
            )

        jobs = [
            job(input_joints=["joint_a"]),
            job(input_joints=["joint_a", "joint_b"], output_joints=["joint_a"]),
            job(input_joints=["joint_a", "joint_b"]),
            job(group_name="other", input_joints=["joint_a"]),
            job(group_name="group_" + "x" * 200, input_joints=["joint_a"]),
        ]
        labels = [_artifact_label(item) for item in jobs]

        self.assertEqual(len(labels), len(set(labels)))
        self.assertEqual(labels[0], _artifact_label(jobs[0]))
        self.assertRegex(labels[0], r"^group__[0-9a-f]{10}$")
        self.assertRegex(labels[3], r"^other__[0-9a-f]{10}$")
        self.assertTrue(all(len(label) <= 96 for label in labels))

    def test_stats_metadata_uses_group_mappings(self):
        from anchor_bench.training.data import TrainingMapping, TrainingSource
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
        job = TrainingJob(
            **base,
            group_name="shared_siso",
            mappings=(
                TrainingMapping(("joint_a",), ("joint_a",)),
                TrainingMapping(("joint_b",), ("joint_b",)),
            ),
        )

        metadata = _training_stats_metadata(job)
        self.assertEqual("shared_siso", metadata["group_name"])
        self.assertEqual(
            [
                {"input_joints": ["joint_a"], "output_joints": ["joint_a"]},
                {"input_joints": ["joint_b"], "output_joints": ["joint_b"]},
            ],
            metadata["joint_mappings"],
        )
        self.assertNotIn("input_joints", metadata)
        self.assertNotIn("output_joints", metadata)
        self.assertEqual(2, metadata["data"]["mapping_count"])
        self.assertEqual(2, metadata["data"]["train_sequence_count"])

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
            artifact_groups = plan.metadata["artifacts"]
            artifacts = artifact_groups["default"]
            self.assertTrue(Path(artifacts["checkpoint"]).exists())
            self.assertTrue(Path(artifacts["network"]).exists())
            self.assertRegex(
                Path(artifacts["stats"]).name,
                r"^gru_default__[0-9a-f]{10}_stats\.json$",
            )
            stats = json.loads(Path(artifacts["stats"]).read_text(encoding="utf-8"))
            self.assertEqual("gru", stats["model"])
            self.assertEqual("default", stats["group_name"])
            self.assertEqual("siso", stats["training_topology"])
            self.assertEqual(
                [{"input_joints": ["Rotation"], "output_joints": ["Rotation"]}],
                stats["joint_mappings"],
            )
            self.assertNotIn("joint_type", stats)
            self.assertNotIn("input_joints", stats)
            self.assertNotIn("output_joints", stats)
            self.assertEqual(0.002, stats["sample_dt_s"])
            self.assertIn("default", artifact_groups)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "TorchScript export test requires torch")
    def test_gru_checkpoint_export_writes_scripted_network_with_gru_attribute(self):
        from anchor_bench.training.export import build_runtime_gru, export_gru_checkpoint

        import torch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint_path = root / "gru_demo_best.pt"
            model = build_runtime_gru(input_size=3, output_size=1, hidden_size=4, num_layers=1, dropout=0.0)
            runtime_metadata = {
                "schema_version": 2,
                "model_type": "gru",
                "group_name": "demo",
                "joint_mappings": [{"input_joints": ["demo"], "output_joints": ["demo"]}],
                "training_topology": "siso",
                "input_columns": ["position", "position_error", "velocity"],
                "input_feature_specs": [
                    {
                        "name": "position",
                        "domain": "input_joints",
                        "channels": ["value"],
                    },
                    {
                        "name": "position_error",
                        "domain": "input_joints",
                        "channels": ["value"],
                    },
                    {
                        "name": "velocity",
                        "domain": "input_joints",
                        "channels": ["value"],
                    },
                ],
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
                    "input_size": 3,
                    "output_size": 1,
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
