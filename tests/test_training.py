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
        position = 0.1 * index
        command = 1.0
        velocity = 0.05 * index
        torque = 2.0 * (command - position) - 0.25 * velocity
        control.append(f'CONTROL,{index},"[{command}]"')
        state.append(
            f'STATE_MOTOR,{index},"[{position}]","[{velocity}]","[{torque}]"'
        )
    (root / "control.csv").write_text("\n".join(control) + "\n", encoding="utf-8")
    (root / "state_motor.csv").write_text("\n".join(state) + "\n", encoding="utf-8")


def _write_multi_joint_motion(root: Path) -> None:
    """Write a deterministic two-joint SAGE episode for tensor tests."""

    root.mkdir(parents=True, exist_ok=True)
    (root / "joint_list.txt").write_text("JointA\nJointB\n", encoding="utf-8")
    (root / "metadata.yaml").write_text(
        "schema: sage\ntime_unit: seconds\nposition_unit: rad\n"
        "velocity_unit: rad/s\ntorque_unit: N*m\n",
        encoding="utf-8",
    )
    (root / "control.csv").write_text(
        "type,timestamp,positions\n"
        'CONTROL,0.0,"[1.0, 12.0]"\n'
        'CONTROL,2.0,"[1.0, 12.0]"\n',
        encoding="utf-8",
    )
    (root / "state_motor.csv").write_text(
        "type,timestamp,positions,velocities,torques\n"
        'STATE_MOTOR,0.0,"[0.0, 10.0]","[0.0, 1.0]","[1.0, 10.0]"\n'
        'STATE_MOTOR,1.0,"[0.1, 10.2]","[0.1, 1.2]","[2.0, 20.0]"\n'
        'STATE_MOTOR,2.0,"[0.2, 10.4]","[0.2, 1.4]","[3.0, 30.0]"\n',
        encoding="utf-8",
    )


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

    def test_mapping_episode_preserves_configured_joint_order(self):
        import numpy as np

        from anchor_bench.training.data import (
            TrainingMapping,
            TrainingSource,
            load_training_episode,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_multi_joint_motion(root)
            episode = load_training_episode(
                TrainingSource(root),
                TrainingMapping(("JointB", "JointA"), ("JointA",)),
            )

        self.assertEqual((3, 2), episode.position.shape)
        self.assertEqual((3, 1), episode.torque.shape)
        np.testing.assert_allclose(episode.position[:, 0], [10.0, 10.2, 10.4])
        np.testing.assert_allclose(episode.position_error[:, 0], [2.0, 1.8, 1.6], rtol=1e-6)
        np.testing.assert_allclose(episode.torque[:, 0], [1.0, 2.0, 3.0])

    def test_mapping_validates_subset_and_training_features(self):
        from anchor_bench.core.schemas import TrainingJointMapping
        from anchor_bench.training.data import TrainingMapping, resolve_training_columns

        for model in ("gru", "hybrid_residual"):
            columns = resolve_training_columns(
                ["position", "position_error", "velocity", "torque"],
                model=model,
                input_columns=["dynamic_bias", "previous_torque"],
            )
            self.assertEqual(["dynamic_bias", "previous_torque"], columns.input_columns)

        with self.assertRaisesRegex(ValueError, "subset"):
            TrainingMapping(("JointA",), ("JointB",))
        with self.assertRaisesRegex(ValueError, "must not contain duplicate columns"):
            resolve_training_columns(
                ["position", "torque"],
                model="gru",
                input_columns=["position", "position"],
            )
        with self.assertRaisesRegex(ValueError, "unknown keys: unexpected"):
            TrainingJointMapping.from_dict(
                {"input_joints": ["JointA"], "unexpected": True},
                "--mappings[0]",
            )


class TrainingTensorTests(unittest.TestCase):
    def _job(self, root, mapping, input_columns, *, model="gru", physics=None, mappings=None, provider=None):
        """Build the smallest TrainingJob needed for sequence assembly."""

        from anchor_bench.training.data import TrainingSource
        from anchor_bench.training.trainer import TrainingJob

        return TrainingJob(
            model=model,
            artifact_name="mapped",
            mappings=tuple(mappings or (mapping,)),
            train_sources=(TrainingSource(root),),
            val_sources=(),
            output_dir=root / "out",
            input_columns=list(input_columns),
            target_columns=["torque_residual" if model == "hybrid_residual" else "torque"],
            source_target_columns=["torque"],
            seed=1,
            final_epochs=1,
            trial_epochs=1,
            trials=1,
            skip_optuna=True,
            n_streams=1,
            noise_std=0.0,
            physics_baselines=physics,
            dynamic_bias_provider=provider,
            metadata={"sample_dt_s": 1.0},
        )

    def test_miso_feature_major_order_and_previous_normalization_shift(self):
        import numpy as np

        from anchor_bench.training.data import TrainingMapping
        from anchor_bench.training.trainer import _load_sequences

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_multi_joint_motion(root)
            mapping = TrainingMapping(("JointA", "JointB"), ("JointB",))
            job = self._job(
                root,
                mapping,
                ["position", "position_error", "velocity", "previous_torque"],
            )
            (features, target), = _load_sequences(job)[0]

        self.assertEqual((3, 7), features.shape)
        self.assertEqual((3, 1), target.shape)
        np.testing.assert_allclose(features[0], [0.0, 10.0, 1.0, 2.0, 0.0, 1.0, 0.0])
        np.testing.assert_allclose(target[:, 0], [10.0, 20.0, 30.0])
        np.testing.assert_allclose(features[:, -1], [0.0, 10.0, 20.0])

    def test_mimo_and_shared_siso_reset_each_recurrent_sequence(self):
        import numpy as np

        from anchor_bench.training.data import TrainingMapping
        from anchor_bench.training.trainer import _load_sequences, _normalization_stats

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_multi_joint_motion(root)
            mimo = TrainingMapping(("JointA", "JointB"), ("JointA", "JointB"))
            (features, target), = _load_sequences(
                self._job(root, mimo, ["position", "previous_torque"])
            )[0]
            shared = (
                TrainingMapping(("JointA",), ("JointA",)),
                TrainingMapping(("JointB",), ("JointB",)),
            )
            sequences, _ = _load_sequences(
                self._job(root, shared[0], ["previous_torque"], mappings=shared)
            )
            stats = _normalization_stats(
                sequences,
                input_columns=["previous_torque"],
                target_columns=["torque"],
                input_widths=[1],
                target_normalization="standard",
                metadata={},
            )

        self.assertEqual((3, 4), features.shape)
        self.assertEqual((3, 2), target.shape)
        np.testing.assert_allclose(features[0, -2:], [0.0, 0.0])
        self.assertEqual(2, len(sequences))
        self.assertEqual(0.0, float(sequences[0][0][0, 0]))
        self.assertEqual(0.0, float(sequences[1][0][0, 0]))
        self.assertEqual(5.5, stats["normalization"]["inputs"]["mean"]["previous_torque"])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "autoregressive test requires torch")
    def test_previous_torque_rolls_predictions_across_chunks_and_resets_sequences(self):
        """Use model outputs in train/validation and reset feedback per sequence."""

        import numpy as np
        import torch

        from anchor_bench.training.trainer import (
            _PreviousTorqueFeedback,
            TrainingHyperParameters,
            _run_epoch,
        )

        class RecordingModel(torch.nn.Module):
            """Record recurrent inputs and emit one call-indexed prediction."""

            def __init__(self):
                """Create one parameter so the training path can backpropagate."""

                super().__init__()
                self.anchor = torch.nn.Parameter(torch.tensor(0.0))
                self.inputs = []
                self.hidden_was_none = []
                self.input_has_grad = []
                self.hidden_has_grad = []

            def forward(self, values, hidden):
                """Record inputs and return a deterministic normalized output."""

                self.inputs.append(values.detach().cpu().numpy().copy())
                self.hidden_was_none.append(hidden is None)
                self.input_has_grad.append(values.requires_grad)
                self.hidden_has_grad.append(None if hidden is None else hidden.requires_grad)
                call = float(len(self.inputs))
                prediction = torch.stack((self.anchor * 0.0 + 2 * call - 1, self.anchor * 0.0 + 2 * call))
                prediction = prediction.reshape(1, 1, 2)
                next_hidden = (self.anchor * 0.0 + call).reshape(1, 1, 1)
                return prediction, next_hidden

        feedback = _PreviousTorqueFeedback(
            input_slice=slice(1, 3),
            input_mean=np.array([10.0, 20.0], dtype=np.float32),
            input_std=np.array([2.0, 4.0], dtype=np.float32),
            target_mean=np.array([100.0, 200.0], dtype=np.float32),
            target_std=np.array([5.0, 10.0], dtype=np.float32),
        )
        sequence = (
            np.array(
                [[1.0, 999.0, 999.0, 4.0], [2.0, 888.0, 888.0, 5.0], [3.0, 777.0, 777.0, 6.0]],
                dtype=np.float32,
            ),
            np.zeros((3, 2), dtype=np.float32),
        )

        for train in (True, False):
            model = RecordingModel()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
            _run_epoch(
                torch,
                model,
                optimizer,
                torch.nn.MSELoss(),
                [sequence, sequence],
                TrainingHyperParameters(chunk_length=2),
                train=train,
                device=torch.device("cpu"),
                noise_std=0.0,
                previous_torque_feedback=feedback,
            )

            observed = np.concatenate(model.inputs, axis=1)[0]
            np.testing.assert_allclose(
                observed[:, 1:3],
                [
                    [-5.0, -5.0],
                    [47.5, 50.0],
                    [52.5, 55.0],
                    [-5.0, -5.0],
                    [62.5, 65.0],
                    [67.5, 70.0],
                ],
            )
            np.testing.assert_allclose(
                observed[:, (0, 3)],
                np.concatenate((sequence[0][:, (0, 3)], sequence[0][:, (0, 3)])),
            )
            self.assertEqual([True, False, False, True, False, False], model.hidden_was_none)
            if train:
                self.assertEqual([False, True, False, False, True, False], model.input_has_grad)
                self.assertEqual([None, True, False, None, True, False], model.hidden_has_grad)

    def test_residual_previous_torque_normalization_uses_residual_target(self):
        import numpy as np

        from anchor_bench.training.data import TrainingMapping
        from anchor_bench.training.trainer import (
            PhysicsBaselines,
            _load_sequences,
            _runtime_metadata,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_multi_joint_motion(root)
            mapping = TrainingMapping(("JointA", "JointB"), ("JointB",))
            physics = PhysicsBaselines(
                kp=(2.0,),
                kd=(0.0,),
                dry_friction=(0.0,),
                viscous_friction=(0.0,),
            )
            job = self._job(
                root,
                mapping,
                ["previous_torque"],
                model="hybrid_residual",
                physics=physics,
            )
            (features, target), = _load_sequences(job)[0]
            metadata = _runtime_metadata(job, {"normalization": {}})

        np.testing.assert_allclose(target[:, 0], [6.0, 16.4, 26.8], rtol=1e-6)
        np.testing.assert_allclose(features[:, 0], [0.0, 6.0, 16.4], rtol=1e-6)
        self.assertEqual("output_joints", metadata["pd_baseline"]["domain"])

    def test_solver_pd_uses_one_input_domain_for_both_models(self):
        import numpy as np

        from anchor_bench.training.data import TrainingMapping
        from anchor_bench.training.trainer import (
            PhysicsBaselines,
            _load_sequences,
            _runtime_metadata,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_multi_joint_motion(root)
            mapping = TrainingMapping(("JointA", "JointB"), ("JointB",))
            baselines = PhysicsBaselines(
                kp=(2.0, 3.0),
                kd=(4.0, 5.0),
                dry_friction=(0.5,),
                viscous_friction=(0.25,),
            )
            for model in ("gru", "hybrid_residual"):
                job = self._job(
                    root, mapping, ["solver_pd"], model=model, physics=baselines
                )
                (features, target), = _load_sequences(job)[0]
                metadata = _runtime_metadata(job, {"normalization": {}})
                expected_target = (
                    [10.0, 20.0, 30.0]
                    if model == "gru"
                    else [8.25, 19.8, 31.35]
                )
                np.testing.assert_allclose(
                    features, [[2.0, 1.0], [1.4, -0.6], [0.8, -2.2]], atol=1e-6
                )
                np.testing.assert_allclose(target[:, 0], expected_target, atol=1e-6)
                self.assertEqual("input_joints", metadata["pd_baseline"]["domain"])

    def test_metadata_uses_input_and_output_domain_widths(self):
        from anchor_bench.training.data import TrainingMapping
        from anchor_bench.training.trainer import (
            _load_sequences,
            _normalization_stats,
            _previous_torque_feedback,
            _runtime_metadata,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_multi_joint_motion(root)
            mapping = TrainingMapping(("JointA", "JointB"), ("JointB",))
            job = self._job(root, mapping, ["position", "previous_torque"])
            sequences, _ = _load_sequences(job)
            stats = _normalization_stats(
                sequences,
                input_columns=job.input_columns,
                target_columns=job.target_columns,
                input_widths=(2, 1),
                target_normalization="standard",
                metadata={},
            )
            metadata = _runtime_metadata(job, stats)
            feedback = _previous_torque_feedback(job, stats)

        self.assertIsNotNone(feedback)
        self.assertEqual(slice(2, 3), feedback.input_slice)
        self.assertEqual(3, metadata["schema_version"])
        self.assertEqual(1.0, metadata["sample_dt_s"])
        self.assertEqual("miso", metadata["training_topology"])
        self.assertEqual(3, metadata["input_size"])
        self.assertEqual(1, metadata["output_size"])
        self.assertEqual(2, len(metadata["normalization"]["inputs"]["mean"]["position"]))
        self.assertIsInstance(metadata["normalization"]["inputs"]["mean"]["previous_torque"], float)
        self.assertEqual(
            "previous_raw_network_output_pre_clamp",
            metadata["previous_torque_derivation"]["source"],
        )
        self.assertEqual(
            "previous_raw_network_output_pre_clamp",
            metadata["previous_torque_derivation"]["runtime_source"],
        )

    def test_identity_target_normalization_preserves_physical_torque(self):
        import numpy as np

        from anchor_bench.training.trainer import (
            _normalization_stats,
            _normalize_sequences,
        )

        sequences = [
            (
                np.asarray([[1.0], [3.0]], dtype=np.float32),
                np.asarray([[-2.0], [6.0]], dtype=np.float32),
            )
        ]
        stats = _normalization_stats(
            sequences,
            input_columns=["position"],
            target_columns=["torque"],
            input_widths=[1],
            target_normalization="identity",
            metadata={},
        )
        normalized = _normalize_sequences(
            sequences,
            stats,
            input_columns=["position"],
            target_columns=["torque"],
            input_widths=[1],
            target_width=1,
        )

        np.testing.assert_allclose(normalized[0][0][:, 0], [-1.0, 1.0])
        np.testing.assert_allclose(normalized[0][1], sequences[0][1])
        self.assertEqual(0.0, stats["normalization"]["targets"]["mean"]["torque"])
        self.assertEqual(1.0, stats["normalization"]["targets"]["std"]["torque"])

    def test_dynamic_bias_is_evaluated_once_per_source_then_sliced(self):
        import numpy as np

        from anchor_bench.training.data import TrainingMapping
        from anchor_bench.training.trainer import _load_sequences

        class Provider:
            def __init__(self):
                self.calls = []

            def compute(self, motion, requested_joints):
                self.calls.append(tuple(requested_joints))
                values = np.column_stack(
                    [
                        np.arange(len(motion.position.time), dtype=np.float32) + index * 10
                        for index, _ in enumerate(requested_joints)
                    ]
                )
                return values

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_multi_joint_motion(root)
            mappings = (
                TrainingMapping(("JointA",), ("JointA",)),
                TrainingMapping(("JointB",), ("JointB",)),
            )
            provider = Provider()
            sequences, _ = _load_sequences(
                self._job(root, mappings[0], ["dynamic_bias"], mappings=mappings, provider=provider)
            )

        self.assertEqual([("JointA", "JointB")], provider.calls)
        np.testing.assert_allclose(sequences[0][0][:, 0], [0.0, 1.0, 2.0])
        np.testing.assert_allclose(sequences[1][0][:, 0], [10.0, 11.0, 12.0])


class TrainingRunnerTests(unittest.TestCase):
    def test_training_config_resolves_run_yaml_defaults(self):
        from anchor_bench.core.schemas import TrainActuatorRunConfig
        from anchor_bench.training.runner import resolve_train_actuator_run_config

        resolved = resolve_train_actuator_run_config(TrainActuatorRunConfig(robot="so101", dry_run=True))

        self.assertEqual("so101", resolved.robot)
        self.assertEqual("gru", resolved.model)
        self.assertTrue(resolved.data.endswith("so101_arm_50motion/train"))
        self.assertTrue(resolved.output.endswith("outputs/training/so101"))
        self.assertEqual(6, len(resolved.mappings))
        self.assertTrue(all(len(mapping.input_joints) == 1 for mapping in resolved.mappings))
        self.assertTrue(resolved.skip_optuna)
        self.assertEqual(42, resolved.seed)
        self.assertEqual(["position", "position_error", "velocity"], resolved.input_columns)
        self.assertEqual("torque", resolved.target_column)

    def test_parity_yaml_defaults_and_cli_overrides_are_reproducible(self):
        from types import SimpleNamespace

        from anchor_bench.core.schemas import TrainActuatorRunConfig
        from anchor_bench.training.runner import (
            build_resolved_train_actuator_command,
            resolve_train_actuator_run_config,
        )
        from anchor_bench.training.trainer import _default_hparams

        yaml_resolved = resolve_train_actuator_run_config(
            TrainActuatorRunConfig(robot="g1", dry_run=True)
        )
        right_arm = (
            "right_shoulder_pitch",
            "right_shoulder_roll",
            "right_shoulder_yaw",
            "right_elbow",
        )
        self.assertEqual("gru", yaml_resolved.model)
        self.assertTrue(yaml_resolved.data.endswith("g1_arm_multijoint/train"))
        self.assertTrue(yaml_resolved.output.endswith("right_arm_fulltorque_fixed192"))
        self.assertEqual(1, len(yaml_resolved.mappings))
        self.assertEqual(right_arm, yaml_resolved.mappings[0].input_joints)
        self.assertEqual(right_arm, yaml_resolved.mappings[0].output_joints)
        self.assertEqual(
            [
                "position",
                "position_error",
                "velocity",
                "solver_pd",
                "dynamic_bias",
                "previous_torque",
            ],
            yaml_resolved.input_columns,
        )
        self.assertEqual("torque", yaml_resolved.target_column)
        self.assertEqual(50, yaml_resolved.final_epochs)
        parity = {
            "hidden_size": 192,
            "num_layers": 2,
            "dropout": 0.225,
            "learning_rate": 0.0004949926657319083,
            "weight_decay": 3.376044479094932e-7,
            "huber_delta": 0.3149430512234676,
            "chunk_length": 100,
            "output_activation": "tanh",
            "output_scale": 15.0,
            "target_normalization": "identity",
            "pd_kp": 40.0,
            "pd_kd": 1.0,
        }
        for name, expected in parity.items():
            self.assertEqual(expected, getattr(yaml_resolved, name))
            self.assertEqual("run.yaml", yaml_resolved.config_sources[name])

        empty = {
            "hidden_size": None,
            "num_layers": None,
            "dropout": None,
            "learning_rate": None,
            "weight_decay": None,
            "huber_delta": None,
            "chunk_length": None,
            "output_activation": None,
            "output_scale": None,
        }
        defaults = _default_hparams(SimpleNamespace(**empty))
        self.assertEqual(192, defaults.hidden_size)
        self.assertEqual(2, defaults.num_layers)
        self.assertEqual(0.225, defaults.dropout)
        single_layer = _default_hparams(
            SimpleNamespace(**{**empty, "num_layers": 1})
        )
        self.assertEqual(0.0, single_layer.dropout)

        resolved = resolve_train_actuator_run_config(
            TrainActuatorRunConfig(
                robot="g1",
                hidden_size=96,
                num_layers=3,
                dropout=0.1,
                learning_rate=0.002,
                weight_decay=0.0001,
                huber_delta=0.5,
                chunk_length=64,
                output_activation="linear",
                output_scale=2.0,
                target_normalization="standard",
                pd_kp=20.0,
                pd_kd=0.5,
                dry_run=True,
            )
        )
        overrides = {
            "hidden_size": 96,
            "num_layers": 3,
            "dropout": 0.1,
            "learning_rate": 0.002,
            "weight_decay": 0.0001,
            "huber_delta": 0.5,
            "chunk_length": 64,
            "output_activation": "linear",
            "output_scale": 2.0,
            "target_normalization": "standard",
            "pd_kp": 20.0,
            "pd_kd": 0.5,
        }
        for name, expected in overrides.items():
            self.assertEqual(expected, getattr(resolved, name))
            self.assertEqual("cli", resolved.config_sources[name])
        command = build_resolved_train_actuator_command(resolved).command
        self.assertEqual(command, build_resolved_train_actuator_command(resolved).command)
        for flag in (
            "--hidden-size",
            "--num-layers",
            "--dropout",
            "--learning-rate",
            "--weight-decay",
            "--huber-delta",
            "--chunk-length",
            "--output-activation",
            "--output-scale",
            "--target-normalization",
            "--pd-kp",
            "--pd-kd",
        ):
            self.assertIn(flag, command)

        for values, message in (
            ({"hidden_size": 0}, "hidden_size must be >= 1"),
            ({"num_layers": 0}, "num_layers must be >= 1"),
            ({"dropout": 1.0}, "dropout must be >= 0 and < 1"),
            ({"num_layers": 1, "dropout": 0.1}, "dropout must be zero when num_layers is 1"),
            ({"learning_rate": 0.0}, "learning_rate.*> 0"),
            ({"weight_decay": -1.0}, "weight_decay must be finite and >= 0"),
            ({"weight_decay": float("nan")}, "weight_decay must be finite and >= 0"),
            ({"weight_decay": float("inf")}, "weight_decay must be finite and >= 0"),
            ({"huber_delta": 0.0}, "huber_delta.*> 0"),
            ({"chunk_length": 0}, "chunk_length must be >= 1"),
            ({"output_activation": "sigmoid"}, "output_activation"),
            ({"output_scale": 0.0}, "output_scale.*> 0"),
            ({"target_normalization": "minmax"}, "target_normalization"),
        ):
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, message):
                resolve_train_actuator_run_config(
                    TrainActuatorRunConfig(robot="g1", dry_run=True, **values)
                )

    def test_optuna_keeps_explicit_training_hparams_fixed(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        from anchor_bench.training.trainer import _select_hparams

        suggested: list[str] = []

        class Trial:
            def __init__(self):
                self.params = {}

            def suggest_categorical(self, name, values):
                suggested.append(name)
                self.params[name] = values[0]
                return values[0]

            def suggest_int(self, name, low, _high):
                suggested.append(name)
                self.params[name] = low
                return low

            def suggest_float(self, name, low, _high, **_kwargs):
                suggested.append(name)
                self.params[name] = low
                return low

        class Study:
            def optimize(self, objective, n_trials):
                self.assertions = n_trials
                trial = Trial()
                objective(trial)
                self.best_params = trial.params

        study = Study()
        optuna = SimpleNamespace(create_study=lambda **_kwargs: study)
        job = SimpleNamespace(
            skip_optuna=False,
            trials=1,
            trial_epochs=1,
            hidden_size=192,
            num_layers=2,
            dropout=0.225,
            learning_rate=0.0005,
            weight_decay=0.000001,
            huber_delta=0.3,
            chunk_length=100,
        )
        with (
            patch.dict("sys.modules", {"optuna": optuna}),
            patch("anchor_bench.training.trainer._fit_once", return_value=(0.0, None)),
        ):
            selected = _select_hparams(job, (), (), {})

        self.assertEqual(192, selected.hidden_size)
        self.assertEqual(2, selected.num_layers)
        self.assertEqual(0.225, selected.dropout)
        self.assertEqual(0.0005, selected.learning_rate)
        self.assertEqual(0.000001, selected.weight_decay)
        self.assertEqual(0.3, selected.huber_delta)
        self.assertEqual(100, selected.chunk_length)
        self.assertTrue(
            {
                "hidden_size",
                "num_layers",
                "dropout",
                "learning_rate",
                "weight_decay",
                "huber_delta",
                "chunk_length",
            }.isdisjoint(suggested)
        )

    def test_training_config_requires_joint_selection(self):
        from anchor_bench.core.schemas import TrainActuatorRunConfig
        from anchor_bench.training.runner import resolve_train_actuator_run_config

        with self.assertRaisesRegex(ValueError, "requires mappings or input_joints"):
            resolve_train_actuator_run_config(
                TrainActuatorRunConfig(robot="h1", data="inputs/datasets/demo")
            )

    def test_training_config_resolves_cli_column_overrides(self):
        from anchor_bench.core.schemas import TrainActuatorRunConfig
        from anchor_bench.training.runner import (
            resolve_train_actuator_run_config,
            run_resolved_train_actuator,
        )

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
        with self.assertRaisesRegex(ValueError, "target_column='torque'"):
            run_resolved_train_actuator(resolved)

    def test_residual_miso_only_reads_output_baselines(self):
        from unittest.mock import patch

        from anchor_bench.training.data import TrainingMapping
        from anchor_bench.training.runner import (
            _param_for_joint,
            _physics_baselines_for_mappings,
        )

        mapping = TrainingMapping(
            ("right_shoulder_pitch", "right_elbow"),
            ("right_elbow",),
        )
        with patch(
            "anchor_bench.training.runner._param_for_joint", wraps=_param_for_joint
        ) as read:
            _physics_baselines_for_mappings(
                "g1", (mapping,), include_solver_pd=False, include_residual=True
            )

        self.assertEqual({"right_elbow"}, {call.args[1] for call in read.call_args_list})

    def test_pd_overrides_broadcast_and_follow_slot_order(self):
        from anchor_bench.training.data import TrainingMapping
        from anchor_bench.training.runner import _physics_baselines_for_mappings

        mapping = TrainingMapping(
            ("right_shoulder_pitch", "right_elbow"),
            ("right_elbow",),
        )
        baselines = _physics_baselines_for_mappings(
            "g1",
            (mapping,),
            include_solver_pd=True,
            include_residual=False,
            pd_kp=40.0,
            pd_kd=(0.5, 1.5),
        )

        self.assertEqual((40.0, 40.0), baselines.kp)
        self.assertEqual((0.5, 1.5), baselines.kd)
        with self.assertRaisesRegex(ValueError, "pd_kp.*2 value"):
            _physics_baselines_for_mappings(
                "g1",
                (mapping,),
                include_solver_pd=True,
                include_residual=False,
                pd_kp=(40.0,),
            )

    def test_partial_parameter_maps_fail_instead_of_defaulting(self):
        from anchor_bench.training.runner import _param_for_joint

        self.assertEqual(0.0, _param_for_joint(None, "JointA", "friction", default=0.0))
        with self.assertRaisesRegex(ValueError, "missing a value"):
            _param_for_joint({"JointB": 1.0}, "JointA", "friction", default=0.0)


    def test_cli_joint_axes_override_yaml_and_infer_miso(self):
        from anchor_bench.core.schemas import TrainActuatorRunConfig
        from anchor_bench.training.runner import resolve_train_actuator_run_config

        resolved = resolve_train_actuator_run_config(
            TrainActuatorRunConfig(
                robot="g1",
                input_joints=["shoulder_pitch", "elbow"],
                output_joints=["elbow"],
                dry_run=True,
            )
        )

        self.assertEqual(
            ("right_shoulder_pitch", "right_elbow"),
            resolved.mappings[0].input_joints,
        )
        self.assertEqual(("right_elbow",), resolved.mappings[0].output_joints)
        self.assertEqual("cli", resolved.config_sources["joint_selection"])

    def test_mappings_represent_shared_siso_and_coupled_mimo(self):
        from anchor_bench.core.schemas import TrainActuatorRunConfig
        from anchor_bench.training.runner import resolve_train_actuator_run_config

        shared = resolve_train_actuator_run_config(
            TrainActuatorRunConfig(
                robot="so101",
                mappings=[
                    {"input_joints": ["Rotation"], "output_joints": ["Rotation"]},
                    {"input_joints": ["Elbow"], "output_joints": ["Elbow"]},
                ],
                dry_run=True,
            )
        )
        coupled = resolve_train_actuator_run_config(
            TrainActuatorRunConfig(
                robot="g1",
                input_joints=["all"],
                dry_run=True,
            )
        )

        self.assertEqual(2, len(shared.mappings))
        self.assertTrue(all(len(item.input_joints) == 1 for item in shared.mappings))
        self.assertEqual(4, len(coupled.mappings[0].input_joints))
        self.assertEqual(coupled.mappings[0].input_joints, coupled.mappings[0].output_joints)
        self.assertNotEqual(shared.artifact_name, coupled.artifact_name)

    def test_mappings_reject_mixed_widths_and_outputs_outside_inputs(self):
        from anchor_bench.core.schemas import TrainActuatorRunConfig
        from anchor_bench.training.runner import resolve_train_actuator_run_config

        with self.assertRaisesRegex(ValueError, "identical input and output widths"):
            resolve_train_actuator_run_config(
                TrainActuatorRunConfig(
                    robot="so101",
                    mappings=[
                        {"input_joints": ["Rotation"], "output_joints": ["Rotation"]},
                        {"input_joints": ["Rotation", "Elbow"], "output_joints": ["Elbow"]},
                    ],
                )
            )
        with self.assertRaisesRegex(ValueError, "subset"):
            resolve_train_actuator_run_config(
                TrainActuatorRunConfig(
                    robot="so101",
                    input_joints=["Rotation"],
                    output_joints=["Elbow"],
                )
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
                    output_joints=["Rotation"],
                    skip_optuna=True,
                    final_epochs=1,
                    n_streams=1,
                    hidden_size=7,
                    num_layers=2,
                    dropout=0.1,
                    learning_rate=0.0005,
                    weight_decay=0.000001,
                    huber_delta=0.3,
                    chunk_length=4,
                    output_activation="tanh",
                    output_scale=15.0,
                    target_normalization="identity",
                    seed=5,
                    dry_run=False,
                )
            )
            plan = run_resolved_train_actuator(resolved)
            prefix = f"gru_{resolved.artifact_name}"
            checkpoint_path = output_dir / f"{prefix}_best.pt"
            network_path = output_dir / f"{prefix}_network.pt"
            stats_path = output_dir / f"{prefix}_stats.json"

            self.assertFalse(plan.dry_run)
            self.assertTrue(checkpoint_path.exists())
            self.assertTrue(network_path.exists())
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            self.assertEqual("gru", stats["model"])
            self.assertEqual(resolved.artifact_name, stats["artifact_name"])
            hyperparameters = stats["metrics"]["hyperparameters"]
            self.assertEqual(7, hyperparameters["hidden_size"])
            self.assertEqual(2, hyperparameters["num_layers"])
            self.assertEqual(0.1, hyperparameters["dropout"])
            self.assertEqual(4, hyperparameters["chunk_length"])
            self.assertEqual("identity", stats["target_normalization"])
            self.assertEqual(
                {"activation": "tanh", "scale": 15.0}, stats["output_head"]
            )
            self.assertEqual(0.0, stats["normalization"]["targets"]["mean"]["torque"])
            self.assertEqual(1.0, stats["normalization"]["targets"]["std"]["torque"])
            self.assertIn("network", plan.metadata["artifacts"])
            import torch
            extra_files = {"metadata.json": ""}
            network = torch.jit.load(str(network_path), _extra_files=extra_files)
            metadata = json.loads(extra_files["metadata.json"])
            self.assertEqual(3, metadata["schema_version"])
            self.assertEqual(0.002, metadata["sample_dt_s"])
            self.assertEqual("identity", metadata["target_normalization"])
            self.assertEqual(
                {"activation": "tanh", "scale": 15.0}, metadata["output_head"]
            )
            output, _ = network(torch.zeros(2, 1, 3), torch.zeros(2, 2, 7))
            self.assertLessEqual(float(output.detach().abs().max()), 15.0)

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
                    "artifact_name": "demo",
                    "model_state_dict": model.state_dict(),
                    "input_size": 3,
                    "output_size": 1,
                    "input_columns": ["position", "position_error", "velocity"],
                    "target_columns": ["torque"],
                    "runtime_metadata": {
                        "schema_version": 3,
                        "model_type": "gru",
                        "training_topology": "siso",
                        "joint_mappings": [
                            {"input_joints": ["demo"], "output_joints": ["demo"]},
                            {"input_joints": ["demo_2"], "output_joints": ["demo_2"]},
                        ],
                        "input_size": 3,
                        "output_size": 1,
                        "input_columns": ["position", "position_error", "velocity"],
                        "input_feature_specs": [
                            {"name": name, "domain": "input_joints", "channels": ["value"]}
                            for name in ("position", "position_error", "velocity")
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
                            "targets": {"mean": {"torque": 0.0}, "std": {"torque": 1.0}},
                        },
                        "delay": {"handling": "learned", "external_delay_s": 0.0},
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
            self.assertEqual(3, metadata["schema_version"])
            self.assertEqual(1, metadata["output_size"])
            self.assertEqual(2, len(metadata["joint_mappings"]))
            output, hidden = network(torch.zeros(2, 1, 3), torch.zeros(1, 2, 4))
            self.assertEqual((2, 1, 1), tuple(output.shape))
            self.assertEqual((1, 2, 4), tuple(hidden.shape))

            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            checkpoint["output_head"] = {"activation": "tanh", "scale": 15.0}
            checkpoint["runtime_metadata"]["output_head"] = {
                "activation": "linear",
                "scale": 1.0,
            }
            torch.save(checkpoint, checkpoint_path)
            with self.assertRaisesRegex(ValueError, "output_head disagrees"):
                export_gru_checkpoint(checkpoint_path)


if __name__ == "__main__":
    unittest.main()
