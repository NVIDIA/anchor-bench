from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from anchor_bench.training.data import FeatureBlock, TrainingEpisode
from anchor_bench.training.trainer import (
    ResidualPhysics,
    TrainingJob,
    _merge_dynamic_bias_provenance,
    _normalization_stats,
    _prepare_episode,
    _runtime_metadata,
)


def _job(
    *,
    model: str = "gru",
    input_joints: tuple[str, ...] = ("j0",),
    output_joints: tuple[str, ...] | None = None,
    input_columns: tuple[str, ...] = ("position",),
    physics: ResidualPhysics | None = None,
) -> TrainingJob:
    return TrainingJob(
        model=model,
        train_sources=[],
        val_sources=[],
        output_dir=Path("unused"),
        input_columns=list(input_columns),
        seed=1,
        final_epochs=1,
        trial_epochs=1,
        trials=1,
        skip_optuna=True,
        n_streams=1,
        noise_std=0.0,
        residual_physics=physics,
        input_joints=input_joints,
        output_joints=input_joints if output_joints is None else output_joints,
        sample_dt_s=0.01,
        metadata={
            "delay": {"handling": "learned", "training_delay_s": 0.0},
            "input_feature_specs": [
                {
                    "name": column,
                    "domain": (
                        "output_joints" if column == "previous_torque" else "input_joints"
                    ),
                    "channels": ["value"],
                }
                for column in input_columns
            ],
        },
    )


def _episode(
    *,
    input_joints: tuple[str, ...],
    output_joints: tuple[str, ...],
    blocks: dict[str, np.ndarray],
    target: np.ndarray,
):
    feature_blocks = {
        name: FeatureBlock(
            values=np.asarray(values, dtype=np.float32)[:, :, np.newaxis],
            domain="input_joints",
            channels=("value",),
        )
        for name, values in blocks.items()
    }
    return TrainingEpisode(
        timestamps=np.arange(len(target), dtype=np.float64) * 0.01,
        input_joints=input_joints,
        output_joints=output_joints,
        feature_blocks=feature_blocks,
        target_torque=target,
    )


class GruEpisodeAssemblyTests(unittest.TestCase):
    def test_miso_is_feature_major_and_previous_target_has_output_width(self):
        job = _job(
            input_joints=("j0", "j1"),
            output_joints=("j1",),
            input_columns=("position", "velocity", "previous_torque"),
        )
        episode = _episode(
            input_joints=("j0", "j1"),
            output_joints=("j1",),
            blocks={
                "position": np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32),
                "velocity": np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32),
            },
            target=np.array([[10], [20], [30]], dtype=np.float32),
        )

        features, target = _prepare_episode(job, episode)

        np.testing.assert_allclose(
            features,
            [
                [1, 2, 0.1, 0.2, 0],
                [3, 4, 0.3, 0.4, 10],
                [5, 6, 0.5, 0.6, 20],
            ],
        )
        np.testing.assert_allclose(target, [[10], [20], [30]])

    def test_mimo_preserves_explicit_output_order(self):
        job = _job(
            input_joints=("j0", "j1"),
            output_joints=("j1", "j0"),
            input_columns=("position", "previous_torque"),
        )
        episode = _episode(
            input_joints=("j0", "j1"),
            output_joints=("j1", "j0"),
            blocks={"position": np.array([[1, 2], [3, 4]], dtype=np.float32)},
            target=np.array([[20, 10], [40, 30]], dtype=np.float32),
        )

        features, target = _prepare_episode(job, episode)

        np.testing.assert_allclose(features, [[1, 2, 0, 0], [3, 4, 20, 10]])
        np.testing.assert_allclose(target, [[20, 10], [40, 30]])

    def test_residual_target_is_computed_before_previous_target(self):
        physics = ResidualPhysics(
            stiffness=2.0,
            damping=1.0,
            dry_friction=0.5,
            viscous_friction=0.25,
        )
        job = _job(
            model="hybrid_residual",
            input_joints=("j0", "j1"),
            output_joints=("j1",),
            input_columns=("solver_pd", "previous_torque"),
            physics=physics,
        )
        episode = _episode(
            input_joints=("j0", "j1"),
            output_joints=("j1",),
            blocks={
                "position_error": np.array([[1, 2], [3, 4]], dtype=np.float32),
                "velocity": np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
            },
            target=np.array([[10], [20]], dtype=np.float32),
        )

        features, target = _prepare_episode(job, episode)

        np.testing.assert_allclose(target[:, 0], [5.65, 11.8], atol=1.0e-6)
        np.testing.assert_allclose(
            features,
            [[1.9, 3.8, 0.0], [5.7, 7.6, 5.65]],
            atol=1.0e-6,
        )

    def test_previous_torque_resets_to_zero_for_each_episode(self):
        job = _job(
            input_joints=("j0", "j1"),
            output_joints=("j1",),
            input_columns=("position", "previous_torque"),
        )
        episodes = [
            _episode(
                input_joints=("j0", "j1"),
                output_joints=("j1",),
                blocks={"position": np.array([[1, 2], [3, 4]], dtype=np.float32)},
                target=np.array([[10], [20]], dtype=np.float32),
            ),
            _episode(
                input_joints=("j0", "j1"),
                output_joints=("j1",),
                blocks={"position": np.array([[5, 6], [7, 8]], dtype=np.float32)},
                target=np.array([[30], [40]], dtype=np.float32),
            ),
        ]

        first_features, _ = _prepare_episode(job, episodes[0])
        second_features, _ = _prepare_episode(job, episodes[1])

        self.assertEqual(0.0, first_features[0, -1])
        self.assertEqual(10.0, first_features[1, -1])
        self.assertEqual(0.0, second_features[0, -1])
        self.assertEqual(30.0, second_features[1, -1])

    def test_normalization_uses_input_and_output_feature_domains(self):
        sequences = [
            (
                np.array([[1, 10, 100], [3, 14, 200]], dtype=np.float32),
                np.array([[5], [9]], dtype=np.float32),
            )
        ]
        stats = _normalization_stats(
            sequences,
            input_columns=["position", "previous_torque"],
            target_column="torque",
            metadata={},
            input_widths=(2, 1),
            target_width=1,
        )

        self.assertEqual([2.0, 12.0], stats["normalization"]["inputs"]["mean"]["position"])
        self.assertEqual(150.0, stats["normalization"]["inputs"]["mean"]["previous_torque"])
        self.assertEqual(7.0, stats["normalization"]["targets"]["mean"]["torque"])

        with self.assertRaisesRegex(ValueError, "cannot override reserved fields"):
            _normalization_stats(
                sequences,
                input_columns=["position", "previous_torque"],
                target_column="torque",
                metadata={"normalization": {}},
                input_widths=(2, 1),
                target_width=1,
            )

    def test_runtime_metadata_describes_miso_axes_and_previous_target(self):
        job = _job(
            input_joints=("j0", "j1"),
            output_joints=("j1",),
            input_columns=("position", "previous_torque"),
        )
        stats = {
            "input_feature_specs": job.metadata["input_feature_specs"],
            "normalization": {
                "inputs": {
                    "mean": {"position": [0.0, 0.0], "previous_torque": 0.0},
                    "std": {"position": [1.0, 1.0], "previous_torque": 1.0},
                },
                "targets": {"mean": {"torque": 0.0}, "std": {"torque": 1.0}},
            }
        }

        metadata = _runtime_metadata(job, stats)

        self.assertEqual("miso", metadata["training_topology"])
        self.assertEqual(["j0", "j1"], metadata["joint_names"])
        self.assertEqual(["j1"], metadata["output_joint_names"])
        self.assertEqual("previous_target", metadata["previous_torque_derivation"]["source"])
        self.assertEqual("physical_zero", metadata["previous_torque_derivation"]["initialization"])
        self.assertEqual(
            "previous_raw_network_output_pre_clamp",
            metadata["previous_torque_derivation"]["runtime_source"],
        )
        self.assertEqual("learned", metadata["delay"]["handling"])

    def test_dynamic_bias_provenance_unions_episode_joint_sets(self):
        first = _merge_dynamic_bias_provenance(
            None,
            {
                "source": "fake",
                "trajectory_joint_names": ["j0", "j1"],
                "requested_joint_names": ["j0"],
                "mapped_joint_names": ["j0", "j1"],
                "unmapped_trajectory_joint_names": [],
                "defaulted_joint_names": ["j2"],
            },
        )
        merged = _merge_dynamic_bias_provenance(
            first,
            {
                "source": "fake",
                "trajectory_joint_names": ["j0", "j3"],
                "requested_joint_names": ["j0"],
                "mapped_joint_names": ["j0"],
                "unmapped_trajectory_joint_names": ["j3"],
                "defaulted_joint_names": ["j1", "j2"],
            },
        )

        self.assertEqual(["j0", "j1", "j3"], merged["trajectory_joint_names"])
        self.assertEqual(["j3"], merged["unmapped_trajectory_joint_names"])
        self.assertEqual(["j1", "j2"], merged["defaulted_joint_names"])


class GruExportContractTests(unittest.TestCase):
    def _metadata(self) -> dict:
        return {
            "schema_version": 1,
            "model_type": "gru",
            "training_topology": "miso",
            "joint_names": ["j0", "j1"],
            "output_joint_names": ["j1"],
            "input_columns": ["position", "previous_torque"],
            "input_feature_specs": [
                {
                    "name": "position",
                    "domain": "input_joints",
                    "channels": ["value"],
                },
                {
                    "name": "previous_torque",
                    "domain": "output_joints",
                    "channels": ["value"],
                },
            ],
            "target_columns": ["torque"],
            "sample_dt_s": 0.01,
            "normalization": {
                "inputs": {
                    "mean": {"position": [0.0, 0.0], "previous_torque": 0.0},
                    "std": {"position": [1.0, 1.0], "previous_torque": 1.0},
                },
                "targets": {"mean": {"torque": 0.0}, "std": {"torque": 1.0}},
            },
            "delay": {"handling": "learned", "training_delay_s": 0.0},
            "previous_torque_derivation": {
                "source": "previous_target",
                "runtime_source": "previous_raw_network_output_pre_clamp",
                "initialization": "physical_zero",
                "reset": "episode",
                "value_space": "physical_torque",
            },
        }

    def _vector_metadata(self) -> dict:
        metadata = self._metadata()
        metadata["input_columns"] = ["velocity", "position"]
        metadata["input_feature_specs"] = [
            {
                "name": "velocity",
                "domain": "input_joints",
                "channels": ["vx", "vy", "vz"],
            },
            {
                "name": "position",
                "domain": "output_joints",
                "channels": ["x", "y", "z"],
            },
        ]
        metadata["normalization"]["inputs"] = {
            "mean": {
                "velocity": [0.0] * 6,
                "position": [0.0] * 3,
            },
            "std": {
                "velocity": [1.0] * 6,
                "position": [1.0] * 3,
            },
        }
        metadata.pop("previous_torque_derivation")
        return metadata

    def _metadata_for_topology(self, topology: str) -> tuple[dict, int, int]:
        metadata = self._metadata()
        metadata["training_topology"] = topology
        if topology == "miso":
            return metadata, 3, 1
        if topology == "siso":
            metadata["joint_names"] = ["j0"]
            metadata["output_joint_names"] = ["j0"]
            metadata["normalization"]["inputs"] = {
                "mean": {"position": 0.0, "previous_torque": 0.0},
                "std": {"position": 1.0, "previous_torque": 1.0},
            }
            return metadata, 2, 1
        if topology == "mimo":
            metadata["output_joint_names"] = ["j0", "j1"]
            metadata["normalization"]["inputs"]["mean"]["previous_torque"] = [
                0.0,
                0.0,
            ]
            metadata["normalization"]["inputs"]["std"]["previous_torque"] = [
                1.0,
                1.0,
            ]
            metadata["normalization"]["targets"] = {
                "mean": {"torque": [0.0, 0.0]},
                "std": {"torque": [1.0, 1.0]},
            }
            return metadata, 4, 2
        raise ValueError(f"unsupported test topology {topology!r}")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "TorchScript export requires torch")
    def test_export_uses_authoritative_metadata_and_explicit_dimensions(self):
        import torch

        from anchor_bench.training.export import build_runtime_gru, export_gru_checkpoint

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "model_best.pt"
            model = build_runtime_gru(
                input_size=3, output_size=1, hidden_size=4, num_layers=1, dropout=0.0
            )
            runtime_metadata = self._metadata()
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_columns": ["ignored_top_level_input"],
                    "target_columns": ["ignored_top_level_target"],
                    "training_topology": "ignored_top_level_topology",
                    "input_feature_specs": [{"ignored": True}],
                    "input_joints": ["ignored_top_level_joint"],
                    "output_joints": ["ignored_top_level_joint"],
                    "normalization": {"ignored": True},
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

        metadata = json.loads(extra_files["metadata.json"].decode("utf-8"))
        self.assertEqual(["j1"], metadata["output_joint_names"])
        output, hidden = network(torch.zeros(2, 1, 3), torch.zeros(1, 2, 4))
        self.assertEqual((2, 1, 1), tuple(output.shape))
        self.assertEqual((1, 2, 4), tuple(hidden.shape))

    def test_runtime_metadata_checkpoint_requires_explicit_dimensions(self):
        from unittest.mock import patch

        from anchor_bench.training.export import export_gru_checkpoint

        metadata = self._metadata()
        cases = {
            "input_size": {"output_size": 1, "runtime_metadata": metadata},
            "output_size": {"input_size": 3, "runtime_metadata": metadata},
        }
        fake_torch = SimpleNamespace(load=lambda *args, **kwargs: None)
        for missing, checkpoint in cases.items():
            fake_torch.load = lambda *args, checkpoint=checkpoint, **kwargs: checkpoint
            with (
                self.subTest(missing=missing),
                patch(
                    "anchor_bench.training.export._import_torch",
                    return_value=fake_torch,
                ),
                self.assertRaisesRegex(
                    ValueError,
                    f"checkpoint {missing} must be a positive integer",
                ),
            ):
                export_gru_checkpoint("model_best.pt")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "TorchScript export requires torch")
    def test_siso_and_mimo_exports_have_deterministic_shapes(self):
        import torch

        from anchor_bench.training.export import build_runtime_gru, export_gru_checkpoint

        siso = self._metadata()
        siso.update(
            {
                "joint_names": ["j0"],
                "output_joint_names": ["j0"],
                "input_columns": ["position"],
                "normalization": {
                    "inputs": {"mean": {"position": 0.0}, "std": {"position": 1.0}},
                    "targets": {"mean": {"torque": 0.0}, "std": {"torque": 1.0}},
                },
            }
        )
        siso["training_topology"] = "siso"
        siso["input_feature_specs"] = [siso["input_feature_specs"][0]]
        siso.pop("previous_torque_derivation")
        mimo = self._metadata()
        mimo["training_topology"] = "mimo"
        mimo["output_joint_names"] = ["j0", "j1"]
        mimo["normalization"] = {
            "inputs": {
                "mean": {"position": [0.0, 0.0], "previous_torque": [0.0, 0.0]},
                "std": {"position": [1.0, 1.0], "previous_torque": [1.0, 1.0]},
            },
            "targets": {
                "mean": {"torque": [0.0, 0.0]},
                "std": {"torque": [1.0, 1.0]},
            },
        }
        cases = [
            ("siso", siso, 1, 1),
            ("mimo", mimo, 4, 2),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            for name, metadata, input_size, output_size in cases:
                with self.subTest(name=name):
                    torch.manual_seed(7)
                    model = build_runtime_gru(
                        input_size=input_size,
                        output_size=output_size,
                        hidden_size=4,
                        num_layers=1,
                        dropout=0.0,
                    )
                    checkpoint_path = Path(tmp) / f"{name}_best.pt"
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "input_size": input_size,
                            "output_size": output_size,
                            "runtime_metadata": metadata,
                            "hyperparameters": {
                                "hidden_size": 4,
                                "num_layers": 1,
                                "dropout": 0.0,
                            },
                        },
                        checkpoint_path,
                    )
                    network = torch.jit.load(str(export_gru_checkpoint(checkpoint_path)))
                    features = torch.zeros(2, 3, input_size)
                    hidden = torch.zeros(1, 2, 4)
                    first, _ = network(features, hidden)
                    second, _ = network(features, hidden)
                    self.assertEqual((2, 3, output_size), tuple(first.shape))
                    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "TorchScript export requires torch")
    def test_legacy_checkpoint_uses_top_level_columns_for_default_dimensions(self):
        import torch

        from anchor_bench.training.export import build_runtime_gru, export_gru_checkpoint

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "legacy_best.pt"
            model = build_runtime_gru(
                input_size=2,
                output_size=1,
                hidden_size=4,
                num_layers=1,
                dropout=0.0,
            )
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_columns": ["position", "velocity"],
                    "target_columns": ["torque"],
                    "hyperparameters": {
                        "hidden_size": 4,
                        "num_layers": 1,
                        "dropout": 0.0,
                    },
                },
                checkpoint_path,
            )
            network = torch.jit.load(str(export_gru_checkpoint(checkpoint_path)))

        output, hidden = network(torch.zeros(2, 3, 2), torch.zeros(1, 2, 4))
        self.assertEqual((2, 3, 1), tuple(output.shape))
        self.assertEqual((1, 2, 4), tuple(hidden.shape))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "TorchScript export requires torch")
    def test_export_supports_multichannel_input_and_output_joint_blocks(self):
        import torch

        from anchor_bench.training.export import build_runtime_gru, export_gru_checkpoint

        metadata = self._vector_metadata()
        input_size = 9
        with tempfile.TemporaryDirectory() as tmp:
            torch.manual_seed(17)
            model = build_runtime_gru(
                input_size=input_size,
                output_size=1,
                hidden_size=4,
                num_layers=1,
                dropout=0.0,
            )
            checkpoint_path = Path(tmp) / "vector_best.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_size": input_size,
                    "output_size": 1,
                    "runtime_metadata": metadata,
                    "hyperparameters": {
                        "hidden_size": 4,
                        "num_layers": 1,
                        "dropout": 0.0,
                    },
                },
                checkpoint_path,
            )
            extra_files = {"metadata.json": ""}
            network = torch.jit.load(
                str(export_gru_checkpoint(checkpoint_path)),
                _extra_files=extra_files,
            )

        embedded = json.loads(extra_files["metadata.json"].decode("utf-8"))
        self.assertEqual(metadata["input_feature_specs"], embedded["input_feature_specs"])
        features = torch.arange(54, dtype=torch.float32).reshape(2, 3, input_size)
        hidden = torch.zeros(1, 2, 4)
        first, first_hidden = network(features, hidden)
        second, second_hidden = network(features, hidden)
        self.assertEqual((2, 3, 1), tuple(first.shape))
        self.assertEqual((1, 2, 4), tuple(first_hidden.shape))
        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
        torch.testing.assert_close(first_hidden, second_hidden, rtol=0.0, atol=0.0)

    def test_feature_specs_reject_malformed_and_inconsistent_contracts(self):
        from anchor_bench.training.export import _validate_runtime_metadata

        valid = self._vector_metadata()
        _validate_runtime_metadata(valid, input_size=9, output_size=1)

        wrong_dimensions = copy.deepcopy(valid)
        with self.assertRaisesRegex(ValueError, "metadata requires 9"):
            _validate_runtime_metadata(
                wrong_dimensions,
                input_size=8,
                output_size=1,
            )

        wrong_stats = copy.deepcopy(valid)
        wrong_stats["normalization"]["inputs"]["mean"]["velocity"] = [0.0] * 5
        with self.assertRaisesRegex(ValueError, "must contain 6 value"):
            _validate_runtime_metadata(wrong_stats, input_size=9, output_size=1)

        missing_specs = copy.deepcopy(valid)
        missing_specs.pop("input_feature_specs")
        not_a_list = copy.deepcopy(valid)
        not_a_list["input_feature_specs"] = {}
        null_specs = copy.deepcopy(valid)
        null_specs["input_feature_specs"] = None
        wrong_count = copy.deepcopy(valid)
        wrong_count["input_feature_specs"].pop()
        not_a_mapping = copy.deepcopy(valid)
        not_a_mapping["input_feature_specs"][0] = "velocity"
        missing_field = copy.deepcopy(valid)
        missing_field["input_feature_specs"][0].pop("channels")
        wrong_order = copy.deepcopy(valid)
        wrong_order["input_feature_specs"].reverse()
        unknown_domain = copy.deepcopy(valid)
        unknown_domain["input_feature_specs"][0]["domain"] = "global"
        empty_channels = copy.deepcopy(valid)
        empty_channels["input_feature_specs"][0]["channels"] = []
        duplicate_channels = copy.deepcopy(valid)
        duplicate_channels["input_feature_specs"][0]["channels"] = ["vx", "vx"]
        invalid_channel = copy.deepcopy(valid)
        invalid_channel["input_feature_specs"][0]["channels"] = ["vx", 3]

        cases = [
            ("missing specs", missing_specs, "input_feature_specs"),
            ("not list", not_a_list, "ordered list"),
            ("null specs", null_specs, "ordered list"),
            ("wrong count", wrong_count, "matching input_columns"),
            ("not mapping", not_a_mapping, "must be a mapping"),
            ("missing field", missing_field, "contain exactly"),
            ("wrong order", wrong_order, "match input_columns in order"),
            ("unknown domain", unknown_domain, "domain must be one of"),
            ("empty channels", empty_channels, "non-empty list"),
            ("duplicate channels", duplicate_channels, "duplicates"),
            ("invalid channel", invalid_channel, "non-empty strings"),
        ]
        for label, metadata, message in cases:
            with self.subTest(label=label), self.assertRaisesRegex(
                ValueError,
                message,
            ):
                _validate_runtime_metadata(
                    metadata,
                    input_size=9,
                    output_size=1,
                )

        previous_torque = self._metadata()
        previous_torque["input_feature_specs"] = [
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
        with self.assertRaisesRegex(ValueError, "previous_torque must use"):
            _validate_runtime_metadata(
                previous_torque,
                input_size=3,
                output_size=1,
            )

    def test_export_rejects_feature_specs_without_runtime_metadata(self):
        from unittest.mock import patch

        from anchor_bench.training.export import export_gru_checkpoint

        metadata = self._vector_metadata()
        base = {
            "input_columns": metadata["input_columns"],
            "input_feature_specs": metadata["input_feature_specs"],
            "target_columns": metadata["target_columns"],
            "input_size": 9,
            "output_size": 1,
        }
        cases = {
            "missing": base,
            "null": {**base, "runtime_metadata": None},
        }
        fake_torch = SimpleNamespace(load=lambda *args, **kwargs: None)
        for label, checkpoint in cases.items():
            fake_torch.load = lambda *args, checkpoint=checkpoint, **kwargs: checkpoint
            with (
                self.subTest(label=label),
                patch(
                    "anchor_bench.training.export._import_torch",
                    return_value=fake_torch,
                ),
                self.assertRaisesRegex(
                    ValueError,
                    "requires non-null runtime_metadata",
                ),
            ):
                export_gru_checkpoint("model_best.pt")

    def test_runtime_metadata_requires_derived_topology(self):
        from anchor_bench.training.export import _validate_runtime_metadata

        metadata = self._metadata()
        metadata.pop("training_topology")
        with self.assertRaisesRegex(ValueError, "training_topology must be 'miso'"):
            _validate_runtime_metadata(metadata, input_size=3, output_size=1)

    def test_new_topology_metadata_accepts_all_supported_shapes(self):
        from anchor_bench.training.export import _validate_runtime_metadata

        for topology in ("siso", "miso", "mimo"):
            with self.subTest(topology=topology):
                metadata, input_size, output_size = self._metadata_for_topology(
                    topology
                )
                _validate_runtime_metadata(
                    metadata,
                    input_size=input_size,
                    output_size=output_size,
                )

    def test_new_topology_metadata_rejects_invalid_and_contradictory_shapes(self):
        from anchor_bench.training.export import _validate_runtime_metadata

        siso_shape = self._metadata()
        siso_shape["training_topology"] = "siso"
        miso_shape, _, _ = self._metadata_for_topology("siso")
        miso_shape["training_topology"] = "miso"
        mimo_shape = self._metadata()
        mimo_shape["training_topology"] = "mimo"
        output_outside_inputs = self._metadata()
        output_outside_inputs["output_joint_names"] = ["j2"]
        invalid = self._metadata()
        invalid["training_topology"] = "many"
        invalid_type = self._metadata()
        invalid_type["training_topology"] = []

        cases = [
            ("siso dimensions", siso_shape, 3, 1, "must be 'miso'"),
            ("miso dimensions", miso_shape, 2, 1, "must be 'siso'"),
            ("mimo dimensions", mimo_shape, 3, 1, "must be 'miso'"),
            ("output subset", output_outside_inputs, 3, 1, "must be a subset"),
            ("invalid topology", invalid, 3, 1, "must be 'miso'"),
            ("invalid topology type", invalid_type, 3, 1, "must be 'miso'"),
        ]
        for label, metadata, input_size, output_size, message in cases:
            with self.subTest(label=label), self.assertRaisesRegex(
                ValueError,
                message,
            ):
                _validate_runtime_metadata(
                    metadata,
                    input_size=input_size,
                    output_size=output_size,
                )

    def test_export_rejects_metadata_dimension_mismatch(self):
        from anchor_bench.training.export import _validate_runtime_metadata

        with self.assertRaisesRegex(ValueError, "metadata requires 3"):
            _validate_runtime_metadata(self._metadata(), input_size=4, output_size=1)

    def test_export_validation_rejects_unsupported_features_and_missing_baselines(self):
        from anchor_bench.training.export import _validate_runtime_metadata

        unsupported = self._metadata()
        unsupported["input_columns"] = ["mystery"]
        with self.assertRaisesRegex(ValueError, "unsupported input features"):
            _validate_runtime_metadata(unsupported, input_size=2, output_size=1)

        duplicate = self._metadata()
        duplicate["input_columns"] = ["position", "position"]
        with self.assertRaisesRegex(ValueError, "must not contain duplicates"):
            _validate_runtime_metadata(duplicate, input_size=4, output_size=1)

        nonfinite = self._metadata()
        nonfinite["normalization"]["inputs"]["mean"]["previous_torque"] = float("nan")
        with self.assertRaisesRegex(ValueError, "must contain finite values"):
            _validate_runtime_metadata(nonfinite, input_size=3, output_size=1)

        residual = self._metadata()
        residual["target_columns"] = ["torque_residual"]
        residual["normalization"]["targets"] = {
            "mean": {"torque_residual": 0.0},
            "std": {"torque_residual": 1.0},
        }
        with self.assertRaisesRegex(ValueError, "pd_baseline"):
            _validate_runtime_metadata(residual, input_size=3, output_size=1)

        invalid_delay = self._metadata()
        invalid_delay["delay"]["handling"] = "none"
        with self.assertRaisesRegex(ValueError, "handling must be 'learned'"):
            _validate_runtime_metadata(invalid_delay, input_size=3, output_size=1)

    def test_export_validation_requires_feature_derivations(self):
        from anchor_bench.training.export import _validate_runtime_metadata

        dynamic = self._metadata()
        dynamic["input_columns"] = ["dynamic_bias"]
        dynamic["input_feature_specs"] = [
            {
                "name": "dynamic_bias",
                "domain": "input_joints",
                "channels": ["value"],
            }
        ]
        dynamic["normalization"]["inputs"] = {
            "mean": {"dynamic_bias": [0.0, 0.0]},
            "std": {"dynamic_bias": [1.0, 1.0]},
        }
        dynamic.pop("previous_torque_derivation")
        with self.assertRaisesRegex(ValueError, "dynamic_bias_derivation"):
            _validate_runtime_metadata(dynamic, input_size=2, output_size=1)

        previous = self._metadata()
        previous["previous_torque_derivation"] = {
            **previous["previous_torque_derivation"],
            "value_space": "normalized",
        }
        with self.assertRaisesRegex(ValueError, "previous_torque_derivation is inconsistent"):
            _validate_runtime_metadata(previous, input_size=3, output_size=1)


if __name__ == "__main__":
    unittest.main()
