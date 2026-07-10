from __future__ import annotations

import unittest

from pydantic import ValidationError

from anchor_bench.training.metadata import RuntimeGRUMetadata
from anchor_bench.training.schema import TrainingAcceptedCatalog, TrainingProfile


_HASH = "a" * 64


def _valid_metadata() -> dict:
    return {
        "schema_version": 3,
        "model_type": "gru",
        "runtime_dtype": "float32",
        "input_size": 3,
        "output_size": 1,
        "recurrent": {"hidden_size": 64, "num_layers": 1, "dropout": 0.0},
        "input_features": [
            {"name": "position", "domain": "input", "components": ["value"], "width": 2},
            {"name": "previous_torque", "domain": "output", "components": ["value"], "width": 1},
        ],
        "flattening_order": ["feature", "joint", "component"],
        "mappings": [
            {
                "input_joints": ["shoulder", "elbow"],
                "output_joints": ["elbow"],
                "pd_baseline": {
                    "joint_names": ["shoulder", "elbow"],
                    "kp": [20.0, 30.0],
                    "kd": [0.5, 0.75],
                    "velocity_target": "zero",
                },
                "friction_baseline": {
                    "joint_names": ["elbow"],
                    "dry": 0.1,
                    "viscous": 0.02,
                },
                "armature": {"joint_names": ["elbow"], "values": 0.005},
                "effort_limits": 25.0,
                "clamping": "residual_output",
            }
        ],
        "target_columns": ["torque_residual"],
        "sample_dt_s": 0.002,
        "normalization": {
            "inputs": {
                "mean": {"position": [0.0, 0.1], "previous_torque": 0.0},
                "std": {"position": [1.0, 1.1], "previous_torque": 2.0},
            },
            "targets": {
                "mean": {"torque_residual": 0.0},
                "std": {"torque_residual": 3.0},
            },
        },
        "delay": {"handling": "learned", "external_delay_s": 0.0},
        "previous_torque_derivation": {
            "source": "previous_target",
            "runtime_source": "previous_raw_network_output_pre_clamp",
            "initialization": "physical_zero",
            "reset_boundaries": ["recorded_episode", "joint_mapping"],
            "value_space": "residual_torque",
        },
        "hashes": {
            "profile_sha256": _HASH,
            "resolved_config_sha256": _HASH,
            "sysid_state_sha256": _HASH,
            "asset_sha256": _HASH,
            "source_sha256": _HASH,
        },
    }


class TrainingProfileTests(unittest.TestCase):
    def test_multiple_siso_mappings_naturally_share_one_profile(self):
        profile = TrainingProfile.from_dict(
            {
                "data": "inputs/so101",
                "sample_dt_s": 0.005,
                "target": "torque",
                "mappings": [
                    {"input_joints": ["Rotation"], "output_joints": ["Rotation"]},
                    {"input_joints": ["Shoulder"], "output_joints": ["Shoulder"]},
                ],
                "input_features": [
                    {"name": "position", "domain": "input", "components": ["value"]},
                    {"name": "velocity", "domain": "input", "components": ["vx", "vy", "vz"]},
                ],
            },
            "train/profile.yaml",
        )

        self.assertEqual(2, len(profile.mappings))
        self.assertEqual(42, profile.fit.seed)
        self.assertEqual(("vx", "vy", "vz"), profile.input_features[1].components)

    def test_miso_outputs_are_subset_not_equal_width(self):
        profile = TrainingProfile.model_validate(
            {
                "data": "inputs/g1",
                "sample_dt_s": 0.002,
                "target": "torque",
                "mappings": [
                    {
                        "input_joints": ["shoulder_pitch", "shoulder_roll", "elbow"],
                        "output_joints": ["elbow"],
                    }
                ],
                "input_features": [{"name": "position", "domain": "input"}],
            }
        )

        self.assertEqual(("elbow",), profile.mappings[0].output_selection)

    def test_profile_rejects_invalid_mapping_shapes_and_duplicate_features(self):
        base = {
            "data": "inputs/data",
            "sample_dt_s": 0.002,
            "target": "torque",
            "mappings": [
                {"input_joints": ["a"], "output_joints": ["a"]},
                {"input_joints": ["b", "c"], "output_joints": ["b"]},
            ],
            "input_features": [{"name": "position", "domain": "input"}],
        }
        with self.assertRaisesRegex(ValidationError, "mapping 1 has shape 2->1"):
            TrainingProfile.model_validate(base)

        base["mappings"] = [{"input_joints": ["a"], "output_joints": ["a"]}]
        base["input_features"] = [
            {"name": "position", "domain": "input"},
            {"name": "position", "domain": "input"},
        ]
        with self.assertRaisesRegex(ValidationError, "duplicate input features"):
            TrainingProfile.model_validate(base)

    def test_all_is_one_coupled_mapping_only(self):
        with self.assertRaisesRegex(ValidationError, "single coupled mapping"):
            TrainingProfile.model_validate(
                {
                    "data": "inputs/data",
                    "sample_dt_s": 0.002,
                    "target": "torque",
                    "mappings": [
                        {"input_joints": "all"},
                        {"input_joints": ["elbow"]},
                    ],
                    "input_features": [{"name": "position", "domain": "input"}],
                }
            )

    def test_accepted_catalog_rejects_cross_model_output_overlap(self):
        entry = {
            "revision": 1,
            "artifact": {"path": "outputs/train/network.pt", "sha256": _HASH, "metadata_sha256": _HASH},
            "contract": {
                "target": "torque",
                "sample_dt_s": 0.002,
                "mappings": [{"input_joints": ["elbow"], "output_joints": ["elbow"]}],
            },
            "source": {
                "run": "outputs/train/run",
                "run_input_sha256": _HASH,
                "run_manifest_sha256": _HASH,
                "sysid_state_sha256": _HASH,
            },
        }
        with self.assertRaisesRegex(ValidationError, "both own output joint 'elbow'"):
            TrainingAcceptedCatalog.model_validate(
                {"schema_version": 1, "revision": 2, "models": {"first": entry, "second": entry}}
            )


class RuntimeMetadataTests(unittest.TestCase):
    def test_schema_v3_represents_miso_with_mapping_specific_physics(self):
        metadata = RuntimeGRUMetadata.model_validate(_valid_metadata())

        self.assertEqual(3, metadata.schema_version)
        self.assertEqual(3, metadata.input_size)
        self.assertEqual(1, metadata.output_size)
        self.assertEqual((20.0, 30.0), metadata.mappings[0].pd_baseline.kp)

    def test_schema_v3_rejects_stale_group_fields(self):
        payload = _valid_metadata()
        payload["group_name"] = "right_arm"
        with self.assertRaisesRegex(ValidationError, "group_name"):
            RuntimeGRUMetadata.model_validate(payload)

    def test_schema_v3_rejects_dimension_and_zero_std_mismatches(self):
        payload = _valid_metadata()
        payload["input_size"] = 4
        with self.assertRaisesRegex(ValidationError, "input_size is 4, expected 3"):
            RuntimeGRUMetadata.model_validate(payload)

        payload = _valid_metadata()
        payload["normalization"]["inputs"]["std"]["previous_torque"] = 0.0
        with self.assertRaisesRegex(ValidationError, "must contain positive values"):
            RuntimeGRUMetadata.model_validate(payload)

    def test_schema_v3_rejects_bad_effort_limit_and_clamp_semantics(self):
        payload = _valid_metadata()
        payload["mappings"][0]["effort_limits"] = 0.0
        with self.assertRaisesRegex(ValidationError, "effort_limits must contain positive values"):
            RuntimeGRUMetadata.model_validate(payload)

        payload = _valid_metadata()
        payload["mappings"][0]["clamping"] = "total_output"
        with self.assertRaisesRegex(ValidationError, "requires clamping 'residual_output'"):
            RuntimeGRUMetadata.model_validate(payload)

    def test_schema_v3_rejects_negative_physical_parameters(self):
        mutations = (
            (
                "pd_baseline.kp",
                lambda mapping: mapping["pd_baseline"].update(kp=[-1.0, 30.0]),
            ),
            (
                "pd_baseline.kd",
                lambda mapping: mapping["pd_baseline"].update(kd=[-1.0, 0.75]),
            ),
            (
                "friction_baseline.dry",
                lambda mapping: mapping["friction_baseline"].update(dry=-1.0),
            ),
            (
                "friction_baseline.viscous",
                lambda mapping: mapping["friction_baseline"].update(viscous=-1.0),
            ),
            ("armature.values", lambda mapping: mapping["armature"].update(values=-1.0)),
        )

        for label, mutate in mutations:
            with self.subTest(label=label):
                payload = _valid_metadata()
                mutate(payload["mappings"][0])
                with self.assertRaisesRegex(ValidationError, "must contain non-negative values"):
                    RuntimeGRUMetadata.model_validate(payload)

    def test_dynamic_bias_requires_matching_derivation(self):
        payload = _valid_metadata()
        payload["input_features"].insert(
            1,
            {"name": "dynamic_bias", "domain": "input", "components": ["value"], "width": 2},
        )
        payload["input_size"] = 5
        payload["normalization"]["inputs"]["mean"]["dynamic_bias"] = [0.0, 0.0]
        payload["normalization"]["inputs"]["std"]["dynamic_bias"] = [1.0, 1.0]
        with self.assertRaisesRegex(ValidationError, "dynamic_bias feature and derivation"):
            RuntimeGRUMetadata.model_validate(payload)

        payload["dynamic_bias_derivation"] = {
            "source": "newton.solvers.SolverMuJoCo.eval_dynamic_bias",
            "provenance": {"asset_sha256": _HASH, "defaulted_joint_names": []},
        }
        RuntimeGRUMetadata.model_validate(payload)


if __name__ == "__main__":
    unittest.main()
