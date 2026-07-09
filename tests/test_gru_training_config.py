from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from anchor_bench.cli.commands import dispatch
from anchor_bench.cli.parser import build_parser
from anchor_bench.core.config_model import parse_gru_columns, parse_joint_selection
from anchor_bench.core.run_schema import (
    TrainingDefaults,
    TrainingGroup,
    TrainingJointMapping,
)
from anchor_bench.core.schemas import (
    ControllerSpec,
    ResidualSpec,
    TrainActuatorRunConfig,
)


class GruFeatureValidationTests(unittest.TestCase):
    def test_dynamic_bias_and_previous_torque_are_supported(self):
        columns = parse_gru_columns("position,dynamic_bias,previous_torque,velocity,solver_pd")

        self.assertEqual(
            [
                "position",
                "dynamic_bias",
                "previous_torque",
                "velocity",
                "solver_pd",
            ],
            columns,
        )

    def test_duplicate_and_legacy_bias_columns_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate GRU input columns: position"):
            parse_gru_columns(["position", "position"])

        with self.assertRaisesRegex(ValueError, "unsupported GRU input columns: qfrc_bias"):
            parse_gru_columns(["position", "qfrc_bias"])

    def test_training_parser_accepts_all_features(self):
        for feature in ("dynamic_bias", "previous_torque"):
            with self.subTest(feature=feature):
                columns = ["position", feature]
                self.assertEqual(columns, parse_gru_columns(columns))

    def test_runtime_schemas_reject_training_only_derived_features(self):
        for feature in ("dynamic_bias", "previous_torque"):
            with self.subTest(feature=feature):
                columns = ["position", feature]

                with self.assertRaisesRegex(ValueError, f"unsupported GRU input columns: {feature}"):
                    ControllerSpec.from_dict(
                        {
                            "type": "gru",
                            "network_file": "runtime.pt",
                            "parameters": {
                                "input_columns": columns,
                            },
                        },
                        "controller",
                    )

                with self.assertRaisesRegex(ValueError, f"unsupported GRU input columns: {feature}"):
                    ResidualSpec.from_dict(
                        {
                            "type": "gru",
                            "network_file": "runtime.pt",
                            "input_columns": columns,
                        },
                        "residual",
                    )


class TrainingDefaultsTests(unittest.TestCase):
    def test_joint_domains_and_baselines_round_trip(self):
        payload = {
            "input_joints": ["shoulder", "elbow"],
            "output_joints": ["elbow"],
            "input_columns": ["position", "dynamic_bias", "previous_torque"],
            "pd_baseline": {"kp": 40.0, "kd": 1.5},
            "friction_baseline": {"dry": 0.2, "viscous": 0.03},
        }

        config = TrainingDefaults.from_dict(payload, "training")

        self.assertEqual(["shoulder", "elbow"], config.input_joints)
        self.assertEqual(["elbow"], config.output_joints)
        self.assertEqual(40.0, config.pd_baseline.kp)
        self.assertEqual(0.03, config.friction_baseline.viscous)
        self.assertEqual(payload, config.to_dict())

        group = config.resolved_groups[0]
        self.assertEqual("default", group.name)
        self.assertEqual(["shoulder", "elbow"], group.mappings[0].input_joints)
        self.assertEqual(["elbow"], group.mappings[0].output_joints)

    def test_all_joint_selectors_round_trip_as_bare_sentinels(self):
        coupled = TrainingDefaults.from_dict(
            {"input_joints": "all", "output_joints": "all"},
            "training",
        )

        self.assertEqual("all", coupled.input_joints)
        self.assertEqual("all", coupled.output_joints)
        self.assertEqual(
            {"input_joints": "all", "output_joints": "all"},
            coupled.to_dict(),
        )

    def test_groups_round_trip_and_preserve_mapping_order(self):
        payload = {
            "groups": [
                {
                    "name": "shared_arm",
                    "mappings": [
                        {
                            "input_joints": ["shoulder"],
                            "output_joints": ["shoulder"],
                        },
                        {"input_joints": ["elbow"]},
                    ],
                }
            ]
        }

        config = TrainingDefaults.from_dict(payload, "training")

        self.assertEqual(payload, config.to_dict())
        self.assertEqual("shared_arm", config.resolved_groups[0].name)
        self.assertEqual(
            ["shoulder", "elbow"],
            [mapping.input_joints[0] for mapping in config.resolved_groups[0].mappings],
        )
        self.assertIsNone(config.resolved_groups[0].mappings[1].output_joints)

    def test_groups_reject_invalid_structure_and_shorthand_mixing(self):
        invalid = [
            ({"groups": []}, "at least 1 item"),
            (
                {"groups": [{"name": "empty", "mappings": []}]},
                "at least 1 item",
            ),
            (
                {"groups": [{"name": "missing", "mappings": [{}]}]},
                "missing required key \x27input_joints\x27",
            ),
            (
                {
                    "groups": [
                        {
                            "name": "duplicate",
                            "mappings": [{"input_joints": ["shoulder"]}],
                        },
                        {
                            "name": "duplicate",
                            "mappings": [{"input_joints": ["elbow"]}],
                        },
                    ]
                },
                "duplicate training group names: duplicate",
            ),
            (
                {
                    "groups": [
                        {
                            "name": "duplicate_mapping",
                            "mappings": [
                                {"input_joints": ["shoulder"]},
                                {
                                    "input_joints": ["shoulder"],
                                    "output_joints": ["shoulder"],
                                },
                            ],
                        }
                    ]
                },
                "contains duplicate mappings",
            ),
            (
                {
                    "groups": [
                        {
                            "name": "shared",
                            "mappings": [{"input_joints": ["shoulder"]}],
                        }
                    ],
                    "input_joints": ["elbow"],
                },
                "groups cannot be combined with input_joints or output_joints",
            ),
        ]

        for payload, message in invalid:
            with self.subTest(payload=payload), self.assertRaisesRegex(ValueError, message):
                TrainingDefaults.from_dict(payload, "training")

    def test_joint_lists_reject_duplicates_and_invalid_selection_modes(self):
        with self.assertRaisesRegex(ValueError, "contains duplicate joints: shoulder"):
            TrainingDefaults.from_dict(
                {"input_joints": ["shoulder", "shoulder"]},
                "training",
            )

        with self.assertRaisesRegex(
            ValueError,
            "input_joints is required when output_joints is configured",
        ):
            TrainingDefaults.from_dict(
                {"output_joints": ["shoulder"]},
                "training",
            )

    def test_all_is_reserved_for_bare_standalone_selection(self):
        self.assertEqual("all", parse_joint_selection("all"))
        for value in (["all"], ["shoulder", "all"], "shoulder,all"):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(
                    ValueError,
                    "'all' must be used as a standalone joint selector",
                ),
            ):
                parse_joint_selection(value)

    def test_removed_joint_type_is_rejected_as_an_unknown_key(self):
        with self.assertRaisesRegex(ValueError, "training has unknown keys: joint_type"):
            TrainingDefaults.from_dict({"joint_type": "full"}, "training")

    def test_nested_baselines_require_complete_pairs(self):
        with self.assertRaisesRegex(ValueError, "missing required key 'kd'"):
            TrainingDefaults.from_dict(
                {"pd_baseline": {"kp": 40.0}},
                "training",
            )

        with self.assertRaisesRegex(ValueError, "missing required key 'viscous'"):
            TrainingDefaults.from_dict(
                {"friction_baseline": {"dry": 0.2}},
                "training",
            )


class TrainActuatorRunConfigTests(unittest.TestCase):
    def test_programmatic_groups_and_shorthand_share_one_resolution_api(self):
        group = TrainingGroup(
            name="shared",
            mappings=[
                TrainingJointMapping(input_joints=["shoulder"]),
                TrainingJointMapping(input_joints=["elbow"]),
            ],
        )

        grouped = TrainActuatorRunConfig(robot="demo", groups=[group])
        shorthand = TrainActuatorRunConfig(robot="demo", input_joints=["shoulder"])

        self.assertEqual([group], grouped.resolved_groups)
        self.assertEqual("default", shorthand.resolved_groups[0].name)
        with self.assertRaisesRegex(ValueError, "groups cannot be combined"):
            TrainActuatorRunConfig(
                robot="demo",
                groups=[group],
                input_joints=["shoulder"],
            )

    def test_direct_config_rejects_duplicate_features_and_joints(self):
        with self.assertRaisesRegex(ValueError, "duplicate GRU input columns"):
            TrainActuatorRunConfig(
                robot="g1",
                input_columns=["position", "position"],
            )

        with self.assertRaisesRegex(ValueError, "contains duplicate joints"):
            TrainActuatorRunConfig(
                robot="g1",
                input_joints=["shoulder", "shoulder"],
            )

        with self.assertRaisesRegex(
            ValueError,
            "input_joints is required when output_joints is configured",
        ):
            TrainActuatorRunConfig(robot="g1", output_joints=["shoulder"])


class TrainActuatorCliTests(unittest.TestCase):
    def test_removed_joint_type_flag_is_rejected(self):
        with self.assertRaises(SystemExit):
            build_parser().parse_args(
                [
                    "train-actuator",
                    "--robot",
                    "so101",
                    "--joint-type",
                    "Rotation",
                ]
            )

    def test_cli_builds_run_config_with_joint_domains_and_baselines(self):
        args = build_parser().parse_args(
            [
                "train-actuator",
                "--robot",
                "g1",
                "--input-joints",
                "shoulder,elbow",
                "--output-joints",
                "elbow",
                "--input-columns",
                "position,dynamic_bias,previous_torque",
                "--pd-kp",
                "40.0",
                "--pd-kd",
                "1.5",
                "--friction-dry",
                "0.2",
                "--friction-viscous",
                "0.03",
                "--dry-run",
            ]
        )
        resolved = SimpleNamespace(data=None)

        with (
            patch(
                "anchor_bench.training.runner.resolve_train_actuator_run_config",
                return_value=resolved,
            ) as resolve,
            patch(
                "anchor_bench.training.runner.run_resolved_train_actuator",
                return_value=object(),
            ),
            patch("anchor_bench.cli.commands._print_plan", return_value=0),
        ):
            status = dispatch(args)

        self.assertEqual(0, status)
        config = resolve.call_args.args[0]
        self.assertEqual(["shoulder", "elbow"], config.input_joints)
        self.assertEqual(["elbow"], config.output_joints)
        self.assertEqual(
            ["position", "dynamic_bias", "previous_torque"],
            config.input_columns,
        )
        self.assertEqual(40.0, config.pd_kp)
        self.assertEqual(1.5, config.pd_kd)
        self.assertEqual(0.2, config.friction_dry)
        self.assertEqual(0.03, config.friction_viscous)


if __name__ == "__main__":
    unittest.main()
