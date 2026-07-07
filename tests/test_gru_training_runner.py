from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from anchor_bench.core.run_schema import SimulationDefaults, TrainingDefaults
from anchor_bench.core.schemas import TrainActuatorRunConfig
from anchor_bench.training.data import TrainingSource
from anchor_bench.training.runner import (
    build_resolved_train_actuator_command,
    resolve_train_actuator_run_config,
    run_resolved_train_actuator,
)


def _bundle(training: TrainingDefaults) -> SimpleNamespace:
    return SimpleNamespace(
        spec=SimpleNamespace(
            joint_names=["joint_a", "joint_b"],
            joint_aliases={"a": "joint_a", "b": "joint_b"},
        ),
        run_config=SimpleNamespace(
            training=training,
            simulation=SimulationDefaults(
                physics_freq=1000,
                control_freq=250,
                render_freq=250,
            ),
        ),
        paths=SimpleNamespace(
            run=Path("/configs/demo/run.yaml"),
            actuators=Path("/configs/demo/actuators.yaml"),
        ),
    )


class TrainingRunnerResolutionTests(unittest.TestCase):
    def test_cli_joint_lists_override_run_joint_type_and_resolve_aliases(self):
        resolved = resolve_train_actuator_run_config(
            TrainActuatorRunConfig(
                robot="g1",
                model="gru",
                input_joints=["shoulder_pitch", "elbow"],
                output_joints=["elbow"],
                dry_run=True,
            )
        )

        self.assertIsNone(resolved.joint_type)
        self.assertEqual(
            ["right_shoulder_pitch", "right_elbow"],
            resolved.input_joints,
        )
        self.assertEqual(["right_elbow"], resolved.output_joints)
        self.assertEqual(0.002, resolved.sample_dt_s)
        self.assertEqual("overridden_by_cli_joint_lists", resolved.config_sources["joint_type"])

        plan = build_resolved_train_actuator_command(resolved)
        self.assertNotIn("--joint-type", plan.command)
        self.assertIn("--input-joints", plan.command)
        self.assertIn("right_shoulder_pitch,right_elbow", plan.command)
        self.assertIn("--output-joints", plan.command)

    def test_explicit_cli_joint_type_overrides_configured_joint_lists(self):
        training = TrainingDefaults(
            model="gru",
            real_data_dir="inputs/demo",
            output_dir="outputs/demo",
            input_joints=["joint_a", "joint_b"],
            output_joints=["joint_b"],
        )

        with patch("anchor_bench.training.runner.get_robot", return_value=_bundle(training)):
            resolved = resolve_train_actuator_run_config(
                TrainActuatorRunConfig(
                    robot="demo",
                    joint_type="a",
                    dry_run=True,
                )
            )

        self.assertEqual("joint_a", resolved.joint_type)
        self.assertIsNone(resolved.input_joints)
        self.assertIsNone(resolved.output_joints)
        self.assertEqual("cli", resolved.config_sources["joint_type"])

    def test_cli_input_list_defaults_outputs_without_inheriting_run_outputs(self):
        training = TrainingDefaults(
            model="gru",
            real_data_dir="inputs/demo",
            output_dir="outputs/demo",
            input_joints=["joint_a", "joint_b"],
            output_joints=["joint_b"],
        )

        with patch("anchor_bench.training.runner.get_robot", return_value=_bundle(training)):
            resolved = resolve_train_actuator_run_config(
                TrainActuatorRunConfig(
                    robot="demo",
                    input_joints=["a"],
                    dry_run=True,
                )
            )

        self.assertEqual(["joint_a"], resolved.input_joints)
        self.assertEqual(["joint_a"], resolved.output_joints)
        self.assertEqual("input_joints", resolved.config_sources["output_joints"])

    def test_output_defaults_to_inputs_and_must_be_a_subset(self):
        resolved = resolve_train_actuator_run_config(
            TrainActuatorRunConfig(
                robot="g1",
                model="gru",
                input_joints=["shoulder_pitch", "elbow"],
                dry_run=True,
            )
        )
        self.assertEqual(resolved.input_joints, resolved.output_joints)

        with self.assertRaisesRegex(ValueError, "must be a subset"):
            resolve_train_actuator_run_config(
                TrainActuatorRunConfig(
                    robot="g1",
                    model="gru",
                    input_joints=["shoulder_pitch"],
                    output_joints=["elbow"],
                    dry_run=True,
                )
            )

    def test_legacy_full_mode_is_preserved(self):
        resolved = resolve_train_actuator_run_config(
            TrainActuatorRunConfig(robot="so101", dry_run=True)
        )
        plan = build_resolved_train_actuator_command(resolved)

        self.assertEqual("full", resolved.joint_type)
        self.assertIsNone(resolved.input_joints)
        self.assertIsNone(resolved.output_joints)
        joint_type_index = plan.command.index("--joint-type")
        self.assertEqual("full", plan.command[joint_type_index + 1])
        self.assertNotIn("--input-joints", plan.command)
        self.assertIsNone(plan.metadata["input_joints"])

    def test_all_topologies_reject_non_torque_source_targets(self):
        with self.assertRaisesRegex(
            ValueError,
            "SAGE actuator training requires target_column='torque'",
        ):
            resolve_train_actuator_run_config(
                TrainActuatorRunConfig(
                    robot="so101",
                    target_column="measured_torque",
                    dry_run=True,
                )
            )


class TrainingRunnerBaselineTests(unittest.TestCase):
    def setUp(self):
        self.training = TrainingDefaults(
            model="hybrid_residual",
            real_data_dir="inputs/demo",
            output_dir="outputs/demo",
            input_joints=["joint_a", "joint_b"],
            output_joints=["joint_b"],
            pd_baseline={"kp": 10.0, "kd": 1.0},
            friction_baseline={"dry": 0.2, "viscous": 0.03},
        )
        self.bundle = _bundle(self.training)

    def test_complete_cli_pairs_override_nested_run_yaml(self):
        with patch("anchor_bench.training.runner.get_robot", return_value=self.bundle):
            resolved = resolve_train_actuator_run_config(
                TrainActuatorRunConfig(
                    robot="demo",
                    pd_kp=20.0,
                    pd_kd=2.0,
                    friction_dry=0.4,
                    friction_viscous=0.05,
                    dry_run=True,
                )
            )

        self.assertEqual((20.0, 2.0), (resolved.pd_kp, resolved.pd_kd))
        self.assertEqual(
            (0.4, 0.05),
            (resolved.friction_dry, resolved.friction_viscous),
        )
        self.assertEqual("cli", resolved.config_sources["pd_baseline"])
        self.assertEqual("cli", resolved.config_sources["friction_baseline"])

        command = build_resolved_train_actuator_command(resolved).command
        self.assertEqual("20.0", command[command.index("--pd-kp") + 1])
        self.assertEqual("0.4", command[command.index("--friction-dry") + 1])

    def test_nested_baselines_are_used_before_actuator_inference(self):
        with (
            patch("anchor_bench.training.runner.get_robot", return_value=self.bundle),
            patch("anchor_bench.training.runner.load_actuator_file") as load_actuators,
        ):
            resolved = resolve_train_actuator_run_config(
                TrainActuatorRunConfig(robot="demo", dry_run=True)
            )

        load_actuators.assert_not_called()
        self.assertEqual((10.0, 1.0), (resolved.pd_kp, resolved.pd_kd))
        self.assertEqual(
            (0.2, 0.03),
            (resolved.friction_dry, resolved.friction_viscous),
        )
        self.assertEqual("run.yaml", resolved.config_sources["pd_baseline"])

    def test_partial_cli_pair_fails_even_when_run_yaml_has_a_pair(self):
        with (
            patch("anchor_bench.training.runner.get_robot", return_value=self.bundle),
            self.assertRaisesRegex(ValueError, "must be provided together"),
        ):
            resolve_train_actuator_run_config(
                TrainActuatorRunConfig(
                    robot="demo",
                    pd_kp=20.0,
                    dry_run=True,
                )
            )

    def test_baseline_inference_uses_the_feature_domain(self):
        single = resolve_train_actuator_run_config(
            TrainActuatorRunConfig(robot="g1", dry_run=True)
        )
        self.assertEqual((30.0, 0.71), (single.pd_kp, single.pd_kd))
        self.assertEqual(
            (0.462, 0.037),
            (single.friction_dry, single.friction_viscous),
        )
        self.assertEqual("actuators.yaml", single.config_sources["pd_baseline"])

        residual = resolve_train_actuator_run_config(
            TrainActuatorRunConfig(
                robot="g1",
                model="hybrid_residual",
                input_joints=["shoulder_pitch", "elbow"],
                output_joints=["elbow"],
                dry_run=True,
            )
        )
        self.assertEqual((27.47, 0.572), (residual.pd_kp, residual.pd_kd))
        self.assertEqual(
            (0.270, 0.079),
            (residual.friction_dry, residual.friction_viscous),
        )

        with self.assertRaisesRegex(ValueError, "nonuniform"):
            resolve_train_actuator_run_config(
                TrainActuatorRunConfig(
                    robot="g1",
                    model="hybrid_residual",
                    input_joints=["shoulder_pitch", "elbow"],
                    output_joints=["elbow"],
                    input_columns=["position", "solver_pd"],
                    dry_run=True,
                )
            )


    def test_uniform_inference_uses_numeric_closeness(self):
        training = TrainingDefaults(
            model="gru",
            real_data_dir="inputs/demo",
            output_dir="outputs/demo",
            input_joints=["joint_a", "joint_b"],
            input_columns=["position", "solver_pd"],
        )
        bundle = _bundle(training)
        actuator = SimpleNamespace(
            name="shared",
            joints=["joint_a", "joint_b"],
            controller=SimpleNamespace(
                parameters={
                    "stiffness": {
                        "joint_a": 10.0,
                        "joint_b": 10.0 + 1.0e-10,
                    },
                    "damping": {
                        "joint_a": 1.0,
                        "joint_b": 1.0 + 1.0e-11,
                    },
                }
            ),
            physical_parameters={},
        )

        with (
            patch("anchor_bench.training.runner.get_robot", return_value=bundle),
            patch(
                "anchor_bench.training.runner.load_actuator_file",
                return_value=SimpleNamespace(actuators=[actuator]),
            ),
        ):
            resolved = resolve_train_actuator_run_config(
                TrainActuatorRunConfig(robot="demo", dry_run=True)
            )

        self.assertEqual((10.0, 1.0), (resolved.pd_kp, resolved.pd_kd))

class TrainingRunnerExecutionTests(unittest.TestCase):
    def test_dynamic_bias_dry_run_does_not_discover_data_or_construct_provider(self):
        resolved = resolve_train_actuator_run_config(
            TrainActuatorRunConfig(
                robot="g1",
                model="gru",
                input_joints=["shoulder_pitch"],
                input_columns=["position", "dynamic_bias"],
                dry_run=True,
            )
        )

        with patch("anchor_bench.training.runner.discover_training_sources") as discover:
            plan = run_resolved_train_actuator(resolved)

        discover.assert_not_called()
        self.assertTrue(plan.dry_run)
        self.assertIn("dynamic_bias", plan.metadata["input_columns"])

    def test_non_dry_grouped_run_passes_axes_timestep_and_provider_to_job(self):
        resolved = resolve_train_actuator_run_config(
            TrainActuatorRunConfig(
                robot="g1",
                model="gru",
                input_joints=["shoulder_pitch", "elbow"],
                output_joints=["elbow"],
                input_columns=["position", "dynamic_bias", "previous_torque"],
                dry_run=False,
            )
        )
        source = TrainingSource(Path("/tmp/episode"))
        provider = object()
        artifacts = SimpleNamespace(
            checkpoint_path=Path("/tmp/model.pt"),
            stats_path=Path("/tmp/stats.json"),
            network_path=Path("/tmp/network.pt"),
            metrics={"loss": 0.1},
        )

        with (
            patch(
                "anchor_bench.training.runner.discover_training_sources",
                return_value=[source],
            ),
            patch(
                "anchor_bench.training.runner.split_training_sources",
                return_value=([source], []),
            ),
            patch(
                "anchor_bench.training.runner.read_training_header",
                return_value=["position", "torque"],
            ),
            patch(
                "anchor_bench.training.dynamic_bias.NewtonMuJoCoDynamicBiasProvider",
                return_value=provider,
            ) as provider_type,
            patch(
                "anchor_bench.training.trainer.train_stateful_gru",
                return_value=artifacts,
            ) as train,
        ):
            plan = run_resolved_train_actuator(resolved)

        provider_type.assert_called_once()
        job = train.call_args.args[0]
        self.assertEqual(
            ["right_shoulder_pitch", "right_elbow"],
            job.input_joints,
        )
        self.assertEqual(["right_elbow"], job.output_joints)
        self.assertEqual(0.002, job.sample_dt_s)
        self.assertIs(provider, job.dynamic_bias_provider)
        self.assertEqual(["torque"], job.source_target_columns)
        self.assertFalse(plan.dry_run)

    def test_legacy_full_execution_expands_independent_joint_sources(self):
        base = resolve_train_actuator_run_config(
            TrainActuatorRunConfig(robot="so101", dry_run=False)
        )
        source = TrainingSource(Path("/tmp/episode"))
        expanded = TrainingSource(Path("/tmp/episode"), "Rotation")
        artifacts = SimpleNamespace(
            checkpoint_path=Path("/tmp/model.pt"),
            stats_path=Path("/tmp/stats.json"),
            network_path=Path("/tmp/network.pt"),
            metrics={},
        )

        with (
            patch(
                "anchor_bench.training.runner.discover_training_sources",
                return_value=[source],
            ),
            patch(
                "anchor_bench.training.runner.expand_joint_sources",
                return_value=[expanded],
            ) as expand,
            patch(
                "anchor_bench.training.runner.split_training_sources",
                return_value=([expanded], []),
            ),
            patch(
                "anchor_bench.training.runner.read_training_header",
                return_value=["position", "position_error", "velocity", "torque"],
            ),
            patch(
                "anchor_bench.training.trainer.train_stateful_gru",
                return_value=artifacts,
            ) as train,
        ):
            run_resolved_train_actuator(base)

        self.assertEqual("full", train.call_args.args[0].joint_type)
        self.assertIsNone(train.call_args.args[0].input_joints)
        expand.assert_called_once()
        self.assertEqual(
            ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"],
            list(expand.call_args.args[1]),
        )


if __name__ == "__main__":
    unittest.main()
