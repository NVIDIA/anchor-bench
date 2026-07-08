
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from anchor_bench.robots.registry import get_robot, list_robots, validate_robot


class RobotRegistryTests(unittest.TestCase):
    def test_registry_loads_all_reference_robots(self):
        # Pin the reference robot set so additions stay deliberate.
        self.assertEqual(list_robots(), ["g1", "h1", "so101", "teststand", "ur10e"])

    def test_h1_resolves_native_assets_and_configs(self):
        robot = get_robot("h1")
        self.assertEqual(robot.spec.name, "h1")
        self.assertTrue(robot.paths.actuators.exists())
        self.assertTrue(robot.paths.bounds.exists())
        self.assertEqual(validate_robot("h1"), [])

    def test_so101_resolves_native_assets_and_configs(self):
        robot = get_robot("so101")
        self.assertEqual(robot.spec.name, "so101")
        self.assertEqual(robot.spec.joint_names, ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"])
        self.assertTrue(robot.paths.actuators.exists())
        self.assertTrue(robot.paths.bounds.exists())
        self.assertEqual(robot.run_config.training.input_joints, ["Rotation"])
        self.assertEqual(validate_robot("so101"), [])

    def test_g1_resolves_right_arm_training_defaults(self):
        robot = get_robot("g1")
        self.assertEqual(robot.spec.name, "g1")
        self.assertEqual(
            robot.spec.joint_names,
            ["right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow"],
        )
        self.assertEqual(robot.run_config.training.model, "hybrid_residual")
        self.assertEqual(
            robot.run_config.training.input_joints,
            ["right_shoulder_pitch"],
        )
        self.assertEqual(validate_robot("g1"), [])

    def test_get_robot_rejects_inconsistent_joint_configs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg_root = root / "configs"
            robot_dir = cfg_root / "badbot"
            asset_dir = root / "assets" / "badbot"
            robot_dir.mkdir(parents=True)
            asset_dir.mkdir(parents=True)
            (asset_dir / "manifest.yaml").write_text("source: test\n", encoding="utf-8")
            (robot_dir / "robot.yaml").write_text(
                "\n".join(
                    [
                        "schema_version: 1",
                        "name: badbot",
                        "display_name: Bad Bot",
                        "fixed_base: true",
                        "asset_path: assets/badbot/manifest.yaml",
                        "joint_names: [elbow]",
                        "joint_aliases: {}",
                        "actuator_file: actuators.yaml",
                        "bounds_file: bounds.yaml",
                        "ee_frame: elbow",
                    ]
                ),
                encoding="utf-8",
            )
            (robot_dir / "run.yaml").write_text("sysid: {cmaes: {sigma: 0.5}}\n", encoding="utf-8")
            (robot_dir / "actuators.yaml").write_text(
                "\n".join(
                    [
                        "schema_version: 2",
                        "backend: newton",
                        "actuators:",
                        "  - name: bad_actuator",
                        "    joints: [missing_joint]",
                        "    controller: {type: pd, parameters: {stiffness: 1.0, damping: 0.1}}",
                    ]
                ),
                encoding="utf-8",
            )
            (robot_dir / "bounds.yaml").write_text(
                "\n".join(
                    [
                        "schema_version: 2",
                        "mirror: false",
                        "joint_types: [elbow]",
                        "parameters:",
                        "  armature: {lower: 0.0, upper: 1.0}",
                    ]
                ),
                encoding="utf-8",
            )
            with (
                patch("anchor_bench.robots.registry.robot_config_root", return_value=cfg_root),
                patch("anchor_bench.robots.registry.repo_root", return_value=root),
            ):
                with self.assertRaisesRegex(ValueError, "missing_joint"):
                    get_robot("badbot")
                self.assertRegex("\n".join(validate_robot("badbot")), "missing_joint")


if __name__ == "__main__":
    unittest.main()
