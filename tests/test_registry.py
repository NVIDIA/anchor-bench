
import unittest

from anchor_bench.robots.registry import get_robot, list_robots, validate_robot


class RobotRegistryTests(unittest.TestCase):
    def test_registry_loads_all_reference_robots(self):
        self.assertEqual(list_robots(), ["g1", "h1", "so101", "teststand", "ur10e"])

    def test_so101_resolves_native_assets_and_configs(self):
        robot = get_robot("so101")
        self.assertEqual(robot.spec.name, "so101")
        self.assertEqual(robot.spec.joint_names, ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"])
        self.assertTrue(robot.paths.actuators.exists())
        self.assertTrue(robot.paths.bounds.exists())
        self.assertTrue(robot.paths.valid_joints.exists())
        self.assertEqual(validate_robot("so101"), [])

    def test_g1_resolves_right_arm_training_defaults(self):
        robot = get_robot("g1")
        self.assertEqual(robot.spec.name, "g1")
        self.assertEqual(
            robot.spec.joint_names,
            ["right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow"],
        )
        self.assertEqual(robot.run_config["training"]["model"], "hybrid_residual")
        self.assertEqual(robot.run_config["training"]["joint_type"], "right_shoulder_pitch")
        self.assertEqual(validate_robot("g1"), [])


if __name__ == "__main__":
    unittest.main()
