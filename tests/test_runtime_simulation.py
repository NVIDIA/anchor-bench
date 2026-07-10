from __future__ import annotations

import unittest
from types import SimpleNamespace

from anchor_bench.runtime.simulation import joint_ids_for


class _FakeTorch:
    int32 = "int32"

    @staticmethod
    def tensor(values, *, dtype, device):
        return values, dtype, device


class RuntimeSimulationTests(unittest.TestCase):
    def test_joint_ids_follow_requested_order(self):
        robot = SimpleNamespace(
            joint_names=["shoulder", "elbow", "wrist"],
            device="cpu",
        )

        result = joint_ids_for(robot, ["wrist", "shoulder"], _FakeTorch)

        self.assertEqual(result, ([2, 0], "int32", "cpu"))

    def test_joint_ids_report_every_missing_joint(self):
        robot = SimpleNamespace(joint_names=["shoulder"], device="cpu")

        with self.assertRaisesRegex(
            ValueError,
            "runtime articulation is missing joints: elbow, wrist",
        ):
            joint_ids_for(robot, ["elbow", "shoulder", "wrist"], _FakeTorch)


if __name__ == "__main__":
    unittest.main()
