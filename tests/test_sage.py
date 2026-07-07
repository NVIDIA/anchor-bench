
import tempfile
import unittest
from pathlib import Path

import numpy as np

from anchor_bench.data.sage import load_initial_state, load_real_data


def write_sage(root: Path, control_rows: int = 2, state_rows: int = 2, time_unit: str = "microseconds"):
    (root / "joint_list.txt").write_text("Rotation\nPitch\n", encoding="utf-8")
    (root / "metadata.yaml").write_text(
        "schema: sage\n"
        f"time_unit: {time_unit}\n"
        "position_unit: rad\n"
        "velocity_unit: rad/s\n"
        "torque_unit: N*m\n",
        encoding="utf-8",
    )
    control = ["type,timestamp,positions"]
    state = ["type,timestamp,positions,velocities,torques"]
    for i in range(control_rows):
        control.append(f'CONTROL,{i * 2000},"[{0.1 + i}, {0.2 + i}]"')
    for i in range(state_rows):
        state.append(
            f'STATE_MOTOR,{i * 2000},"[{0.3 + i}, {0.4 + i}]",'
            f'"[{float(i)}, {2.0 * i}]","[{10.0 + i}, {20.0 + 2.0 * i}]"'
        )
    (root / "control.csv").write_text("\n".join(control) + "\n", encoding="utf-8")
    (root / "state_motor.csv").write_text("\n".join(state) + "\n", encoding="utf-8")


class SageLoaderTests(unittest.TestCase):
    def test_loader_preserves_requested_joint_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_sage(root)
            data = load_real_data(root, joint_names=["Pitch", "Rotation"])
            self.assertEqual(data.joint_names, ["Pitch", "Rotation"])
            self.assertEqual(data.control_positions[0], [0.2, 0.1])
            self.assertEqual(data.metadata["time_unit"], "seconds")
            self.assertEqual(data.control_timestamps, [0.0, 0.002])
            self.assertEqual(data.state_velocities[1], [2.0, 1.0])
            self.assertEqual(data.state_torques[1], [22.0, 11.0])

    def test_initial_state_uses_first_motor_state_in_requested_joint_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_sage(root)

            state = load_initial_state(root, joint_names=["Pitch", "Rotation"])

        self.assertEqual(state.joint_names, ["Pitch", "Rotation"])
        self.assertEqual(state.positions, [0.4, 0.3])
        self.assertEqual(state.velocities, [0.0, 0.0])

    def test_benchmark_initial_state_loader_is_robot_generic(self):
        from anchor_bench.benchmark.init_sync import load_benchmark_initial_state

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_sage(root)

            state = load_benchmark_initial_state(root, joint_names=["Rotation"])

        self.assertEqual(state.positions, [0.3])
        self.assertEqual(state.velocities, [0.0])

    def test_loader_rejects_large_row_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_sage(root, control_rows=2, state_rows=5)
            with self.assertRaisesRegex(ValueError, "row counts differ"):
                load_real_data(root)

    def test_loader_interpolates_positions_to_target_dt_on_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_sage(root, control_rows=3, state_rows=3)

            data = load_real_data(root, target_dt=0.001, joint_names=["Rotation"])
            held = load_real_data(
                root, target_dt=0.001, joint_names=["Rotation"], hold_controls=True
            )

        self.assertEqual(data.metadata["time_unit"], "seconds")
        self.assertEqual(data.metadata["target_dt"], 0.001)
        self.assertEqual(data.control_timestamps, [0.0, 0.001, 0.002, 0.003, 0.004])
        self.assertEqual(data.state_timestamps, data.control_timestamps)
        np.testing.assert_allclose(data.control_positions, [[0.1], [0.6], [1.1], [1.6], [2.1]])
        np.testing.assert_allclose(data.state_positions, [[0.3], [0.8], [1.3], [1.8], [2.3]])

        np.testing.assert_allclose(data.state_velocities, [[0.0], [0.5], [1.0], [1.5], [2.0]])
        np.testing.assert_allclose(data.state_torques, [[10.0], [10.5], [11.0], [11.5], [12.0]])

        np.testing.assert_allclose(
            held.control_positions, [[0.1], [0.1], [1.1], [1.1], [2.1]]
        )

    def test_uniform_grid_discards_partial_tail_without_rounding_past_overlap(self):
        from anchor_bench.data.sage import _uniform_grid

        self.assertEqual([0.0, 0.002], _uniform_grid(0.0, 0.0032, 0.002))

    def test_loader_allows_position_only_state_outside_actuator_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_sage(root)
            (root / "state_motor.csv").write_text(
                'type,timestamp,positions\n'
                'STATE_MOTOR,0,"[0.3, 0.4]"\n'
                'STATE_MOTOR,2000,"[1.3, 1.4]"\n',
                encoding="utf-8",
            )

            data = load_real_data(root)
            self.assertEqual([], data.state_velocities)
            self.assertEqual([], data.state_torques)
            with self.assertRaisesRegex(ValueError, "actuator-training columns"):
                load_real_data(root, require_state_dynamics=True)

    def test_loader_accepts_seconds_without_scaling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_sage(root, control_rows=2, state_rows=2, time_unit="seconds")

            data = load_real_data(root)

        self.assertEqual(data.control_timestamps, [0.0, 2000.0])
        self.assertEqual(data.state_timestamps, [0.0, 2000.0])

    def test_loader_rejects_unsupported_time_unit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_sage(root, time_unit="milliseconds")

            with self.assertRaisesRegex(ValueError, "time_unit"):
                load_real_data(root)

    def test_loader_rejects_target_dt_when_traces_do_not_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_sage(root)
            (root / "state_motor.csv").write_text(
                'type,timestamp,positions,velocities,torques\n'
                'STATE_MOTOR,10000,"[0.3, 0.4]","[0.0, 0.0]","[0.0, 0.0]"\n'
                'STATE_MOTOR,12000,"[1.3, 1.4]","[0.0, 0.0]","[0.0, 0.0]"\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "do not overlap"):
                load_real_data(root, target_dt=0.001)

    def test_loader_rejects_wrong_state_vector_width(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_sage(root)
            (root / "state_motor.csv").write_text(
                'type,timestamp,positions,velocities,torques\n'
                'STATE_MOTOR,0,"[0.3, 0.4]","[0.0]","[10.0, 20.0]"\n'
                'STATE_MOTOR,2000,"[1.3, 1.4]","[1.0]","[11.0, 22.0]"\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "velocities row 0"):
                load_real_data(root)


if __name__ == "__main__":
    unittest.main()
