import tempfile
import unittest
from pathlib import Path

import numpy as np

from anchor_bench.data.sage import load_sage_motion


def write_sage(root: Path, control_rows: int = 2, state_rows: int = 2, time_unit: str = "microseconds"):
    (root / "joint_list.txt").write_text("Rotation\nPitch\n", encoding="utf-8")
    (root / "metadata.yaml").write_text(
        f"schema: sage\ntime_unit: {time_unit}\nposition_unit: rad\nvelocity_unit: rad/s\ntorque_unit: N*m\n",
        encoding="utf-8",
    )
    control = ["type,timestamp,positions"]
    state = ["type,timestamp,positions,velocities,torques"]
    for i in range(control_rows):
        control.append(f'CONTROL,{i * 2000},"[{0.1 + i}, {0.2 + i}]"')
    for i in range(state_rows):
        state.append(f'STATE_MOTOR,{i * 2000},"[{0.3 + i}, {0.4 + i}]","[0.0, 0.0]","[0.0, 0.0]"')
    (root / "control.csv").write_text("\n".join(control) + "\n", encoding="utf-8")
    (root / "state_motor.csv").write_text("\n".join(state) + "\n", encoding="utf-8")


def write_named_sage(root: Path, names: tuple[str, ...]) -> None:
    """Write one two-row SAGE motion with a caller-selected three-column axis."""

    if len(names) != 3:
        raise ValueError("test joint axis must contain three names")
    (root / "joint_list.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    (root / "metadata.yaml").write_text(
        "time_unit: seconds\nposition_unit: rad\nvelocity_unit: rad/s\ntorque_unit: N*m\n",
        encoding="utf-8",
    )
    (root / "control.csv").write_text(
        'type,timestamp,positions\nCONTROL,0,"[1.0, 2.0, 9.0]"\n'
        'CONTROL,0.1,"[1.1, 2.1, 9.1]"\n',
        encoding="utf-8",
    )
    (root / "state_motor.csv").write_text(
        "type,timestamp,positions,velocities,torques\n"
        'STATE_MOTOR,0,"[3.0, 4.0, 8.0]","[0.1, 0.2, 0.3]","[5.0, 6.0, 7.0]"\n'
        'STATE_MOTOR,0.1,"[3.1, 4.1, 8.1]","[0.4, 0.5, 0.6]","[5.1, 6.1, 7.1]"\n',
        encoding="utf-8",
    )


class SageLoaderTests(unittest.TestCase):
    def test_alias_axis_canonicalizes_before_selection_and_preserves_context(self):
        """Resolve known aliases without dropping an unmapped dynamic-bias column."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_named_sage(root, ("AliasA", "b", "context"))
            complete = load_sage_motion(
                root,
                canonical_joint_names=("a", "b"),
                joint_aliases={"AliasA": "a"},
            )
            selected = load_sage_motion(
                root,
                joint_names=("b", "a"),
                canonical_joint_names=("a", "b"),
                joint_aliases={"AliasA": "a"},
            )

        self.assertEqual(["a", "b", "context"], complete.joint_names)
        self.assertEqual(["b", "a"], selected.joint_names)
        np.testing.assert_allclose(selected.position.values[0], [4.0, 3.0])

    def test_alias_collision_and_missing_selected_joint_fail(self):
        """Reject duplicate physical columns while retaining normal missing checks."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_named_sage(root, ("AliasA", "a", "context"))
            with self.assertRaisesRegex(ValueError, "maps columns 'AliasA' and 'a'.*joint 'a'"):
                load_sage_motion(
                    root,
                    canonical_joint_names=("a", "b"),
                    joint_aliases={"AliasA": "a"},
                )

            write_named_sage(root, ("AliasA", "b", "context"))
            with self.assertRaisesRegex(ValueError, "joints not found.*missing"):
                load_sage_motion(
                    root,
                    joint_names=("a", "missing"),
                    canonical_joint_names=("a", "b"),
                    joint_aliases={"AliasA": "a"},
                )

    def test_loader_preserves_requested_joint_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_sage(root)
            motion = load_sage_motion(root, joint_names=["Pitch", "Rotation"])
            self.assertEqual(motion.joint_names, ["Pitch", "Rotation"])
            self.assertEqual(motion.command_position.values[0].tolist(), [0.2, 0.1])
            self.assertEqual(motion.metadata["time_unit"], "seconds")
            self.assertEqual(motion.command_position.time.tolist(), [0.0, 0.002])

    def test_canonical_motion_carries_all_signals_and_derives_training_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_sage(root)

            motion = load_sage_motion(root, joint_names=["Pitch", "Rotation"])
            columns = motion.training_columns("Pitch")

        self.assertEqual(["Pitch", "Rotation"], motion.joint_names)
        self.assertEqual((2, 2), motion.velocity.values.shape)
        self.assertEqual((2, 2), motion.effort.values.shape)
        np.testing.assert_allclose(columns["command_position"], [0.2, 1.2])
        np.testing.assert_allclose(columns["position"], [0.4, 1.4])
        np.testing.assert_allclose(columns["position_error"], [-0.2, -0.2])
        self.assertFalse(motion.position.values.flags.writeable)

    def test_canonical_motion_rejects_duplicate_joints_and_vector_width_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_sage(root)
            (root / "joint_list.txt").write_text("Rotation\nRotation\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate joint names"):
                load_sage_motion(root)

            (root / "joint_list.txt").write_text("Rotation\nPitch\n", encoding="utf-8")
            (root / "control.csv").write_text(
                'type,timestamp,positions\nCONTROL,0,"[0.0]"\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "joint_list.txt has 2 joints"):
                load_sage_motion(root)

    def test_canonical_motion_rejects_duplicate_timestamps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_sage(root)
            (root / "control.csv").write_text(
                'type,timestamp,positions\nCONTROL,0,"[0.0, 0.0]"\nCONTROL,0,"[1.0, 1.0]"\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                load_sage_motion(root)

    def test_initial_state_uses_first_motor_state_in_requested_joint_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_sage(root)

            state = load_sage_motion(
                root, joint_names=["Pitch", "Rotation"]
            ).initial_state()

        self.assertEqual(list(state.joints.names), ["Pitch", "Rotation"])
        self.assertEqual(state.positions.tolist(), [0.4, 0.3])
        self.assertEqual(state.velocities.tolist(), [0.0, 0.0])

    def test_loader_interpolates_positions_to_target_dt_on_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_sage(root, control_rows=3, state_rows=3)

            motion = load_sage_motion(
                root, target_dt=0.001, joint_names=["Rotation"]
            )

        self.assertEqual(motion.metadata["time_unit"], "seconds")
        self.assertEqual(motion.metadata["target_dt"], 0.001)
        self.assertEqual(
            motion.command_position.time.tolist(),
            [0.0, 0.001, 0.002, 0.003, 0.004],
        )
        self.assertEqual(motion.position.time.tolist(), motion.command_position.time.tolist())
        np.testing.assert_allclose(
            motion.command_position.values,
            [[0.1], [0.6], [1.1], [1.6], [2.1]],
        )
        np.testing.assert_allclose(
            motion.position.values,
            [[0.3], [0.8], [1.3], [1.8], [2.3]],
        )

    def test_loader_accepts_seconds_without_scaling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_sage(root, control_rows=2, state_rows=2, time_unit="seconds")

            motion = load_sage_motion(root)

        self.assertEqual(motion.command_position.time.tolist(), [0.0, 2000.0])
        self.assertEqual(motion.position.time.tolist(), [0.0, 2000.0])

    def test_loader_rejects_unsupported_time_unit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_sage(root, time_unit="milliseconds")

            with self.assertRaisesRegex(ValueError, "time_unit"):
                load_sage_motion(root)

    def test_loader_rejects_target_dt_when_traces_do_not_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_sage(root)
            (root / "state_motor.csv").write_text(
                "type,timestamp,positions,velocities,torques\n"
                'STATE_MOTOR,10000,"[0.3, 0.4]","[0.0, 0.0]","[0.0, 0.0]"\n'
                'STATE_MOTOR,12000,"[1.3, 1.4]","[0.0, 0.0]","[0.0, 0.0]"\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "do not overlap"):
                load_sage_motion(root, target_dt=0.001)


if __name__ == "__main__":
    unittest.main()
