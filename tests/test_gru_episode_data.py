from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from anchor_bench.data.motion import JointLayout, MotionData, MotionSignal
from anchor_bench.training.data import assemble_training_episode
from anchor_bench.training.dynamic_bias import DynamicBiasResult


def _signal(
    name: str,
    time: list[float],
    values: list[list[float]],
    unit: str,
) -> MotionSignal:
    return MotionSignal(
        time=np.asarray(time, dtype=float),
        values=np.asarray(values, dtype=float),
        unit=unit,
        name=name,
    )


def _motion(*, state_time: list[float] | None = None) -> MotionData:
    timestamps = state_time or [0.0, 1.0, 2.0]
    return MotionData(
        joints=JointLayout(("context", "output", "extra")),
        command_position=_signal(
            "command_position",
            [0.0, 2.0],
            [[0.0, 10.0, 20.0], [20.0, 30.0, 40.0]],
            "rad",
        ),
        position=_signal(
            "position",
            timestamps,
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ],
            "rad",
        ),
        velocity=_signal(
            "velocity",
            [0.0, 2.0],
            [[0.1, 0.2, 0.3], [2.1, 2.2, 2.3]],
            "rad/s",
        ),
        effort=_signal(
            "effort",
            [0.0, 2.0],
            [[100.0, 200.0, 300.0], [300.0, 400.0, 500.0]],
            "N*m",
        ),
        metadata={},
        source=Path("episode"),
    )


class _FakeBiasProvider:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def compute(
        self,
        *,
        joint_names,
        positions,
        velocities,
        requested_joint_names,
    ) -> DynamicBiasResult:
        self.calls.append(
            {
                "joint_names": tuple(joint_names),
                "positions": np.array(positions, copy=True),
                "velocities": np.array(velocities, copy=True),
                "requested_joint_names": tuple(requested_joint_names),
            }
        )
        indices = [joint_names.index(name) for name in requested_joint_names]
        values = positions[:, indices] + 10.0 * velocities[:, indices]
        return DynamicBiasResult(values, {"provider": "fake", "mapped": list(joint_names)})


class GRUEpisodeDataTests(unittest.TestCase):
    def test_preserves_state_grid_and_builds_ordered_feature_blocks(self):
        provider = _FakeBiasProvider()
        episode = assemble_training_episode(
            _motion(),
            input_joints=["output", "context"],
            output_joints=["output"],
            input_columns=[
                "position",
                "position_error",
                "velocity",
                "dynamic_bias",
                "previous_torque",
            ],
            sample_dt_s=1.0,
            dynamic_bias_provider=provider,
        )

        self.assertEqual(("output", "context"), episode.input_joints)
        self.assertEqual(("output",), episode.output_joints)
        np.testing.assert_allclose(episode.timestamps, [0.0, 1.0, 2.0])
        np.testing.assert_allclose(
            episode.feature_blocks["position"],
            [[2.0, 1.0], [5.0, 4.0], [8.0, 7.0]],
        )
        # Commands remain linearly interpolated onto the recorded state grid.
        np.testing.assert_allclose(
            episode.feature_blocks["position_error"],
            [[8.0, -1.0], [15.0, 6.0], [22.0, 13.0]],
        )
        np.testing.assert_allclose(
            episode.feature_blocks["velocity"],
            [[0.2, 0.1], [1.2, 1.1], [2.2, 2.1]],
        )
        np.testing.assert_allclose(episode.target_torque, [[200.0], [300.0], [400.0]])
        self.assertNotIn("previous_torque", episode.feature_blocks)
        np.testing.assert_allclose(
            episode.feature_blocks["dynamic_bias"],
            [[4.0, 2.0], [17.0, 15.0], [30.0, 28.0]],
        )
        self.assertEqual(
            {"provider": "fake", "mapped": ["context", "output", "extra"]},
            dict(episode.dynamic_bias_provenance),
        )

        self.assertEqual(1, len(provider.calls))
        call = provider.calls[0]
        self.assertEqual(("context", "output", "extra"), call["joint_names"])
        self.assertEqual(("output", "context"), call["requested_joint_names"])
        self.assertEqual((3, 3), np.asarray(call["positions"]).shape)
        np.testing.assert_allclose(
            call["velocities"],
            [[0.1, 0.2, 0.3], [1.1, 1.2, 1.3], [2.1, 2.2, 2.3]],
        )

    def test_rejects_missing_input_joint_and_non_subset_output(self):
        with self.assertRaisesRegex(ValueError, "joints not found.*missing"):
            assemble_training_episode(
                _motion(),
                input_joints=["output", "missing"],
                sample_dt_s=1.0,
            )
        with self.assertRaisesRegex(ValueError, "subset of input_joints"):
            assemble_training_episode(
                _motion(),
                input_joints=["context"],
                output_joints=["output"],
                sample_dt_s=1.0,
            )

    def test_rejects_cadence_mismatch_and_missing_provider(self):
        with self.assertRaisesRegex(ValueError, "state cadence does not match"):
            assemble_training_episode(
                _motion(state_time=[0.0, 1.0, 2.1]),
                input_joints=["output"],
                sample_dt_s=1.0,
            )
        with self.assertRaisesRegex(ValueError, "requires a DynamicBiasProvider"):
            assemble_training_episode(
                _motion(),
                input_joints=["output"],
                input_columns=["dynamic_bias"],
                sample_dt_s=1.0,
            )


if __name__ == "__main__":
    unittest.main()
