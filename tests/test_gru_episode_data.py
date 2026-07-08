from __future__ import annotations

import operator
import unittest
from collections.abc import MutableMapping
from pathlib import Path
from typing import cast

import numpy as np

from anchor_bench.data.motion import JointLayout, MotionData, MotionSignal
from anchor_bench.training.data import (
    FeatureBlock,
    FeatureDomain,
    TrainingEpisode,
    assemble_training_episode,
)
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


class FeatureBlockTests(unittest.TestCase):
    def test_canonicalizes_values_and_is_immutable(self):
        source = np.arange(12, dtype=np.float64).reshape(2, 2, 3)

        block = FeatureBlock(
            values=source,
            domain="input_joints",
            channels=("vx", "vy", "vz"),
        )

        self.assertEqual(np.dtype(np.float32), block.values.dtype)
        self.assertEqual((2, 2, 3), block.values.shape)
        self.assertEqual(("vx", "vy", "vz"), block.channels)
        self.assertFalse(block.values.flags.writeable)
        source.fill(-1.0)
        self.assertEqual(0.0, float(block.values[0, 0, 0]))
        with self.assertRaises(ValueError):
            block.values[0, 0, 0] = 1.0
        with self.assertRaises(ValueError):
            block.values.setflags(write=True)

    def test_rejects_invalid_shape_domain_and_channels(self):
        with self.assertRaisesRegex(ValueError, "3-D"):
            FeatureBlock(
                values=np.zeros((2, 2), dtype=np.float32),
                domain="input_joints",
                channels=("value",),
            )
        with self.assertRaisesRegex(ValueError, "non-finite"):
            FeatureBlock(
                values=np.full((2, 2, 1), np.nan, dtype=np.float32),
                domain="input_joints",
                channels=("value",),
            )
        with self.assertRaisesRegex(ValueError, "domain must be one of"):
            FeatureBlock(
                values=np.zeros((2, 2, 1), dtype=np.float32),
                domain=cast(FeatureDomain, "body"),
                channels=("value",),
            )
        with self.assertRaisesRegex(ValueError, "at least one channel"):
            FeatureBlock(
                values=np.zeros((2, 2, 0), dtype=np.float32),
                domain="input_joints",
                channels=(),
            )
        with self.assertRaisesRegex(ValueError, "duplicate.*channels"):
            FeatureBlock(
                values=np.zeros((2, 2, 2), dtype=np.float32),
                domain="input_joints",
                channels=("vx", "vx"),
            )
        with self.assertRaisesRegex(ValueError, "names must be non-empty"):
            FeatureBlock(
                values=np.zeros((2, 2, 1), dtype=np.float32),
                domain="input_joints",
                channels=("",),
            )
        with self.assertRaisesRegex(ValueError, "channel axis must match"):
            FeatureBlock(
                values=np.zeros((2, 2, 2), dtype=np.float32),
                domain="input_joints",
                channels=("value",),
            )
        with self.assertRaisesRegex(ValueError, "non-empty strings"):
            FeatureBlock(
                values=np.zeros((2, 2, 1), dtype=np.float32),
                domain="input_joints",
                channels=(cast(str, 1),),
            )
        with self.assertRaisesRegex(ValueError, "global.*width 1"):
            FeatureBlock(
                values=np.zeros((2, 2, 1), dtype=np.float32),
                domain="global",
                channels=("value",),
            )


class TrainingEpisodeContractTests(unittest.TestCase):
    def test_validates_joint_and_feature_axes_and_freezes_mapping(self):
        block = FeatureBlock(
            values=np.zeros((2, 2, 1), dtype=np.float32),
            domain="input_joints",
            channels=("value",),
        )
        episode = TrainingEpisode(
            timestamps=np.array([0.0, 1.0]),
            input_joints=("j0", "j1"),
            output_joints=("j1",),
            feature_blocks={"position": block},
            target_torque=np.zeros((2, 1), dtype=np.float32),
        )
        for array in (episode.timestamps, episode.target_torque):
            self.assertFalse(array.flags.writeable)
            with self.assertRaises(ValueError):
                array.setflags(write=True)

        with self.assertRaises(TypeError):
            operator.setitem(
                cast(MutableMapping[str, FeatureBlock], episode.feature_blocks),
                "position",
                block,
            )
        with self.assertRaisesRegex(ValueError, "duplicate input_joints"):
            TrainingEpisode(
                timestamps=np.array([0.0, 1.0]),
                input_joints=("j0", "j0"),
                output_joints=("j0",),
                feature_blocks={"position": block},
                target_torque=np.zeros((2, 1), dtype=np.float32),
            )
        with self.assertRaisesRegex(ValueError, "subset of input_joints"):
            TrainingEpisode(
                timestamps=np.array([0.0, 1.0]),
                input_joints=("j0", "j1"),
                output_joints=("missing",),
                feature_blocks={"position": block},
                target_torque=np.zeros((2, 1), dtype=np.float32),
            )
        with self.assertRaisesRegex(ValueError, "input_joints axis has width"):
            TrainingEpisode(
                timestamps=np.array([0.0, 1.0]),
                input_joints=("j0",),
                output_joints=("j0",),
                feature_blocks={"position": block},
                target_torque=np.zeros((2, 1), dtype=np.float32),
            )
        short_block = FeatureBlock(
            values=np.zeros((1, 2, 1), dtype=np.float32),
            domain="input_joints",
            channels=("value",),
        )
        with self.assertRaisesRegex(ValueError, "one row per timestamp"):
            TrainingEpisode(
                timestamps=np.array([0.0, 1.0]),
                input_joints=("j0", "j1"),
                output_joints=("j1",),
                feature_blocks={"position": short_block},
                target_torque=np.zeros((2, 1), dtype=np.float32),
            )

    def test_rejects_empty_and_non_string_episode_identifiers(self):
        block = FeatureBlock(
            values=np.zeros((1, 1, 1), dtype=np.float32),
            domain="input_joints",
            channels=("value",),
        )
        joint_cases = (
            ("input_joints", ("",), ("j0",)),
            ("input_joints", (cast(str, 1),), ("j0",)),
            ("output_joints", ("j0",), ("",)),
            ("output_joints", ("j0",), (cast(str, 1),)),
        )
        for label, input_joints, output_joints in joint_cases:
            with self.subTest(label=label, joints=(input_joints, output_joints)):
                with self.assertRaisesRegex(
                    ValueError,
                    f"{label}.*non-empty strings",
                ):
                    TrainingEpisode(
                        timestamps=np.array([0.0]),
                        input_joints=input_joints,
                        output_joints=output_joints,
                        feature_blocks={"position": block},
                        target_torque=np.zeros((1, 1), dtype=np.float32),
                    )
        invalid_joint_collections = (
            cast(tuple[str, ...], "j0"),
            cast(tuple[str, ...], 1),
        )
        for input_joints in invalid_joint_collections:
            with self.subTest(input_joints=input_joints):
                with self.assertRaisesRegex(ValueError, "ordered sequence"):
                    TrainingEpisode(
                        timestamps=np.array([0.0]),
                        input_joints=input_joints,
                        output_joints=("j0",),
                        feature_blocks={"position": block},
                        target_torque=np.zeros((1, 1), dtype=np.float32),
                    )

        invalid_feature_maps = (
            {"": block},
            {cast(str, 1): block, "1": block},
        )
        for feature_blocks in invalid_feature_maps:
            with self.subTest(feature_names=tuple(feature_blocks)):
                with self.assertRaisesRegex(ValueError, "feature names.*non-empty"):
                    TrainingEpisode(
                        timestamps=np.array([0.0]),
                        input_joints=("j0",),
                        output_joints=("j0",),
                        feature_blocks=feature_blocks,
                        target_torque=np.zeros((1, 1), dtype=np.float32),
                    )



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
        for name in ("position", "position_error", "velocity", "dynamic_bias"):
            block = episode.feature_blocks[name]
            self.assertEqual("input_joints", block.domain)
            self.assertEqual(("value",), block.channels)
            self.assertEqual((3, 2, 1), block.values.shape)
            self.assertEqual(np.dtype(np.float32), block.values.dtype)
            self.assertFalse(block.values.flags.writeable)
        np.testing.assert_allclose(
            episode.feature_blocks["position"].values[:, :, 0],
            [[2.0, 1.0], [5.0, 4.0], [8.0, 7.0]],
        )
        # Commands remain linearly interpolated onto the recorded state grid.
        np.testing.assert_allclose(
            episode.feature_blocks["position_error"].values[:, :, 0],
            [[8.0, -1.0], [15.0, 6.0], [22.0, 13.0]],
        )
        np.testing.assert_allclose(
            episode.feature_blocks["velocity"].values[:, :, 0],
            [[0.2, 0.1], [1.2, 1.1], [2.2, 2.1]],
        )
        np.testing.assert_allclose(episode.target_torque, [[200.0], [300.0], [400.0]])
        self.assertNotIn("previous_torque", episode.feature_blocks)
        np.testing.assert_allclose(
            episode.feature_blocks["dynamic_bias"].values[:, :, 0],
            [[4.0, 2.0], [17.0, 15.0], [30.0, 28.0]],
        )
        provenance = episode.dynamic_bias_provenance
        assert provenance is not None
        self.assertEqual(
            {"provider": "fake", "mapped": ["context", "output", "extra"]},
            dict(provenance),
        )

        self.assertEqual(1, len(provider.calls))
        call = provider.calls[0]
        self.assertEqual(("context", "output", "extra"), call["joint_names"])
        self.assertEqual(("output", "context"), call["requested_joint_names"])
        self.assertEqual((3, 3), np.asarray(call["positions"]).shape)
        velocities = call["velocities"]
        assert isinstance(velocities, np.ndarray)
        np.testing.assert_allclose(
            velocities,
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
