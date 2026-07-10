from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from anchor_bench.data.motion import JointLayout, MotionData, MotionSignal
from anchor_bench.data.sage import write_sage_motion
from anchor_bench.training.episode import (
    EpisodeAssembler,
    FeatureBlock,
    prepare_episode_source,
)
from anchor_bench.training.schema import (
    ResolvedTrainingMapping,
    TrainingFeature,
)


def _signal(time, values, name):
    return MotionSignal(np.asarray(time), np.asarray(values), "unit", name)


def _motion() -> MotionData:
    return MotionData(
        joints=JointLayout(("a", "b", "context")),
        command_position=_signal(
            [0.0, 0.2],
            [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]],
            "command_position",
        ),
        position=_signal(
            [0.0, 0.1, 0.2],
            [[0.0, 0.5, 1.0], [0.1, 0.6, 1.1], [0.2, 0.7, 1.2]],
            "position",
        ),
        velocity=_signal(
            [0.0, 0.1, 0.2],
            [[0.0, 0.1, 0.2], [0.3, 0.4, 0.5], [0.6, 0.7, 0.8]],
            "velocity",
        ),
        effort=_signal(
            [0.0, 0.1, 0.2],
            [[10.0, 20.0, 30.0], [11.0, 21.0, 31.0], [12.0, 22.0, 32.0]],
            "effort",
        ),
        metadata={},
        source=Path("episode"),
    )


def _write_alias_motion(root: Path, names: tuple[str, ...]) -> None:
    """Write one training episode with a three-column recorded joint axis."""

    write_sage_motion(
        root,
        names,
        control_rows=(
            ("CONTROL", 0.0, [1.0, 2.0, 9.0]),
            ("CONTROL", 0.2, [1.2, 2.2, 9.2]),
        ),
        state_rows=(
            ("STATE_MOTOR", 0.0, [3.0, 4.0, 8.0], [0.1, 0.2, 0.3], [5.0, 6.0, 7.0]),
            ("STATE_MOTOR", 0.1, [3.1, 4.1, 8.1], [0.4, 0.5, 0.6], [5.1, 6.1, 7.1]),
            ("STATE_MOTOR", 0.2, [3.2, 4.2, 8.2], [0.7, 0.8, 0.9], [5.2, 6.2, 7.2]),
        ),
        metadata={
            "time_unit": "seconds",
            "position_unit": "rad",
            "velocity_unit": "rad/s",
            "torque_unit": "N*m",
        },
    )


class _BiasProvider:
    def __init__(self):
        self.calls = []

    def compute(self, **kwargs):
        self.calls.append(kwargs)
        rows = len(kwargs["positions"])
        width = len(kwargs["requested_joint_names"])
        return SimpleNamespace(
            values=np.arange(rows * width, dtype=np.float32).reshape(rows, width),
            provenance={"source": "fake", "requested_joint_names": list(kwargs["requested_joint_names"])},
        )


class FeatureBlockTests(unittest.TestCase):
    def test_feature_block_supports_multiple_components(self):
        block = FeatureBlock(
            np.zeros((3, 2, 3), dtype=np.float32),
            "input",
            ("vx", "vy", "vz"),
        )

        self.assertEqual((3, 2, 3), block.values.shape)
        self.assertFalse(block.values.flags.writeable)


class EpisodeAssemblerTests(unittest.TestCase):
    def test_load_canonicalizes_aliases_and_keeps_dynamic_bias_context(self):
        """Canonicalize mapped columns without dropping an unmapped recorded joint."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_alias_motion(root, ("AliasA", "b", "context"))
            provider = _BiasProvider()
            assembler = EpisodeAssembler(
                provider,
                canonical_joint_names=("a", "b"),
                joint_aliases={"AliasA": "a"},
            )
            source = assembler.load(root, 0.1)
            episode = assembler.assemble(
                source,
                mapping=ResolvedTrainingMapping(
                    input_joints=("b", "a"),
                    output_joints=("b",),
                ),
                features=(TrainingFeature(name="dynamic_bias", domain="input"),),
                target="torque",
            )

        self.assertEqual(("a", "b", "context"), source.motion.joints.names)
        self.assertEqual(("a", "b", "context"), provider.calls[0]["joint_names"])
        self.assertEqual((3, 3), provider.calls[0]["positions"].shape)
        np.testing.assert_allclose(episode.measured_torque[:, 0], [6.0, 6.1, 6.2])

    def test_preserves_state_grid_and_linearly_interpolates_commands(self):
        source = prepare_episode_source(_motion(), 0.1)
        mapping = ResolvedTrainingMapping(
            input_joints=("b", "a"),
            output_joints=("b",),
        )
        episode = EpisodeAssembler().assemble(
            source,
            mapping=mapping,
            features=(
                TrainingFeature(name="position", domain="input"),
                TrainingFeature(name="position_error", domain="input"),
                TrainingFeature(name="velocity", domain="input"),
            ),
            target="torque",
        )

        np.testing.assert_allclose(episode.timestamps, [0.0, 0.1, 0.2])
        np.testing.assert_allclose(
            episode.feature_blocks["position"].values[:, :, 0],
            [[0.5, 0.0], [0.6, 0.1], [0.7, 0.2]],
        )
        np.testing.assert_allclose(
            episode.feature_blocks["position_error"].values[:, :, 0],
            [[1.5, 1.0], [2.4, 1.9], [3.3, 2.8]],
        )
        np.testing.assert_allclose(episode.measured_torque[:, 0], [20.0, 21.0, 22.0])

    def test_dynamic_bias_receives_full_recorded_state_before_mapping_selection(self):
        provider = _BiasProvider()
        source = prepare_episode_source(_motion(), 0.1)
        mapping = ResolvedTrainingMapping(
            input_joints=("b", "a"),
            output_joints=("b",),
        )
        episode = EpisodeAssembler(provider).assemble(
            source,
            mapping=mapping,
            features=(TrainingFeature(name="dynamic_bias", domain="input"),),
            target="torque",
            dynamic_bias_cache_key="content-hash",
        )

        call = provider.calls[0]
        self.assertEqual(("a", "b", "context"), call["joint_names"])
        self.assertEqual(("b", "a"), call["requested_joint_names"])
        self.assertEqual((3, 3), call["positions"].shape)
        self.assertEqual("content-hash", call["cache_key"])
        self.assertEqual((3, 2, 1), episode.feature_blocks["dynamic_bias"].values.shape)

    def test_residual_target_retains_physics_sources_without_exposing_them_as_inputs(self):
        source = prepare_episode_source(_motion(), 0.1)
        episode = EpisodeAssembler().assemble(
            source,
            mapping=ResolvedTrainingMapping(input_joints=("a", "b"), output_joints=("b",)),
            features=(TrainingFeature(name="position", domain="input"),),
            target="torque_residual",
        )

        self.assertEqual({"position", "position_error", "velocity"}, set(episode.feature_blocks))

    def test_missing_input_joint_fails_but_unselected_robot_joints_are_irrelevant(self):
        source = prepare_episode_source(_motion(), 0.1)
        assembler = EpisodeAssembler()
        with self.assertRaisesRegex(ValueError, "joints not found.*missing"):
            assembler.assemble(
                source,
                mapping=ResolvedTrainingMapping(input_joints=("a", "missing"), output_joints=("a",)),
                features=(TrainingFeature(name="position", domain="input"),),
                target="torque",
            )

        episode = assembler.assemble(
            source,
            mapping=ResolvedTrainingMapping(input_joints=("a",), output_joints=("a",)),
            features=(TrainingFeature(name="position", domain="input"),),
            target="torque",
        )
        self.assertEqual((3, 1), episode.measured_torque.shape)

    def test_cadence_mismatch_fails_without_resampling(self):
        with self.assertRaisesRegex(ValueError, "median state cadence"):
            prepare_episode_source(_motion(), 0.05)

    def test_recorded_scalar_features_reject_unavailable_component_layouts(self):
        source = prepare_episode_source(_motion(), 0.1)
        with self.assertRaisesRegex(ValueError, "provides only the 'value' component"):
            EpisodeAssembler().assemble(
                source,
                mapping=ResolvedTrainingMapping(input_joints=("a",), output_joints=("a",)),
                features=(
                    TrainingFeature(
                        name="velocity",
                        domain="input",
                        components=("vx", "vy", "vz"),
                    ),
                ),
                target="torque",
            )


if __name__ == "__main__":
    unittest.main()
