from __future__ import annotations

import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from anchor_bench.data.motion import JointLayout, MotionData, MotionSignal
from anchor_bench.training.dynamic_bias import NewtonMuJoCoDynamicBiasProvider


class _Array:
    """Small host-array stand-in for the Warp arrays used by the provider."""

    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32).copy()

    def numpy(self):
        return self.values.copy()

    def assign(self, values):
        self.values = np.asarray(values, dtype=np.float32).copy()


def _motion(rows: int = 2) -> MotionData:
    """Build two recorded joints while leaving one model joint unrecorded."""

    time = np.arange(rows, dtype=float)
    sequence = np.arange(1, rows + 1, dtype=float)
    position = np.column_stack((sequence, 10.0 * sequence))
    velocity = np.column_stack((0.2 * sequence - 0.1, 0.2 * sequence))
    joints = JointLayout(("JointA", "JointB"))
    return MotionData(
        joints=joints,
        command_position=MotionSignal(
            time,
            position,
            "rad",
            "command_position",
        ),
        position=MotionSignal(
            time,
            position,
            "rad",
            "position",
        ),
        velocity=MotionSignal(
            time,
            velocity,
            "rad/s",
            "velocity",
        ),
        effort=None,
        metadata={},
        source=Path("fixture"),
    )


def _provider(batch_size: int = 2) -> tuple[
    NewtonMuJoCoDynamicBiasProvider, list[tuple[np.ndarray, np.ndarray]], list[None]
]:
    """Construct an initialized provider around a MuJoCo-Warp-shaped fake."""

    class JointType:
        REVOLUTE = 1
        PRISMATIC = 2

    builder = SimpleNamespace(
        joint_label=["JointA_joint", "JointB_joint", "JointC_joint"],
        joint_type=[JointType.REVOLUTE] * 3,
        joint_dof_dim=[(1,), (1,), (1,)],
        joint_q_start=[0, 1, 2],
        joint_qd_start=[0, 1, 2],
    )
    default_q = np.tile([5.0, 6.0, 7.0], batch_size)
    model = SimpleNamespace(
        device="cpu",
        joint_dof_count=3 * batch_size,
        joint_q=_Array(default_q),
    )
    state = SimpleNamespace(
        joint_q=_Array(default_q),
        joint_qd=_Array(np.zeros(3 * batch_size, dtype=np.float32)),
    )
    mjw_data = SimpleNamespace(
        qpos=None,
        qvel=None,
        qfrc_bias=_Array(np.zeros((batch_size, 3), dtype=np.float32)),
    )
    captured: list[tuple[np.ndarray, np.ndarray]] = []
    forward_calls: list[None] = []
    mj_qd_start = np.asarray([1, 0, 2], dtype=np.int64)

    def sync(data, _model, current_state):
        q = current_state.joint_q.numpy().reshape(batch_size, 3)
        qd = current_state.joint_qd.numpy().reshape(batch_size, 3)
        captured.append((q.copy(), qd.copy()))
        data.qpos, data.qvel = q, qd

    def forward(_model, data):
        forward_calls.append(None)
        bias = np.empty((batch_size, 3), dtype=np.float32)
        for world in range(batch_size):
            coupled = float(data.qpos[world].sum())
            for newton_index, mujoco_index in enumerate(mj_qd_start):
                bias[world, mujoco_index] = (
                    100.0 * data.qpos[world, newton_index]
                    + data.qvel[world, newton_index]
                    + coupled
                )
        data.qfrc_bias.assign(bias)

    solver = SimpleNamespace(
        model=model,
        mjw_model=object(),
        mjw_data=mjw_data,
        _mujoco_warp=SimpleNamespace(forward=forward),
        _update_mjc_data=sync,
    )
    provider = NewtonMuJoCoDynamicBiasProvider(SimpleNamespace(), batch_size=batch_size)
    provider._initialized = True
    provider._newton = SimpleNamespace(JointType=JointType)
    provider._warp = SimpleNamespace(ScopedDevice=lambda _device: nullcontext())
    provider._builder = builder
    provider._model = model
    provider._solver = solver
    provider._state = state
    provider._default_q = model.joint_q.numpy()
    provider._mj_qd_start = mj_qd_start
    provider._coordinates_per_world = 3
    provider._dofs_per_world = 3
    return provider, captured, forward_calls


class DynamicBiasProviderTests(unittest.TestCase):
    def test_full_recorded_state_is_injected_and_output_is_reordered(self):
        provider, captured, forward_calls = _provider()

        values = provider.compute(_motion(), ("JointB", "JointA"))

        np.testing.assert_allclose(
            values,
            [[1018.2, 118.1], [2029.4, 229.3]],
            rtol=1e-6,
        )
        self.assertEqual(1, len(captured))
        self.assertEqual(1, len(forward_calls))
        self.assertFalse(hasattr(provider._solver, "eval_dynamic_bias"))
        np.testing.assert_allclose(captured[0][0][0], [1.0, 10.0, 7.0])
        np.testing.assert_allclose(captured[0][1][0], [0.1, 0.2, 0.0])

    def test_requested_joint_must_exist_in_recording(self):
        provider, _, _ = _provider()

        with self.assertRaisesRegex(ValueError, "joints not found"):
            provider.compute(_motion(), ("JointC",))

    def test_unrequested_recorded_joint_is_still_injected(self):
        """Bias evaluation uses all recorded mapped state, not only requested slots."""

        provider, captured, _ = _provider()

        values = provider.compute(_motion(), ("JointA",))

        self.assertEqual((2, 1), values.shape)
        np.testing.assert_allclose(captured[0][0][0], [1.0, 10.0, 7.0])

    def test_partial_batch_resets_inactive_world_to_defaults(self):
        provider, captured, forward_calls = _provider(batch_size=2)

        values = provider.compute(_motion(rows=3), ("JointA",))

        self.assertEqual((3, 1), values.shape)
        self.assertEqual(2, len(forward_calls))
        np.testing.assert_allclose(captured[1][0][1], [5.0, 6.0, 7.0])
        np.testing.assert_allclose(captured[1][1][1], [0.0, 0.0, 0.0])

    def test_batch_size_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "batch_size"):
            NewtonMuJoCoDynamicBiasProvider(SimpleNamespace(), batch_size=0)


if __name__ == "__main__":
    unittest.main()
