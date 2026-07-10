from __future__ import annotations

import unittest
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


def _motion() -> MotionData:
    """Build two recorded joints while leaving one model joint unrecorded."""

    time = np.asarray([0.0, 1.0])
    joints = JointLayout(("JointA", "JointB"))
    return MotionData(
        joints=joints,
        command_position=MotionSignal(
            time,
            np.asarray([[1.0, 10.0], [2.0, 20.0]]),
            "rad",
            "command_position",
        ),
        position=MotionSignal(
            time,
            np.asarray([[1.0, 10.0], [2.0, 20.0]]),
            "rad",
            "position",
        ),
        velocity=MotionSignal(
            time,
            np.asarray([[0.1, 0.2], [0.3, 0.4]]),
            "rad/s",
            "velocity",
        ),
        effort=None,
        metadata={},
        source=Path("fixture"),
    )


def _provider() -> tuple[
    NewtonMuJoCoDynamicBiasProvider, list[tuple[np.ndarray, np.ndarray]], list[None]
]:
    """Construct an initialized provider around an upstream-1.4-shaped fake."""

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
    model = SimpleNamespace(joint_dof_count=3, joint_q=_Array([5.0, 6.0, 7.0]))
    state = SimpleNamespace(joint_q=_Array([5.0, 6.0, 7.0]), joint_qd=_Array([0.0, 0.0, 0.0]))
    mj_data = SimpleNamespace(qpos=None, qvel=None, qfrc_bias=np.zeros(3, dtype=np.float32))
    captured: list[tuple[np.ndarray, np.ndarray]] = []
    forward_calls: list[None] = []
    mj_qd_start = np.asarray([1, 0, 2], dtype=np.int64)

    def sync(data, _model, current_state):
        q = current_state.joint_q.numpy()
        qd = current_state.joint_qd.numpy()
        captured.append((q, qd))
        data.qpos, data.qvel = q, qd

    def forward(_model, data):
        forward_calls.append(None)
        bias = np.empty(3, dtype=np.float32)
        coupled = float(data.qpos.sum())
        for newton_index, mujoco_index in enumerate(mj_qd_start):
            bias[mujoco_index] = 100.0 * data.qpos[newton_index] + data.qvel[newton_index] + coupled
        data.qfrc_bias = bias

    solver = SimpleNamespace(
        model=model,
        mj_model=object(),
        mj_data=mj_data,
        _mujoco=SimpleNamespace(mj_forward=forward),
        _update_mjc_data=sync,
        use_mujoco_cpu=True,
    )
    provider = NewtonMuJoCoDynamicBiasProvider(SimpleNamespace())
    provider._initialized = True
    provider._newton = SimpleNamespace(JointType=JointType)
    provider._builder = builder
    provider._model = model
    provider._solver = solver
    provider._state = state
    provider._default_q = model.joint_q.numpy()
    provider._mj_qd_start = mj_qd_start
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
        self.assertEqual(2, len(captured))
        self.assertEqual(2, len(forward_calls))
        np.testing.assert_allclose(captured[0][0], [1.0, 10.0, 7.0])
        np.testing.assert_allclose(captured[0][1], [0.1, 0.2, 0.0])

    def test_requested_joint_must_exist_in_recording(self):
        provider, _, _ = _provider()

        with self.assertRaisesRegex(ValueError, "joints not found"):
            provider.compute(_motion(), ("JointC",))

    def test_unrequested_recorded_joint_is_still_injected(self):
        """Bias evaluation uses all recorded mapped state, not only requested slots."""

        provider, captured, _ = _provider()

        values = provider.compute(_motion(), ("JointA",))

        self.assertEqual((2, 1), values.shape)
        np.testing.assert_allclose(captured[0][0], [1.0, 10.0, 7.0])


if __name__ == "__main__":
    unittest.main()
