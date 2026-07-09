from __future__ import annotations

import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from anchor_bench.training.dynamic_bias import (
    NewtonMuJoCoDynamicBiasProvider,
    _import_newton_runtime,
)


class _FakeArray:
    def __init__(self, values) -> None:
        self.values = np.asarray(values, dtype=np.float32)

    def numpy(self):
        return np.array(self.values, copy=True)

    def assign(self, values) -> None:
        self.values = np.asarray(values, dtype=np.float32).copy()


class _FakeState:
    def __init__(self, q, qd) -> None:
        self.joint_q = _FakeArray(q)
        self.joint_qd = _FakeArray(qd)


class _FakeModel:
    def __init__(self, batch_size: int) -> None:
        self.joint_dof_count = batch_size * 3
        self.device = "cpu"
        self.joint_q = _FakeArray(np.tile([1.0, 2.0, 3.0], batch_size))
        self._batch_size = batch_size

    def state(self):
        return _FakeState(
            self.joint_q.numpy(),
            np.zeros(self.joint_dof_count, dtype=np.float32),
        )


class _FakeModelBuilder:
    def __init__(self) -> None:
        self.joint_label = ["a_joint", "b_joint", "c_joint"]
        self.joint_dof_dim = [(1, 0), (1, 0), (1, 0)]
        self.joint_type = [1, 1, 1]
        self.joint_q_start = [0, 1, 2]
        self.joint_qd_start = [0, 1, 2]
        self.joint_coord_count = 3
        self.joint_dof_count = 3
        self._batch_size = 1

    def add_usd(self, *_args, **_kwargs) -> None:
        return None

    def replicate(self, _template, batch_size: int) -> None:
        self._batch_size = batch_size

    def finalize(self):
        return _FakeModel(self._batch_size)


class _FakeSolver:
    @staticmethod
    def register_custom_attributes(_builder) -> None:
        return None

    def __init__(self, _model, **_kwargs) -> None:
        return None

    def eval_dynamic_bias(self, state, out) -> None:
        out.assign(state.joint_q.values + 10.0 * state.joint_qd.values)


def _fake_runtime():
    newton = SimpleNamespace(
        __version__="test-newton",
        ModelBuilder=_FakeModelBuilder,
        JointType=SimpleNamespace(REVOLUTE=1, PRISMATIC=2),
        solvers=SimpleNamespace(SolverMuJoCo=_FakeSolver),
    )
    wp = SimpleNamespace(
        float32=np.float32,
        empty=lambda size, **_kwargs: _FakeArray(np.zeros(size, dtype=np.float32)),
    )
    return newton, wp


class DynamicBiasProviderTests(unittest.TestCase):
    def test_training_data_import_keeps_newton_and_warp_lazy(self):
        script = "\n".join(
            (
                "import sys",
                "assert 'newton' not in sys.modules",
                "assert 'warp' not in sys.modules",
                "import anchor_bench.training.data",
                "assert 'newton' not in sys.modules",
                "assert 'warp' not in sys.modules",
            )
        )

        subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
        )

    def test_constructor_is_lazy_and_validates_batch_size(self):
        bundle = SimpleNamespace(spec=SimpleNamespace(fixed_base=True))
        with patch(
            "anchor_bench.training.dynamic_bias._import_newton_runtime",
            side_effect=AssertionError("runtime import must be lazy"),
        ):
            NewtonMuJoCoDynamicBiasProvider(bundle)

        with self.assertRaisesRegex(ValueError, "batch_size must be >= 1"):
            NewtonMuJoCoDynamicBiasProvider(bundle, batch_size=0)

    def test_missing_input_joint_fails_before_runtime_import(self):
        bundle = SimpleNamespace(spec=SimpleNamespace(fixed_base=True))
        provider = NewtonMuJoCoDynamicBiasProvider(bundle)
        with patch(
            "anchor_bench.training.dynamic_bias._import_newton_runtime",
            side_effect=AssertionError("must validate trajectory first"),
        ):
            with self.assertRaisesRegex(ValueError, "missing input joints: b"):
                provider.compute(
                    joint_names=["a"],
                    positions=np.zeros((2, 1)),
                    velocities=np.zeros((2, 1)),
                    requested_joint_names=["b"],
                )

    def test_missing_newton_capability_has_clear_error(self):
        bundle = SimpleNamespace(spec=SimpleNamespace(fixed_base=True))
        newton = SimpleNamespace(
            solvers=SimpleNamespace(SolverMuJoCo=object),
        )
        provider = NewtonMuJoCoDynamicBiasProvider(bundle)
        with patch(
            "anchor_bench.training.dynamic_bias._import_newton_runtime",
            return_value=(newton, object()),
        ):
            with self.assertRaisesRegex(RuntimeError, "SolverMuJoCo.eval_dynamic_bias"):
                provider.compute(
                    joint_names=["a"],
                    positions=np.zeros((1, 1)),
                    velocities=np.zeros((1, 1)),
                    requested_joint_names=["a"],
                )

    def test_missing_newton_dependency_has_clear_error(self):
        with patch.dict(sys.modules, {"newton": None}):
            with self.assertRaisesRegex(RuntimeError, "requires Newton"):
                _import_newton_runtime()

    def test_rejects_multiple_trajectory_labels_for_same_newton_joint(self):
        bundle = SimpleNamespace(spec=SimpleNamespace(fixed_base=True))
        with tempfile.TemporaryDirectory() as tmp:
            asset = Path(tmp) / "robot.usd"
            asset.write_bytes(b"usd")
            provider = NewtonMuJoCoDynamicBiasProvider(bundle)
            with (
                patch(
                    "anchor_bench.training.dynamic_bias._import_newton_runtime",
                    return_value=_fake_runtime(),
                ),
                patch(
                    "anchor_bench.training.dynamic_bias.resolve_runtime_asset",
                    return_value=asset,
                ),
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "a.*a_joint.*same Newton scalar joint/DOF",
                ):
                    provider.compute(
                        joint_names=["a", "a_joint"],
                        positions=np.zeros((1, 2)),
                        velocities=np.zeros((1, 2)),
                        requested_joint_names=["a"],
                    )

    def test_injects_all_mapped_trajectory_joints_and_defaults_others(self):
        bundle = SimpleNamespace(spec=SimpleNamespace(fixed_base=True))
        with tempfile.TemporaryDirectory() as tmp:
            asset = Path(tmp) / "robot.usd"
            asset.write_bytes(b"usd")
            provider = NewtonMuJoCoDynamicBiasProvider(bundle, batch_size=2)
            with (
                patch(
                    "anchor_bench.training.dynamic_bias._import_newton_runtime",
                    return_value=_fake_runtime(),
                ),
                patch(
                    "anchor_bench.training.dynamic_bias.resolve_runtime_asset",
                    return_value=asset,
                ),
            ):
                result = provider.compute(
                    joint_names=["a", "outside", "b"],
                    positions=np.asarray([[4.0, 40.0, 5.0], [6.0, 60.0, 7.0], [8.0, 80.0, 9.0]]),
                    velocities=np.asarray([[0.1, 1.0, 0.2], [0.3, 3.0, 0.4], [0.5, 5.0, 0.6]]),
                    requested_joint_names=["b", "a"],
                )

        np.testing.assert_allclose(
            result.values,
            [[7.0, 5.0], [11.0, 9.0], [15.0, 13.0]],
        )
        provenance = dict(result.provenance)
        self.assertEqual(["b", "a"], provenance["requested_joint_names"])
        self.assertEqual(["a", "b"], provenance["mapped_joint_names"])
        self.assertEqual(["outside"], provenance["unmapped_trajectory_joint_names"])
        self.assertEqual(["c"], provenance["defaulted_joint_names"])
        self.assertEqual(
            hashlib.sha256(b"usd").hexdigest(),
            provenance["asset_sha256"],
        )
        self.assertEqual(
            "newton.solvers.SolverMuJoCo.eval_dynamic_bias",
            provenance["source"],
        )
        self.assertEqual("qfrc_bias", provenance["quantity"])
        self.assertEqual(2, provenance["batch_size"])
        self.assertEqual("disabled", provenance["contacts"])
        self.assertIs(False, provenance["use_mujoco_cpu"])
        self.assertEqual("test-newton", provenance["newton_version"])
        self.assertEqual("model_defaults", provenance["unrecorded_positions"])
        self.assertEqual("zero", provenance["unrecorded_velocities"])


if __name__ == "__main__":
    unittest.main()
