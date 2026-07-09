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


class _FakeIntArray:
    def __init__(self, values) -> None:
        self._values = np.asarray(values, dtype=np.int32)

    def numpy(self):
        return np.array(self._values, copy=True)


class _FakeFloatArray:
    def __init__(self, values) -> None:
        self._values = np.asarray(values, dtype=np.float64)

    def numpy(self):
        return np.array(self._values, copy=True)


class _FakeMjData:
    def __init__(self, size: int) -> None:
        self.qpos = np.array([1.0, 2.0, 3.0], dtype=np.float64)[:size]
        self.qvel = np.zeros(size, dtype=np.float64)
        self.qfrc_bias = np.zeros(size, dtype=np.float64)


class _FakeModel:
    def __init__(self) -> None:
        self.joint_dof_count = 3
        self.joint_coord_count = 3
        self.device = "cpu"
        # Per-DOF qpos offsets, indexed by Newton DOF index (a, b, c).
        self.mujoco = SimpleNamespace(dof_ref=_FakeFloatArray([0.5, -0.25, 0.0]))


class _FakeModelBuilder:
    def __init__(self) -> None:
        self.joint_label = ["a_joint", "b_joint", "c_joint"]
        self.joint_dof_dim = [(1, 0), (1, 0), (1, 0)]
        self.joint_type = [1, 1, 1]
        self.joint_q_start = [0, 1, 2]
        self.joint_qd_start = [0, 1, 2]

    def add_usd(self, *_args, **_kwargs) -> None:
        return None

    def finalize(self):
        return _FakeModel()


class _FakeSolver:
    """Fake solver whose MuJoCo address maps are permuted relative to Newton's.

    Newton joint order (a, b, c) has q/qd starts [0, 1, 2]; the maps below send
    a->qpos[2]/qvel[1], b->qpos[0]/qvel[2], c->qpos[1]/qvel[0], so a provider
    that indexes MuJoCo arrays with Newton addresses produces wrong values.
    """

    @staticmethod
    def register_custom_attributes(_builder) -> None:
        return None

    def __init__(self, _model, **kwargs) -> None:
        self.kwargs = kwargs
        self.mj_model = SimpleNamespace()
        self.mj_data = _FakeMjData(3)
        self.mj_q_start = _FakeIntArray([2, 0, 1])
        self.mj_qd_start = _FakeIntArray([1, 2, 0])


class _NegativeAddressSolver:
    """Fake solver where joint a has no MuJoCo coordinates (address -1)."""

    @staticmethod
    def register_custom_attributes(_builder) -> None:
        return None

    def __init__(self, _model, **kwargs) -> None:
        self.kwargs = kwargs
        self.mj_model = SimpleNamespace()
        self.mj_data = _FakeMjData(2)
        self.mj_q_start = _FakeIntArray([-1, 0, 1])
        self.mj_qd_start = _FakeIntArray([-1, 0, 1])


class _SolverMissingMaps:
    def __init__(self, _model, **_kwargs) -> None:
        self.mj_model = SimpleNamespace()
        self.mj_data = _FakeMjData(3)


def _fake_mj_forward(_model, data) -> None:
    # Couple every qpos entry into every bias entry so a misdirected qpos
    # write (wrong address, missing dof_ref offset) shifts observed values.
    data.qfrc_bias[:] = data.qpos + 10.0 * data.qvel + data.qpos.sum()


def _fake_runtime(solver_cls=_FakeSolver):
    newton = SimpleNamespace(
        __version__="test-newton",
        ModelBuilder=_FakeModelBuilder,
        JointType=SimpleNamespace(REVOLUTE=1, PRISMATIC=2),
        solvers=SimpleNamespace(SolverMuJoCo=solver_cls),
    )
    mujoco = SimpleNamespace(__version__="test-mujoco", mj_forward=_fake_mj_forward)
    return newton, mujoco


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

    def test_constructor_is_lazy(self):
        bundle = SimpleNamespace(spec=SimpleNamespace(fixed_base=True))
        with patch(
            "anchor_bench.training.dynamic_bias._import_newton_runtime",
            side_effect=AssertionError("runtime import must be lazy"),
        ):
            NewtonMuJoCoDynamicBiasProvider(bundle)

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
            __version__="test-newton",
            ModelBuilder=_FakeModelBuilder,
            JointType=SimpleNamespace(REVOLUTE=1, PRISMATIC=2),
            solvers=SimpleNamespace(SolverMuJoCo=_SolverMissingMaps),
        )
        provider = NewtonMuJoCoDynamicBiasProvider(bundle)
        with (
            patch(
                "anchor_bench.training.dynamic_bias._import_newton_runtime",
                return_value=(newton, object()),
            ),
            patch(
                "anchor_bench.training.dynamic_bias.resolve_runtime_asset",
                return_value=Path("/nonexistent/robot.usd"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "mj_q_start"):
                provider.compute(
                    joint_names=["a"],
                    positions=np.zeros((1, 1)),
                    velocities=np.zeros((1, 1)),
                    requested_joint_names=["a"],
                )

    def test_missing_newton_dependency_has_clear_error(self):
        with patch.dict(sys.modules, {"newton": None, "mujoco": None}):
            with self.assertRaisesRegex(RuntimeError, "requires Newton and MuJoCo"):
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
                result = provider.compute(
                    joint_names=["a", "outside", "b"],
                    positions=np.asarray(
                        [[0.1, 9.0, 0.2], [0.3, 9.0, 0.4], [0.5, 9.0, 0.6]]
                    ),
                    velocities=np.asarray(
                        [[1.0, 9.0, 2.0], [3.0, 9.0, 4.0], [5.0, 9.0, 6.0]]
                    ),
                    requested_joint_names=["b", "a"],
                )

        # Per row: qpos = [pos_b - 0.25, 2.0 (c default), pos_a + 0.5] and
        # qvel = [0, vel_a, vel_b] via the permuted maps; with
        # bias[i] = qpos[i] + 10*qvel[i] + sum(qpos) and S = pos_a + pos_b + 2.25:
        #   b (qvel addr 2) = pos_a + 0.5 + 10*vel_b + S
        #   a (qvel addr 1) = 2.0 + 10*vel_a + S
        np.testing.assert_allclose(
            result.values,
            [[23.15, 14.55], [43.75, 34.95], [64.35, 55.35]],
            rtol=1e-6,
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
        self.assertEqual("mujoco.mj_forward.qfrc_bias", provenance["source"])
        self.assertEqual(
            "newton.solvers.SolverMuJoCo(use_mujoco_cpu=True)",
            provenance["model_source"],
        )
        self.assertEqual("qfrc_bias", provenance["quantity"])
        self.assertEqual("disabled", provenance["contacts"])
        self.assertTrue(provenance["use_mujoco_cpu"])
        self.assertEqual("test-newton", provenance["newton_version"])
        self.assertEqual("test-mujoco", provenance["mujoco_version"])
        self.assertNotIn("batch_size", provenance)
        self.assertEqual("model_defaults", provenance["unrecorded_positions"])
        self.assertEqual("zero", provenance["unrecorded_velocities"])

    def test_requested_joint_without_mujoco_coordinates_errors(self):
        bundle = SimpleNamespace(spec=SimpleNamespace(fixed_base=True))
        with tempfile.TemporaryDirectory() as tmp:
            asset = Path(tmp) / "robot.usd"
            asset.write_bytes(b"usd")
            provider = NewtonMuJoCoDynamicBiasProvider(bundle)
            with (
                patch(
                    "anchor_bench.training.dynamic_bias._import_newton_runtime",
                    return_value=_fake_runtime(_NegativeAddressSolver),
                ),
                patch(
                    "anchor_bench.training.dynamic_bias.resolve_runtime_asset",
                    return_value=asset,
                ),
            ):
                with self.assertRaisesRegex(ValueError, "'a'.*no MuJoCo coordinates"):
                    provider.compute(
                        joint_names=["a"],
                        positions=np.zeros((1, 1)),
                        velocities=np.zeros((1, 1)),
                        requested_joint_names=["a"],
                    )

    def test_optional_joint_without_mujoco_coordinates_is_unmapped(self):
        bundle = SimpleNamespace(spec=SimpleNamespace(fixed_base=True))
        with tempfile.TemporaryDirectory() as tmp:
            asset = Path(tmp) / "robot.usd"
            asset.write_bytes(b"usd")
            provider = NewtonMuJoCoDynamicBiasProvider(bundle)
            with (
                patch(
                    "anchor_bench.training.dynamic_bias._import_newton_runtime",
                    return_value=_fake_runtime(_NegativeAddressSolver),
                ),
                patch(
                    "anchor_bench.training.dynamic_bias.resolve_runtime_asset",
                    return_value=asset,
                ),
            ):
                result = provider.compute(
                    joint_names=["a", "b"],
                    positions=np.asarray([[0.7, 0.9]]),
                    velocities=np.asarray([[0.0, 0.3]]),
                    requested_joint_names=["b"],
                )

        # b maps to qpos[0]/qvel[0] with dof_ref -0.25; c stays at qpos[1]=2.0:
        # qpos = [0.65, 2.0], qvel = [0.3, 0], S = 2.65 ->
        # b = 0.65 + 10*0.3 + 2.65 = 6.3
        np.testing.assert_allclose(result.values, [[6.3]], rtol=1e-6)
        provenance = dict(result.provenance)
        self.assertEqual(["b"], provenance["mapped_joint_names"])
        self.assertEqual(["a"], provenance["unmapped_trajectory_joint_names"])


if __name__ == "__main__":
    unittest.main()
