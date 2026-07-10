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

from anchor_bench.training.dynamic_bias import NewtonMuJoCoDynamicBiasProvider


class _FakeArray:
    def __init__(self, values, *, dtype=None) -> None:
        self._values = np.asarray(values, dtype=dtype).copy()

    def numpy(self) -> np.ndarray:
        return np.array(self._values, copy=True)

    def assign(self, values) -> None:
        incoming = np.asarray(values, dtype=self._values.dtype)
        if incoming.shape != self._values.shape:
            raise AssertionError(f"fake array shape changed from {self._values.shape} to {incoming.shape}")
        self._values[...] = incoming


class _FakeState:
    def __init__(self, q: np.ndarray, qd: np.ndarray) -> None:
        self.joint_q = _FakeArray(q, dtype=np.float32)
        self.joint_qd = _FakeArray(qd, dtype=np.float32)


class _FakeModel:
    def __init__(
        self,
        *,
        worlds: int,
        joint_types: tuple[int, ...],
        effort_limits: tuple[float, ...],
    ) -> None:
        self.world_count = worlds
        self.device = "cpu"
        self.joint_count = worlds * 3
        self.joint_coord_count = worlds * 3
        self.joint_dof_count = worlds * 3
        self.joint_type = _FakeArray(np.tile(joint_types, worlds), dtype=np.int32)
        starts = np.arange(self.joint_count + 1, dtype=np.int32)
        self.joint_q_start = _FakeArray(starts, dtype=np.int32)
        self.joint_qd_start = _FakeArray(starts, dtype=np.int32)
        self.joint_q = _FakeArray(np.tile([1.0, 2.0, 3.0], worlds), dtype=np.float32)
        # A non-zero model default verifies that the provider explicitly zeros qd.
        self.joint_qd = _FakeArray(np.tile([7.0, 8.0, 9.0], worlds), dtype=np.float32)
        self.joint_effort_limit = _FakeArray(np.tile(effort_limits, worlds), dtype=np.float64)
        self.gravity = _FakeArray(np.tile([[0.0, 0.0, -9.7]], (worlds, 1)), dtype=np.float64)

    def state(self) -> _FakeState:
        return _FakeState(self.joint_q.numpy(), self.joint_qd.numpy())


def _fake_runtime(
    *,
    joint_types: tuple[int, ...] = (1, 2, 1),
    effort_limits: tuple[float, ...] = (10.0, 20.0, 0.0),
):
    stats: dict[str, object] = {
        "eval_calls": 0,
        "state_ids": [],
        "out_ids": [],
        "q": [],
        "qd": [],
        "solver_settings": None,
    }

    class FakeBuilder:
        def __init__(self) -> None:
            self.joint_label = ["robot/a_joint", "robot/b_joint", "robot/c_joint"]
            self.joint_count = 3
            self.joint_coord_count = 3
            self.joint_dof_count = 3
            self._template = None
            self._worlds = 1

        def add_usd(self, *_args, **_kwargs) -> None:
            return None

        def replicate(self, template, worlds: int) -> None:
            self._template = template
            self._worlds = worlds

        def finalize(self, **_kwargs) -> _FakeModel:
            return _FakeModel(
                worlds=self._worlds,
                joint_types=joint_types,
                effort_limits=effort_limits,
            )

    class FakeSolver:
        @staticmethod
        def register_custom_attributes(_builder) -> None:
            return None

        def __init__(self, model: _FakeModel, **settings) -> None:
            self.model = model
            stats["solver_settings"] = dict(settings)

        def eval_dynamic_bias(self, state: _FakeState, *, out: _FakeArray) -> _FakeArray:
            stats["eval_calls"] = int(stats["eval_calls"]) + 1
            stats["state_ids"].append(id(state))
            stats["out_ids"].append(id(out))
            q = state.joint_q.numpy().reshape(self.model.world_count, 3)
            qd = state.joint_qd.numpy().reshape(self.model.world_count, 3)
            stats["q"].append(np.array(q, copy=True))
            stats["qd"].append(np.array(qd, copy=True))
            bias = q + 10.0 * qd + q.sum(axis=1, keepdims=True)
            out.assign(bias.reshape(-1))
            return out

    class FakeWarp:
        __version__ = "test-warp"
        float32 = np.float32

        @staticmethod
        def empty(size: int, *, dtype, device) -> _FakeArray:
            del device
            return _FakeArray(np.empty(size, dtype=dtype), dtype=dtype)

    runtime = SimpleNamespace(
        newton=SimpleNamespace(
            __version__="1.4.0.dev0-test",
            ModelBuilder=FakeBuilder,
            JointType=SimpleNamespace(REVOLUTE=1, PRISMATIC=2),
            solvers=SimpleNamespace(SolverMuJoCo=FakeSolver),
        ),
        warp=FakeWarp,
        # Deliberately no mj_forward, mj_data, or qfrc_bias surface.
        mujoco=SimpleNamespace(__version__="3.5.1-test"),
        mujoco_warp=SimpleNamespace(__version__="3.5.1-warp-test"),
    )
    return runtime, stats


def _bundle():
    return SimpleNamespace(spec=SimpleNamespace(fixed_base=True))


class NewtonTrainingAssetTests(unittest.TestCase):
    def _asset(self, contents: bytes = b"usd-snapshot"):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        asset = Path(temporary.name) / "robot.usd"
        asset.write_bytes(contents)
        return asset

    def _provider(self, *, batch_size: int = 2, runtime=None, solver_settings=None):
        asset = self._asset()
        fake_runtime, stats = _fake_runtime() if runtime is None else runtime
        provider = NewtonMuJoCoDynamicBiasProvider(
            _bundle(),
            batch_size=batch_size,
            solver_settings=solver_settings,
        )
        imports = patch("anchor_bench.training.dynamic_bias._import_newton_runtime", return_value=fake_runtime)
        resolution = patch("anchor_bench.training.dynamic_bias.resolve_runtime_asset", return_value=asset)
        imports.start()
        resolution.start()
        self.addCleanup(imports.stop)
        self.addCleanup(resolution.stop)
        return provider, stats, asset

    def test_import_and_constructor_keep_all_simulator_modules_lazy(self):
        script = "\n".join(
            (
                "import sys",
                "for name in ('newton', 'warp', 'mujoco', 'mujoco_warp'):",
                "    assert name not in sys.modules",
                "import anchor_bench.training.asset_snapshot",
                "import anchor_bench.training.dynamic_bias",
                "for name in ('newton', 'warp', 'mujoco', 'mujoco_warp'):",
                "    assert name not in sys.modules",
            )
        )
        subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
        )

        with patch(
            "anchor_bench.training.dynamic_bias._import_newton_runtime",
            side_effect=AssertionError("constructor must not import the runtime"),
        ):
            NewtonMuJoCoDynamicBiasProvider(_bundle())

    def test_missing_configured_input_fails_before_runtime_import(self):
        provider = NewtonMuJoCoDynamicBiasProvider(_bundle())
        with patch(
            "anchor_bench.training.dynamic_bias._import_newton_runtime",
            side_effect=AssertionError("input contract must be checked first"),
        ):
            with self.assertRaisesRegex(ValueError, "missing input joints: b"):
                provider.compute(
                    joint_names=["a"],
                    positions=np.zeros((2, 1)),
                    velocities=np.zeros((2, 1)),
                    requested_joint_names=["b"],
                )

    def test_full_recorded_state_is_injected_and_requested_bias_is_reordered(self):
        provider, stats, _asset = self._provider(batch_size=2)
        result = provider.compute(
            joint_names=["a", "outside", "b"],
            positions=np.asarray(
                [[0.1, 99.0, 0.2], [0.3, 99.0, 0.4], [0.5, 99.0, 0.6]],
                dtype=np.float32,
            ),
            velocities=np.asarray(
                [[1.0, 99.0, 2.0], [3.0, 99.0, 4.0], [5.0, 99.0, 6.0]],
                dtype=np.float32,
            ),
            requested_joint_names=["b", "a"],
        )

        np.testing.assert_allclose(
            result.values,
            [[23.5, 13.4], [44.1, 34.0], [64.7, 54.6]],
            rtol=1.0e-6,
        )
        self.assertEqual(2, stats["eval_calls"])
        self.assertEqual(1, len(set(stats["state_ids"])))
        self.assertEqual(1, len(set(stats["out_ids"])))
        np.testing.assert_allclose(
            stats["q"][0],
            [[0.1, 0.2, 3.0], [0.3, 0.4, 3.0]],
        )
        np.testing.assert_allclose(
            stats["qd"][0],
            [[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]],
        )
        np.testing.assert_allclose(stats["q"][1][1], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(stats["qd"][1][1], [0.0, 0.0, 0.0])

        provenance = dict(result.provenance)
        self.assertEqual("newton.solvers.SolverMuJoCo.eval_dynamic_bias", provenance["source"])
        self.assertEqual(["a", "b"], provenance["mapped_joint_names"])
        self.assertEqual(["a_joint", "b_joint"], provenance["mapped_asset_joint_names"])
        self.assertEqual(["outside"], provenance["unmapped_trajectory_joint_names"])
        self.assertEqual(["c_joint"], provenance["defaulted_joint_names"])
        self.assertEqual("model_defaults", provenance["unrecorded_positions"])
        self.assertEqual("zero", provenance["unrecorded_velocities"])

    def test_cache_key_reuse_is_guarded_by_trajectory_content(self):
        provider, stats, _asset = self._provider(batch_size=2)
        positions = np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        velocities = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        common = {
            "joint_names": ["a", "b"],
            "positions": positions,
            "velocities": velocities,
            "cache_key": "same-source-path",
        }

        first = provider.compute(requested_joint_names=["a"], **common)
        second = provider.compute(requested_joint_names=["b"], **common)
        self.assertEqual(1, stats["eval_calls"])
        np.testing.assert_allclose(first.values[:, 0], [13.4, 34.0])
        np.testing.assert_allclose(second.values[:, 0], [23.5, 44.1])

        changed = positions.copy()
        changed[0, 0] = 0.9
        third = provider.compute(
            joint_names=["a", "b"],
            positions=changed,
            velocities=velocities,
            requested_joint_names=["a"],
            cache_key="same-source-path",
        )
        self.assertEqual(2, stats["eval_calls"])
        self.assertNotEqual(float(first.values[0, 0]), float(third.values[0, 0]))

    def test_snapshot_exposes_effort_limits_and_complete_provenance(self):
        provider, stats, asset = self._provider(
            solver_settings={"integrator": "implicitfast", "iterations": 12},
        )
        snapshot = provider.asset_snapshot

        self.assertEqual((20.0, 10.0), snapshot.effort_limits(["b", "a"]))
        with self.assertRaisesRegex(ValueError, "c.*finite and positive"):
            snapshot.effort_limits(["c"])
        self.assertEqual(hashlib.sha256(asset.read_bytes()).hexdigest(), snapshot.asset_sha256)
        self.assertEqual((0.0, 0.0, -9.7), snapshot.gravity)
        self.assertEqual("1.4.0.dev0-test", snapshot.versions["newton"])
        self.assertEqual("3.5.1-test", snapshot.versions["mujoco"])
        self.assertEqual("3.5.1-warp-test", snapshot.versions["mujoco_warp"])
        self.assertEqual("implicitfast", snapshot.solver_settings["integrator"])
        self.assertEqual(12, snapshot.solver_settings["iterations"])
        self.assertFalse(snapshot.solver_settings["use_mujoco_cpu"])
        self.assertTrue(snapshot.solver_settings["disable_contacts"])
        self.assertEqual(dict(snapshot.solver_settings), stats["solver_settings"])

        provenance = dict(snapshot.provenance)
        self.assertEqual(snapshot.asset_sha256, provenance["asset_sha256"])
        self.assertEqual([0.0, 0.0, -9.7], provenance["gravity"])
        self.assertIn("scalar_joint_mapping", provenance)
        self.assertIn("solver_settings", provenance)

    def test_non_scalar_recorded_joint_is_optional_but_cannot_be_an_input(self):
        runtime = _fake_runtime(joint_types=(1, 3, 1))
        provider, _stats, _asset = self._provider(runtime=runtime)
        result = provider.compute(
            joint_names=["a", "b"],
            positions=np.asarray([[0.1, 0.2]], dtype=np.float32),
            velocities=np.asarray([[0.0, 0.0]], dtype=np.float32),
            requested_joint_names=["a"],
        )
        self.assertEqual(["b"], result.provenance["unmapped_trajectory_joint_names"])

        with self.assertRaisesRegex(ValueError, "b.*scalar revolute or prismatic"):
            provider.compute(
                joint_names=["a", "b"],
                positions=np.asarray([[0.1, 0.2]], dtype=np.float32),
                velocities=np.asarray([[0.0, 0.0]], dtype=np.float32),
                requested_joint_names=["b"],
            )

    def test_aliases_cannot_map_two_recorded_columns_to_one_dof(self):
        provider, _stats, _asset = self._provider()
        with self.assertRaisesRegex(ValueError, "a.*a_joint.*same Newton scalar joint/DOF"):
            provider.compute(
                joint_names=["a", "a_joint"],
                positions=np.zeros((1, 2), dtype=np.float32),
                velocities=np.zeros((1, 2), dtype=np.float32),
                requested_joint_names=["a"],
            )

    def test_same_asset_path_is_rehashed_after_content_changes(self):
        runtime, _stats = _fake_runtime()
        asset = self._asset(b"first")
        imports = patch("anchor_bench.training.dynamic_bias._import_newton_runtime", return_value=runtime)
        resolution = patch("anchor_bench.training.dynamic_bias.resolve_runtime_asset", return_value=asset)
        with imports, resolution:
            first = NewtonMuJoCoDynamicBiasProvider(_bundle(), batch_size=1).asset_snapshot
            asset.write_bytes(b"second")
            second = NewtonMuJoCoDynamicBiasProvider(_bundle(), batch_size=1).asset_snapshot

        self.assertNotEqual(first.asset_sha256, second.asset_sha256)
        self.assertEqual(hashlib.sha256(b"second").hexdigest(), second.asset_sha256)


if __name__ == "__main__":
    unittest.main()
