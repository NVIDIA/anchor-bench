from __future__ import annotations

import io
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import yaml


REPO = Path(__file__).resolve().parents[1]


def run_cli(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "anchor_bench.cli", *args],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=check,
    )


def write_sage_metadata(folder: Path, time_unit: str = "seconds") -> None:
    (folder / "metadata.yaml").write_text(
        "schema: sage\n"
        f"time_unit: {time_unit}\n"
        "position_unit: rad\n"
        "velocity_unit: rad/s\n"
        "torque_unit: N*m\n",
        encoding="utf-8",
    )


class RuntimeWorkflowTests(unittest.TestCase):
    def test_to_torch_accepts_isaac_lab_proxy_array(self):
        from anchor_bench.runtime.robot_scene import to_torch

        class FakeTensor:
            pass

        class FakeWarp:
            class array:
                pass

            @staticmethod
            def to_torch(value):
                raise AssertionError("ProxyArray should use its explicit .torch accessor")

        tensor = FakeTensor()
        proxy = SimpleNamespace(torch=tensor)
        fake_torch = ModuleType("torch")
        fake_torch.Tensor = FakeTensor

        with patch.dict(sys.modules, {"torch": fake_torch}):
            self.assertIs(to_torch(proxy, FakeWarp), tensor)

    def test_actuator_file_converts_to_isaac_implicit_params(self):
        from anchor_bench.runtime.robot_config import build_implicit_actuator_params
        from anchor_bench.robots.registry import get_robot

        params = build_implicit_actuator_params(get_robot("so101"))

        self.assertEqual(params["effort_limit_sim"], 3.35)
        self.assertEqual(params["velocity_limit_sim"], 30.0)
        self.assertEqual(params["stiffness"]["Rotation"], 48.419899)
        self.assertEqual(params["damping"]["Jaw"], 2.81873)
        self.assertEqual(params["armature"]["Elbow"], 0.03772)
        self.assertEqual(params["dry_friction"]["Wrist_Roll"], 0.221761)
        old_key = "dynamic" + "_friction"
        self.assertNotIn(old_key, params)
        self.assertEqual(params["viscous_friction"]["Pitch"], 0.561425)
        self.assertEqual(params["motor_lag_ms"], 0.0)

    def test_fitted_actuator_document_preserves_static_limits(self):
        from anchor_bench.runtime.robot_config import (
            build_fitted_actuator_document,
            load_implicit_actuator_params,
        )
        from anchor_bench.robots.registry import get_robot

        bundle = get_robot("so101")
        with tempfile.TemporaryDirectory() as tmp:
            params_path = Path(tmp) / "best_params.yaml"
            document = build_fitted_actuator_document(
                bundle,
                {
                    "stiffness": {"Rotation": 10.0},
                    "damping": {"Jaw": 1.5},
                },
                metadata={"source": "sysid", "best_mse": 0.01},
            )
            params_path.write_text(
                yaml.safe_dump(document, sort_keys=False),
                encoding="utf-8",
            )

            params = load_implicit_actuator_params(bundle, params_path)

        self.assertEqual(params["stiffness"]["Rotation"], 10.0)
        self.assertEqual(params["stiffness"]["Pitch"], 47.782635)
        self.assertEqual(params["damping"]["Jaw"], 1.5)
        self.assertEqual(params["effort_limit_sim"], 3.35)
        self.assertEqual(params["velocity_limit_sim"], 30.0)
        self.assertNotIn("best_mse", params)
        self.assertNotIn("joints", params)

    def test_bounds_loader_returns_typed_bounds_file(self):
        from anchor_bench.runtime.robot_config import load_sysid_bounds
        from anchor_bench.robots.registry import get_robot

        bounds = load_sysid_bounds(get_robot("so101"))

        self.assertEqual(bounds.schema_version, 2)
        self.assertFalse(bounds.mirror)
        self.assertEqual(bounds.joint_types, ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"])
        self.assertIn("armature", bounds.parameters)

    def test_missing_full_runtime_error_is_actionable(self):
        from anchor_bench.runtime.environment import RuntimeUnavailableError, require_full_runtime

        with self.assertRaises(RuntimeUnavailableError) as ctx:
            require_full_runtime(("definitely_missing_isaac_runtime_module",), workflow="sysid")

        message = str(ctx.exception)
        self.assertIn("sysid requires the full Anchor Bench Docker runtime", message)
        self.assertIn("bash setup.sh --full", message)
        self.assertIn("anchor-bench:full", message)

    def test_asset_resolution_error_names_supported_locations(self):
        from anchor_bench.runtime.assets import AssetNotFoundError, resolve_robot_asset
        from anchor_bench.robots.registry import get_robot

        with patch.dict("os.environ", {}, clear=False) as env:
            env.pop("ANCHOR_BENCH_ASSET_DIR", None)
            with self.assertRaises(AssetNotFoundError) as ctx:
                resolve_robot_asset(get_robot("so101"), "missing-so101.usd")

        message = str(ctx.exception)
        self.assertEqual(
            message,
            "Missing asset 'missing-so101.usd' for robot so101.\n"
            "Looked in:\n"
            "- $ANCHOR_BENCH_ASSET_DIR\n"
            "- assets/robots/so101/files\n"
            "\n"
            "Set ANCHOR_BENCH_ASSET_DIR or place assets under assets/robots/so101/files/.",
        )

    def test_sysid_command_preflights_so101_asset(self):
        from anchor_bench.core.schemas import SysIdRunConfig
        from anchor_bench.runtime.assets import AssetNotFoundError
        from anchor_bench.sysid.runner import resolve_sysid_run_config, run_resolved_sysid

        with tempfile.TemporaryDirectory() as tmp:
            with patch("anchor_bench.runtime.environment.require_full_runtime"):
                with patch("anchor_bench.runtime.assets._candidate_roots", return_value=[Path(tmp)]):
                    with self.assertRaises(AssetNotFoundError) as ctx:
                        resolved = resolve_sysid_run_config(
                            SysIdRunConfig(
                                robot="so101",
                                data="inputs/datasets/so101/custom_figure8_with_camera",
                                output="outputs/sysid/so101/smoke",
                                dry_run=False,
                            )
                        )
                        run_resolved_sysid(resolved)

        self.assertIn("Missing asset 'so101.usd' for robot so101.", str(ctx.exception))

    def test_benchmark_command_preflights_so101_asset(self):
        from anchor_bench.benchmark.newton_runner import BenchmarkMotion, run_benchmark_batch
        from anchor_bench.runtime.assets import AssetNotFoundError

        with tempfile.TemporaryDirectory() as tmp:
            with patch("anchor_bench.runtime.environment.require_full_runtime"):
                with patch("anchor_bench.runtime.assets._candidate_roots", return_value=[Path(tmp)]):
                    with self.assertRaises(AssetNotFoundError) as ctx:
                        run_benchmark_batch(
                            "so101",
                            [BenchmarkMotion("inputs/datasets/so101/custom_figure8_with_camera")],
                            dry_run=False,
                        )

        self.assertIn("Missing asset 'so101.usd' for robot so101.", str(ctx.exception))

    def test_cli_accepts_runtime_options_in_dry_run(self):
        result = run_cli(
            "sysid",
            "--robot",
            "so101",
            "--data",
            "inputs/datasets/so101/custom_figure8_with_camera",
            "--output",
            "outputs/sysid/so101/smoke",
            "--headless",
            "--num-envs",
            "2",
            "--max-iter",
            "1",
            "--max-trajectory-len",
            "200",
            "--seed",
            "7",
            "--dry-run",
        )
        self.assertIn("--headless", result.stdout)
        self.assertIn("--num-envs", result.stdout)
        self.assertIn("'2'", result.stdout)
        self.assertIn("--seed", result.stdout)

        result = run_cli(
            "benchmark",
            "--robot",
            "so101",
            "--motion",
            "inputs/datasets/so101/custom_figure8_with_camera",
            "--output",
            "outputs/sim2real_benchmark",
            "--headless",
            "--dataset",
            "example",
            "--motion-name",
            "custom_figure8_with_camera",
            "--actuator-model",
            "implicit",
            "--seed",
            "42",
            "--actuator-params",
            "outputs/sysid/so101/best_params.yaml",
            "--dry-run",
        )
        self.assertIn("--dataset", result.stdout)
        self.assertIn("custom_figure8_with_camera", result.stdout)
        self.assertIn("--motion-name", result.stdout)
        self.assertIn("custom_figure8_with_camera", result.stdout)
        self.assertIn("--actuator-model", result.stdout)
        self.assertIn("implicit", result.stdout)
        self.assertIn("--seed", result.stdout)
        self.assertIn("--actuator-params", result.stdout)

    def test_cli_rejects_non_positive_sysid_runtime_options(self):
        result = run_cli(
            "sysid",
            "--robot",
            "so101",
            "--data",
            "inputs/datasets/so101/custom_figure8_with_camera",
            "--output",
            "outputs/sysid/so101/smoke",
            "--num-envs",
            "0",
            "--dry-run",
            check=False,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("num_envs must be >= 1", result.stderr)

    def test_sysid_runtime_rejects_non_positive_numeric_inputs(self):
        from anchor_bench.sysid.runtime import _resolve_optional_positive_int, _resolve_positive_int

        self.assertEqual(64, _resolve_positive_int("num_envs", None, 64))
        self.assertEqual(2, _resolve_positive_int("num_envs", 2, 64))
        with self.assertRaisesRegex(ValueError, "num_envs must be >= 1"):
            _resolve_positive_int("num_envs", 0, 64)
        with self.assertRaisesRegex(ValueError, "max_iter must be >= 1"):
            _resolve_positive_int("max_iter", -1, 200)
        self.assertIsNone(_resolve_optional_positive_int("max_trajectory_len", None))
        with self.assertRaisesRegex(ValueError, "max_trajectory_len must be >= 1"):
            _resolve_optional_positive_int("max_trajectory_len", 0)

    def test_simulation_timing_uses_render_freq(self):
        from anchor_bench.core.run_config import resolve_simulation_timing
        from anchor_bench.core.schemas import SimulationDefaults

        timing = resolve_simulation_timing(
            SimulationDefaults(
                physics_freq=500,
                control_freq=100,
                render_freq=50,
            ),
            "run.yaml.simulation",
        )

        self.assertEqual(timing.control_divisor, 5)
        self.assertEqual(timing.render_interval, 10)
        self.assertAlmostEqual(timing.physics_dt, 0.002)
        self.assertAlmostEqual(timing.control_dt, 0.01)

        with self.assertRaisesRegex(ValueError, "physics_freq divisible by render_freq"):
            resolve_simulation_timing(
                SimulationDefaults(
                    physics_freq=500,
                    control_freq=100,
                    render_freq=333,
                ),
                "run.yaml.simulation",
            )

    def test_sysid_joint_selection_uses_run_yaml_value(self):
        from anchor_bench.sysid.runtime import _resolve_sysid_joint_names

        bundle = SimpleNamespace(
            spec=SimpleNamespace(
                joint_names=["Rotation", "Pitch", "Elbow"],
                joint_aliases={"rot": "Rotation"},
            )
        )

        self.assertEqual(["Rotation", "Pitch", "Elbow"], _resolve_sysid_joint_names(bundle, "full"))
        self.assertEqual(["Rotation", "Elbow"], _resolve_sysid_joint_names(bundle, ["rot", "Elbow"]))
        with self.assertRaisesRegex(ValueError, "unknown sysid joint"):
            _resolve_sysid_joint_names(bundle, ["missing"])
        with self.assertRaisesRegex(ValueError, "must select at least one joint"):
            _resolve_sysid_joint_names(bundle, [])

    def test_sysid_runtime_rejects_invalid_numeric_inputs_before_isaac_launch(self):
        from anchor_bench.sysid.runtime import run_robot_sysid

        def fail_launch(*_args, **_kwargs):
            raise AssertionError("invalid sysid inputs should be validated before Isaac Sim launch")

        with tempfile.TemporaryDirectory() as tmp:
            with patch("anchor_bench.runtime.app_launcher.launch_isaac_sim", fail_launch):
                with self.assertRaisesRegex(ValueError, "num_envs must be >= 1"):
                    run_robot_sysid(
                        "so101",
                        data="inputs/datasets/so101/custom_figure8_with_camera",
                        output=tmp,
                        num_envs=0,
                        max_iter=1,
                    )

    def test_sysid_summary_uses_type_not_legacy_model_type(self):
        import yaml

        from anchor_bench.robots.registry import get_robot
        from anchor_bench.runtime.robot_config import load_sysid_bounds
        from anchor_bench.sysid.runtime import _write_summary

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            _write_summary(
                output,
                run_cfg=get_robot("so101").run_config,
                bounds=load_sysid_bounds(get_robot("so101")),
                actuator_params={"stiffness": {"Rotation": 1.0}},
                args={"num_envs": 1, "config_sources": {"num_envs": "cli"}},
                robot_name="so101",
            )
            summary = yaml.safe_load((output / "run_summary.yaml").read_text(encoding="utf-8"))

        self.assertEqual("implicit", summary["actuator_model"]["type"])
        self.assertNotIn("model_type", summary["actuator_model"])
        self.assertEqual("cli", summary["sysid"]["config_sources"]["num_envs"])

    def test_sysid_resolved_config_tracks_cli_override_sources(self):
        from anchor_bench.core.schemas import SysIdRunConfig
        from anchor_bench.sysid.runner import resolve_sysid_run_config

        resolved = resolve_sysid_run_config(
            SysIdRunConfig(
                robot="so101",
                data="inputs/datasets/so101/custom_figure8_with_camera",
                output="outputs/sysid/so101/override",
                num_envs=2,
                max_iter=3,
                max_trajectory_len=10,
            )
        )

        self.assertEqual("cli", resolved.config_sources["real_data_dir"])
        self.assertEqual("cli", resolved.config_sources["output_dir"])
        self.assertEqual("cli", resolved.config_sources["num_envs"])
        self.assertEqual("cli", resolved.config_sources["max_iter"])
        self.assertEqual("cli", resolved.config_sources["max_trajectory_len"])
        self.assertEqual(3, resolved.max_iter)

    def test_sysid_cli_surfaces_runtime_unavailable_on_non_dry_run(self):
        from anchor_bench.cli.main import main
        from anchor_bench.runtime.environment import RuntimeUnavailableError

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "custom_figure8_with_camera"
            data_dir.mkdir()
            stderr = io.StringIO()
            with patch(
                "anchor_bench.runtime.environment.require_full_runtime",
                side_effect=RuntimeUnavailableError(
                    "sysid requires the full Anchor Bench Docker runtime, but these modules are missing: "
                    "isaaclab, isaaclab_newton, newton. Build it with `bash setup.sh --full`, then run inside "
                    "`anchor-bench:full` with GPU/runtime mounts."
                ),
            ):
                with redirect_stderr(stderr):
                    rc = main(
                        [
                            "sysid",
                            "--robot",
                            "so101",
                            "--data",
                            str(data_dir),
                            "--output",
                            "outputs/sysid/so101/smoke",
                        ]
                    )

        self.assertEqual(rc, 2)
        self.assertIn("anchor-bench:full", stderr.getvalue())

    def test_sysid_cli_surfaces_missing_asset_on_non_dry_run(self):
        from anchor_bench.cli.main import main

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "custom_figure8_with_camera"
            data_dir.mkdir()
            empty_assets = Path(tmp) / "assets"
            empty_assets.mkdir()
            stderr = io.StringIO()
            with patch("anchor_bench.runtime.environment.require_full_runtime"):
                with patch("anchor_bench.runtime.assets._candidate_roots", return_value=[empty_assets]):
                    with redirect_stderr(stderr):
                        rc = main(
                            [
                                "sysid",
                                "--robot",
                                "so101",
                                "--data",
                                str(data_dir),
                                "--output",
                                "outputs/sysid/so101/smoke",
                            ]
                        )

        self.assertEqual(rc, 2)
        self.assertIn("Missing asset 'so101.usd'", stderr.getvalue())

    def test_runtime_helpers_use_configured_joint_names(self):
        from anchor_bench.runtime.robot_scene import _matrix_from_param_map, robot_joint_names

        bundle = SimpleNamespace(spec=SimpleNamespace(joint_names=["JointA", "JointB"]))

        class FakeTorch:
            float32 = "float32"

            @staticmethod
            def tensor(values, dtype=None, device=None):
                return {"values": values, "dtype": dtype, "device": device}

        params = {"stiffness": {"JointA": 1.0, "JointB": 2.0}}

        values = _matrix_from_param_map(params, "stiffness", robot_joint_names(bundle), 2, FakeTorch, "cpu")

        self.assertEqual(robot_joint_names(bundle), ["JointA", "JointB"])
        self.assertEqual(values["values"], [[1.0, 2.0], [1.0, 2.0]])

    def test_runtime_helper_rejects_missing_configured_joint_params(self):
        from anchor_bench.runtime.robot_scene import _matrix_from_param_map

        class FakeTorch:
            float32 = "float32"

            @staticmethod
            def tensor(values, dtype=None, device=None):
                return values

        with self.assertRaisesRegex(ValueError, "missing stiffness values"):
            _matrix_from_param_map({"stiffness": {"JointA": 1.0}}, "stiffness", ["JointA", "JointB"], 1, FakeTorch, "cpu")

    def test_matching_fixed_base_config_does_not_override_usd_root_mode(self):
        from anchor_bench.runtime.robot_scene import resolve_fix_root_override

        self.assertIsNone(resolve_fix_root_override(True, True))
        self.assertIsNone(resolve_fix_root_override(False, False))
        self.assertFalse(resolve_fix_root_override(False, True))
        self.assertTrue(resolve_fix_root_override(True, False))

    def test_scene_builder_passes_fix_root_link_to_usd_spawn(self):
        from anchor_bench.runtime.robot_scene import build_scene_class

        class FakeUsdFileCfg:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.articulation_props = kwargs.get("articulation_props")

        class FakeArticulationRootPropertiesCfg:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.fix_root_link = kwargs.get("fix_root_link")

        class FakeArticulationCfg:
            class InitialStateCfg:
                def __init__(self, **kwargs):
                    self.kwargs = kwargs

            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.spawn = kwargs["spawn"]

        class FakeAssetBaseCfg:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        fake_sim = SimpleNamespace(
            GroundPlaneCfg=lambda **kwargs: SimpleNamespace(kwargs=kwargs),
            UsdFileCfg=FakeUsdFileCfg,
            ArticulationRootPropertiesCfg=FakeArticulationRootPropertiesCfg,
            DomeLightCfg=lambda **kwargs: SimpleNamespace(kwargs=kwargs),
        )
        fake_isaaclab = ModuleType("isaaclab")
        fake_isaaclab.sim = fake_sim
        fake_modules = {
            "isaaclab": fake_isaaclab,
            "isaaclab.sim": fake_sim,
            "isaaclab.actuators": SimpleNamespace(ImplicitActuatorCfg=lambda **kwargs: SimpleNamespace(kwargs=kwargs)),
            "isaaclab.assets": SimpleNamespace(ArticulationCfg=FakeArticulationCfg, AssetBaseCfg=FakeAssetBaseCfg),
            "isaaclab.scene": SimpleNamespace(InteractiveSceneCfg=object),
            "isaaclab.utils": SimpleNamespace(configclass=lambda cls: cls),
        }

        with patch.dict(sys.modules, fake_modules):
            SceneCfg = build_scene_class(
                Path("robot.usd"),
                {"stiffness": {"JointA": 1.0}, "damping": {"JointA": 1.0}},
                ["JointA"],
                fix_root=True,
            )

        self.assertTrue(SceneCfg.robot.spawn.articulation_props.fix_root_link)

    def test_write_viscous_friction_uses_newton_manager_without_root_view_search(self):
        import numpy as np

        from anchor_bench.runtime.robot_scene import _write_viscous_friction

        class FakeTensor:
            def __init__(self, values):
                self._values = np.asarray(values, dtype=float)

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self._values

        class FakeDamping:
            def __init__(self):
                self.values = np.zeros((2, 3), dtype=float)

            def numpy(self):
                return self.values.copy()

            def assign(self, values):
                self.values = np.asarray(values, dtype=float)

        class FakeRobot:
            joint_names = ["JointA", "JointB", "JointC"]

            @property
            def root_view(self):
                raise AssertionError("root_view should not be searched")

        changes = []

        class FakeNewtonManager:

            @classmethod
            def get_model(cls):
                return SimpleNamespace(mujoco=SimpleNamespace(dof_passive_damping=fake_damping))

            @classmethod
            def add_model_change(cls, change):
                changes.append(change)

        fake_damping = FakeDamping()
        fake_flags = SimpleNamespace(JOINT_DOF_PROPERTIES="joint_dof_properties")

        with patch.dict(
            sys.modules,
            {
                "isaaclab_newton": SimpleNamespace(),
                "isaaclab_newton.physics": SimpleNamespace(NewtonManager=FakeNewtonManager),
                "newton": SimpleNamespace(),
                "newton.solvers": SimpleNamespace(SolverNotifyFlags=fake_flags),
            },
        ):
            _write_viscous_friction(FakeRobot(), FakeTensor([[0.2, 0.4], [0.6, 0.8]]), [0, 2])

        np.testing.assert_allclose(fake_damping.values, [[0.2, 0.0, 0.4], [0.6, 0.0, 0.8]])
        self.assertEqual(changes, ["joint_dof_properties"])

    def test_write_viscous_friction_fails_fast_without_passive_damping(self):
        from anchor_bench.runtime.robot_scene import _write_viscous_friction

        class FakeNewtonManager:
            @classmethod
            def get_model(cls):
                return SimpleNamespace(mujoco=SimpleNamespace())

        with patch.dict(
            sys.modules,
            {
                "isaaclab_newton": SimpleNamespace(),
                "isaaclab_newton.physics": SimpleNamespace(NewtonManager=FakeNewtonManager),
            },
        ):
            with self.assertRaisesRegex(RuntimeError, "mujoco\\.dof_passive_damping"):
                _write_viscous_friction(SimpleNamespace(joint_names=["JointA"]), SimpleNamespace(), [0])


    def test_isaac_actuator_kwargs_maps_dry_friction_to_upstream_keyword(self):
        from anchor_bench.runtime.robot_scene import _isaac_implicit_actuator_kwargs

        kwargs = _isaac_implicit_actuator_kwargs(
            {
                "stiffness": {"Rotation": 1.0},
                "damping": {"Rotation": 2.0},
                "dry_friction": {"Rotation": 0.3},
            }
        )

        self.assertEqual({"Rotation": 0.3}, kwargs["dynamic_friction"])
        self.assertNotIn("dry_friction", kwargs)

    def test_fk_startup_closes_sim_app_when_post_launch_setup_fails(self):
        from anchor_bench.analysis.runtime import EndEffectorFkUnavailable, IsaacLabNewtonRobotFk

        class FakeApp:
            closed = False

            def close(self):
                self.closed = True

        class FakeSimulationCfg:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.physics = None

        class FakeSimulationContext:
            def __init__(self, cfg):
                self.cfg = cfg

            def reset(self):
                pass

        class FakePhysicsCfg:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FakeRobot:
            body_names = []
            joint_names = []

        class FakeScene:
            def __init__(self, cfg):
                self.cfg = cfg
                self.robot = FakeRobot()

            def reset(self):
                pass

            def __getitem__(self, name):
                if name != "robot":
                    raise KeyError(name)
                return self.robot

        def fake_scene_cfg(**kwargs):
            return kwargs

        fake_app = FakeApp()
        fake_modules = {
            "torch": SimpleNamespace(),
            "isaaclab": SimpleNamespace(),
            "isaaclab.scene": SimpleNamespace(InteractiveScene=FakeScene),
            "isaaclab.sim": SimpleNamespace(SimulationCfg=FakeSimulationCfg, SimulationContext=FakeSimulationContext),
            "isaaclab_newton": SimpleNamespace(),
            "isaaclab_newton.physics": SimpleNamespace(MJWarpSolverCfg=FakePhysicsCfg, NewtonCfg=FakePhysicsCfg),
        }
        with patch("anchor_bench.runtime.assets.resolve_runtime_asset", return_value=Path("robot.usd")):
            with patch("anchor_bench.runtime.app_launcher.launch_isaac_sim", return_value=(object(), fake_app)):
                with patch("anchor_bench.runtime.robot_scene.build_scene_class", return_value=fake_scene_cfg):
                    with patch.dict(sys.modules, fake_modules):
                        with self.assertRaises(EndEffectorFkUnavailable):
                            IsaacLabNewtonRobotFk("so101", ee_frame="jaw")

        self.assertTrue(fake_app.closed)

    def test_benchmark_run_summary_uses_actual_num_envs(self):
        from anchor_bench.benchmark.runtime import _build_run_summary

        summary = _build_run_summary(
            motion_path=Path("inputs/motion"),
            motion_name="motion",
            dataset_name="dataset",
            actuator_model_name="implicit",
            seed=42,
            actual_num_envs=1,
            configured_batch_size=4,
            batch_size=1,
            batch_index=2,
            env_index=0,
            actuator_values={"stiffness": {"Rotation": 1.0}},
            actuator_params="best_params.yaml",
            robot_name="so101",
        )

        self.assertEqual(summary["num_envs"], 1)
        self.assertEqual(summary["configured_batch_size"], 4)
        self.assertEqual(summary["batch_size"], 1)
        self.assertFalse(summary["runtime_options"]["real_init_pose_sync"])

    def test_run_schema_accepts_generic_benchmark_init_pose_sync(self):
        from anchor_bench.core.schemas import RobotRunFile

        run_file = RobotRunFile.from_dict(
            {
                "benchmark": {
                    "motion_files": "inputs/datasets/example/motion",
                    "real_init_pose_sync": True,
                },
            },
            "configs/robots/example/run.yaml",
        )

        self.assertTrue(run_file.benchmark.real_init_pose_sync)

    def test_benchmark_sample_rows_preserve_motion_timestamps(self):
        from anchor_bench.benchmark.runtime import _append_benchmark_sample, _buffer_substeps

        control_rows: list[list[object]] = []
        state_rows: list[list[object]] = []
        self.assertEqual(1000, _buffer_substeps(buffer_time=2.0, control_dt=0.002, control_divisor=1))
        with self.assertRaisesRegex(ValueError, "benchmark.buffer_time must be >= 0"):
            _buffer_substeps(buffer_time=-1.0, control_dt=0.002, control_divisor=1)

        _append_benchmark_sample(
            control_rows,
            state_rows,
            timestamp_s=0.0,
            command=[0.0, 0.1],
            position=[0.0, 0.1],
            velocity=[0.0, 0.0],
            effort=[0.0, 0.0],
        )

        self.assertEqual(["CONTROL", "0.0", [0.0, 0.1]], control_rows[0])
        self.assertEqual(["STATE_MOTOR", "0.0", [0.0, 0.1], [0.0, 0.0], [0.0, 0.0]], state_rows[0])

        _append_benchmark_sample(
            control_rows,
            state_rows,
            timestamp_s=0.002,
            command=[0.01, 0.11],
            position=[0.01, 0.12],
            velocity=[5.0, 10.0],
            effort=[0.5, -0.25],
        )

        self.assertEqual(["CONTROL", "2000.0", [0.01, 0.11]], control_rows[1])
        self.assertAlmostEqual(state_rows[1][3][0], 5.0)
        self.assertAlmostEqual(state_rows[1][3][1], 10.0)

    def test_benchmark_sample_rows_preserve_effort(self):
        from anchor_bench.benchmark.runtime import _append_benchmark_sample

        control_rows: list[list[object]] = []
        state_rows: list[list[object]] = []

        _append_benchmark_sample(
            control_rows,
            state_rows,
            timestamp_s=0.002,
            command=[0.01, 0.11],
            position=[0.01, 0.12],
            velocity=[5.0, 10.0],
            effort=[0.5, -0.25],
        )

        self.assertEqual([0.5, -0.25], state_rows[0][4])

    def test_benchmark_state_reads_applied_torque(self):
        import numpy as np

        from anchor_bench.benchmark import runtime

        reader = getattr(runtime, "_read_benchmark_state", None)
        self.assertIsNotNone(reader)

        class FakeTensor:
            def __init__(self, values):
                self.values = np.asarray(values, dtype=float)

            def __getitem__(self, key):
                return FakeTensor(self.values[key])

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self.values

        robot = SimpleNamespace(
            data=SimpleNamespace(
                joint_pos=FakeTensor([[1.0, 2.0, 3.0]]),
                joint_vel=FakeTensor([[4.0, 5.0, 6.0]]),
                applied_torque=FakeTensor([[7.0, 8.0, 9.0]]),
            )
        )
        with patch("anchor_bench.benchmark.runtime.to_torch", side_effect=lambda value, _wp: value):
            position, velocity, effort = reader(robot, object(), [2, 0])

        np.testing.assert_allclose(position, [[3.0, 1.0]])
        np.testing.assert_allclose(velocity, [[6.0, 4.0]])
        np.testing.assert_allclose(effort, [[9.0, 7.0]])

    def test_optimizer_requires_cmaes_unless_random_fallback_is_explicit(self):
        from anchor_bench.core.schemas import CMAESConfig
        from anchor_bench.robots.registry import get_robot
        from anchor_bench.runtime.robot_config import load_sysid_bounds
        from anchor_bench.sysid.optimizer import CMAESOptimizer

        bounds = load_sysid_bounds(get_robot("so101"))
        with patch.dict("sys.modules", {"cmaes": None}):
            with self.assertRaisesRegex(RuntimeError, "cmaes package is required"):
                CMAESOptimizer(bounds, num_envs=4, cmaes=CMAESConfig(), seed=11)

    def test_optimizer_sampling_is_seed_deterministic(self):
        from anchor_bench.core.schemas import CMAESConfig
        from anchor_bench.sysid.optimizer import CMAESOptimizer
        from anchor_bench.runtime.robot_config import load_sysid_bounds
        from anchor_bench.robots.registry import get_robot

        bounds = load_sysid_bounds(get_robot("so101"))
        first = CMAESOptimizer(bounds, num_envs=3, cmaes=CMAESConfig(), seed=11, allow_random_fallback=True).sample_population()
        second = CMAESOptimizer(bounds, num_envs=3, cmaes=CMAESConfig(), seed=11, allow_random_fallback=True).sample_population()
        different = CMAESOptimizer(bounds, num_envs=3, cmaes=CMAESConfig(), seed=12, allow_random_fallback=True).sample_population()

        self.assertEqual(first, second)
        self.assertNotEqual(first, different)

    def test_optimizer_tell_tracks_best_candidate(self):
        from anchor_bench.core.schemas import CMAESConfig
        from anchor_bench.robots.registry import get_robot
        from anchor_bench.runtime.robot_config import load_sysid_bounds
        from anchor_bench.sysid.optimizer import CMAESOptimizer

        bounds = load_sysid_bounds(get_robot("so101"))
        optimizer = CMAESOptimizer(bounds, num_envs=3, cmaes=CMAESConfig(), seed=11, allow_random_fallback=True)
        population = optimizer.sample_population()
        optimizer.tell([2.0, 0.5, 1.0])

        self.assertEqual(optimizer.best_params, population[1])
        self.assertEqual(optimizer.best_score, 0.5)

    def test_optimizer_does_not_stop_on_same_generation_score_spread(self):
        from anchor_bench.core.schemas import CMAESConfig
        from anchor_bench.robots.registry import get_robot
        from anchor_bench.runtime.robot_config import load_sysid_bounds
        from anchor_bench.sysid.optimizer import CMAESOptimizer

        bounds = load_sysid_bounds(get_robot("so101"))
        optimizer = CMAESOptimizer(bounds, num_envs=3, cmaes=CMAESConfig(), seed=11, allow_random_fallback=True)

        optimizer.sample_population()

        self.assertFalse(optimizer.tell([100.0, 100.1, 100.05]))

    def test_optimizer_stops_after_best_score_stalls_beyond_minimum_window(self):
        from anchor_bench.core.schemas import CMAESConfig
        from anchor_bench.robots.registry import get_robot
        from anchor_bench.runtime.robot_config import load_sysid_bounds
        from anchor_bench.sysid.optimizer import CMAESOptimizer, _MIN_CONVERGENCE_GENERATIONS

        bounds = load_sysid_bounds(get_robot("so101"))
        optimizer = CMAESOptimizer(bounds, num_envs=3, cmaes=CMAESConfig(), seed=11, allow_random_fallback=True)
        converged = False

        for _ in range(_MIN_CONVERGENCE_GENERATIONS):
            optimizer.sample_population()
            converged = optimizer.tell([1.0, 1.001, 1.002])
            self.assertFalse(converged)
        optimizer.sample_population()
        converged = optimizer.tell([1.0, 1.001, 1.002])

        self.assertTrue(converged)

    def test_optimizer_clips_cma_samples_before_denormalizing(self):
        import numpy as np

        from anchor_bench.core.schemas import CMAESConfig
        from anchor_bench.robots.registry import get_robot
        from anchor_bench.runtime.robot_config import load_sysid_bounds
        from anchor_bench.sysid.optimizer import CMAESOptimizer

        class FakeCMA:
            def __init__(self, samples):
                self.samples = list(samples)
                self.told = None

            def ask(self):
                return self.samples.pop(0)

            def tell(self, solutions):
                self.told = solutions

        bounds = load_sysid_bounds(get_robot("so101"))
        optimizer = CMAESOptimizer(bounds, num_envs=2, cmaes=CMAESConfig(), seed=11, allow_random_fallback=True)
        dimension = len(optimizer.param_names)
        fake_cma = FakeCMA(
            [
                np.linspace(-1.5, 1.5, dimension),
                np.linspace(1.5, -1.5, dimension),
            ]
        )
        optimizer._cma = fake_cma

        optimizer.sample_matrix()

        self.assertTrue(np.all(optimizer._last_norm >= -1.0))
        self.assertTrue(np.all(optimizer._last_norm <= 1.0))
        self.assertTrue(np.all(optimizer._last_matrix >= optimizer._lower))
        self.assertTrue(np.all(optimizer._last_matrix <= optimizer._upper))

        optimizer.tell([0.25, 1.0])

        self.assertEqual(optimizer.best_params, optimizer._row_to_candidate(optimizer._last_matrix[0]))
        self.assertTrue(np.all(np.asarray(fake_cma.told[0][0], dtype=np.float64) >= -1.0))
        self.assertTrue(np.all(np.asarray(fake_cma.told[0][0], dtype=np.float64) <= 1.0))

    def test_batched_fk_helper_preserves_order_and_uses_chunks(self):
        import numpy as np

        from anchor_bench.analysis.runtime import _batched_fk_positions

        positions = np.arange(10, dtype=float).reshape(5, 2)
        calls = []

        def evaluate_chunk(chunk):
            calls.append(chunk.copy())
            return np.column_stack((chunk[:, 0], chunk[:, 1], chunk.sum(axis=1)))

        xyz = _batched_fk_positions(positions, batch_size=2, evaluate_chunk=evaluate_chunk)

        np.testing.assert_allclose(
            xyz,
            np.array(
                [
                    [0.0, 1.0, 1.0],
                    [2.0, 3.0, 5.0],
                    [4.0, 5.0, 9.0],
                    [6.0, 7.0, 13.0],
                    [8.0, 9.0, 17.0],
                ]
            ),
        )
        self.assertEqual([call.shape[0] for call in calls], [2, 2, 1])

    def test_analysis_writes_full_metrics_and_plots_for_sage_benchmark_layout(self):
        from anchor_bench.analysis.runtime import run_robot_analysis

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real = root / "bench" / "real" / "so101" / "example" / "motion"
            sim = root / "bench" / "sim" / "so101" / "example" / "implicit" / "motion"
            sim_lstm = root / "bench" / "sim" / "so101" / "example" / "lstm" / "motion"
            for folder in (real, sim, sim_lstm):
                folder.mkdir(parents=True)
                (folder / "joint_list.txt").write_text("Rotation\nPitch\n", encoding="utf-8")
                write_sage_metadata(folder)
                (folder / "control.csv").write_text(
                    'type,timestamp,positions\nCONTROL,0.0,"[0.0, 0.0]"\nCONTROL,1000.0,"[1.0, 2.0]"\n',
                    encoding="utf-8",
                )
            real_state = (
                'type,timestamp,positions,velocities,torques\n'
                'STATE_MOTOR,0.0,"[0.0, 0.0]","[0.0, 0.0]","[0.0, 0.0]"\n'
                'STATE_MOTOR,1000.0,"[1.0, 2.5]","[2.0, 4.0]","[3.0, 5.0]"\n'
            )
            sim_state = (
                'type,timestamp,positions,velocities,torques\n'
                'STATE_MOTOR,0.0,"[0.0, 0.0]","[0.0, 0.0]","[0.0, 0.0]"\n'
                'STATE_MOTOR,1000.0,"[1.0, 2.0]","[1.0, 2.0]","[1.0, 3.0]"\n'
            )
            (real / "state_motor.csv").write_text(real_state, encoding="utf-8")
            (sim / "state_motor.csv").write_text(sim_state, encoding="utf-8")
            (sim_lstm / "state_motor.csv").write_text(sim_state, encoding="utf-8")
            output = root / "analysis"

            def fake_fk(joint_names, positions):
                return positions[:, :2].sum(axis=1, keepdims=True).repeat(3, axis=1)

            run_robot_analysis("so101", root / "bench", output, ee_fk=fake_fk)

            metrics = (output / "metrics" / "so101_metrics.csv").read_text(encoding="utf-8")
            self.assertIn(
                "dataset,motion,actuator_model,seed,joint,position_rmse,velocity_rmse,torque_rmse,"
                "position_correlation,velocity_correlation,torque_correlation,"
                "position_cosine_similarity,velocity_cosine_similarity,torque_cosine_similarity",
                metrics,
            )
            self.assertIn("example,motion,implicit,,Rotation,0.0,0.707107,1.414214", metrics)
            self.assertIn("example,motion,implicit,,Pitch,0.353553,1.414214,1.414214", metrics)
            self.assertIn("example,motion,lstm,,Rotation,0.0,0.707107,1.414214", metrics)

            plot_dir = output / "plots" / "example" / "motion" / "joints"
            self.assertEqual(["Pitch.png", "Rotation.png"], sorted(path.name for path in plot_dir.glob("*.png")))
            self.assertFalse((output / "plots" / "example" / "motion_implicit").exists())
            self.assertTrue((output / "plots" / "example" / "motion" / "end_effector.png").exists())

            ee_metrics = (output / "metrics" / "so101_end_effector_metrics.csv").read_text(encoding="utf-8")
            self.assertIn(
                "dataset,motion,actuator_model,seed,rmse_x_m,rmse_y_m,rmse_z_m,rmse_dist_m,max_err_m,duration_s",
                ee_metrics,
            )
            self.assertIn("example,motion,implicit,", ee_metrics)

    def test_analysis_sample_dt_resamples_trace_vectors(self):
        from anchor_bench.analysis.runtime import _read_trace

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "joint_list.txt").write_text("Rotation\n", encoding="utf-8")
            write_sage_metadata(folder)
            (folder / "control.csv").write_text(
                'type,timestamp,positions\n'
                'CONTROL,0.0,"[0.0]"\n'
                'CONTROL,1.0,"[1.0]"\n',
                encoding="utf-8",
            )
            (folder / "state_motor.csv").write_text(
                'type,timestamp,positions,velocities,torques\n'
                'STATE_MOTOR,0.0,"[0.0]","[0.0]","[1.0]"\n'
                'STATE_MOTOR,1.0,"[2.0]","[4.0]","[5.0]"\n',
                encoding="utf-8",
            )

            trace = _read_trace(folder, sample_dt=0.5)

        self.assertEqual(trace.command_position.time.tolist(), [0.0, 0.5, 1.0])
        self.assertEqual(trace.position.time.tolist(), [0.0, 0.5, 1.0])
        self.assertEqual(trace.command_position.values[:, 0].tolist(), [0.0, 0.5, 1.0])
        self.assertEqual(trace.position.values[:, 0].tolist(), [0.0, 1.0, 2.0])
        self.assertEqual(trace.velocity.values[:, 0].tolist(), [0.0, 2.0, 4.0])
        self.assertEqual(trace.effort.values[:, 0].tolist(), [1.0, 3.0, 5.0])

    def test_analysis_motion_filter_supports_globs(self):
        from anchor_bench.analysis.runtime import run_robot_analysis

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for motion in ("elbow_sweep", "hold"):
                real = root / "bench" / "real" / "so101" / "example" / motion
                sim = root / "bench" / "sim" / "so101" / "example" / "implicit" / motion
                for folder in (real, sim):
                    folder.mkdir(parents=True)
                    (folder / "joint_list.txt").write_text("Rotation\n", encoding="utf-8")
                    write_sage_metadata(folder)
                    (folder / "control.csv").write_text(
                        'type,timestamp,positions\nCONTROL,0.0,"[0.0]"\nCONTROL,1000.0,"[1.0]"\n',
                        encoding="utf-8",
                    )
                    (folder / "state_motor.csv").write_text(
                        'type,timestamp,positions,velocities,torques\n'
                        'STATE_MOTOR,0.0,"[0.0]","[0.0]","[0.0]"\n'
                        'STATE_MOTOR,1000.0,"[1.0]","[1.0]","[1.0]"\n',
                        encoding="utf-8",
                    )

            output = root / "analysis"
            run_robot_analysis("so101", root / "bench", output, motions=["*sweep*"], make_end_effector_plots=False)
            metrics = (output / "metrics" / "so101_metrics.csv").read_text(encoding="utf-8")

        self.assertIn("example,elbow_sweep,implicit,", metrics)
        self.assertNotIn("example,hold,implicit,", metrics)

    def test_analysis_skips_end_effector_when_fk_runtime_is_missing(self):
        from anchor_bench.analysis.runtime import EndEffectorFkUnavailable, run_robot_analysis

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real = root / "bench" / "real" / "so101" / "example" / "motion"
            sim = root / "bench" / "sim" / "so101" / "example" / "implicit" / "motion"
            for folder in (real, sim):
                folder.mkdir(parents=True)
                (folder / "joint_list.txt").write_text("Rotation\nPitch\n", encoding="utf-8")
                write_sage_metadata(folder)
                (folder / "control.csv").write_text(
                    'type,timestamp,positions\nCONTROL,0.0,"[0.0, 0.0]"\nCONTROL,1000.0,"[1.0, 2.0]"\n',
                    encoding="utf-8",
                )
                (folder / "state_motor.csv").write_text(
                    'type,timestamp,positions,velocities,torques\n'
                    'STATE_MOTOR,0.0,"[0.0, 0.0]","[0.0, 0.0]","[0.0, 0.0]"\n'
                    'STATE_MOTOR,1000.0,"[1.0, 2.0]","[1.0, 2.0]","[1.0, 3.0]"\n',
                    encoding="utf-8",
                )

            def missing_fk(joint_names, positions):
                raise EndEffectorFkUnavailable("isaaclab-newton runtime unavailable")

            output = root / "analysis"
            run_robot_analysis("so101", root / "bench", output, ee_fk=missing_fk)

            self.assertTrue((output / "metrics" / "so101_metrics.csv").exists())
            self.assertTrue((output / "plots" / "example" / "motion" / "joints" / "Rotation.png").exists())
            self.assertFalse((output / "metrics" / "so101_end_effector_metrics.csv").exists())

    def test_analysis_requires_end_effector_when_requested(self):
        from anchor_bench.analysis.runtime import EndEffectorFkUnavailable, run_robot_analysis

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real = root / "bench" / "real" / "so101" / "example" / "motion"
            sim = root / "bench" / "sim" / "so101" / "example" / "implicit" / "motion"
            for folder in (real, sim):
                folder.mkdir(parents=True)
                (folder / "joint_list.txt").write_text("Rotation\nPitch\n", encoding="utf-8")
                write_sage_metadata(folder)
                (folder / "control.csv").write_text(
                    'type,timestamp,positions\nCONTROL,0.0,"[0.0, 0.0]"\nCONTROL,1000.0,"[1.0, 2.0]"\n',
                    encoding="utf-8",
                )
                (folder / "state_motor.csv").write_text(
                    'type,timestamp,positions,velocities,torques\n'
                    'STATE_MOTOR,0.0,"[0.0, 0.0]","[0.0, 0.0]","[0.0, 0.0]"\n'
                    'STATE_MOTOR,1000.0,"[1.0, 2.0]","[1.0, 2.0]","[1.0, 3.0]"\n',
                    encoding="utf-8",
                )

            def missing_fk(joint_names, positions):
                raise EndEffectorFkUnavailable("isaaclab-newton runtime unavailable")

            with self.assertRaisesRegex(EndEffectorFkUnavailable, "isaaclab-newton runtime unavailable"):
                run_robot_analysis("so101", root / "bench", root / "analysis", ee_fk=missing_fk, require_end_effector=True)

    def test_analysis_places_multi_seed_plots_above_dataset(self):
        from anchor_bench.analysis.runtime import run_robot_analysis

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real = root / "bench" / "seed_42" / "real" / "so101" / "example" / "motion"
            sim = root / "bench" / "seed_42" / "sim" / "so101" / "example" / "implicit" / "motion"
            for folder in (real, sim):
                folder.mkdir(parents=True)
                (folder / "joint_list.txt").write_text("Rotation\n", encoding="utf-8")
                write_sage_metadata(folder)
                (folder / "control.csv").write_text(
                    'type,timestamp,positions\nCONTROL,0.0,"[0.0]"\nCONTROL,1000.0,"[1.0]"\n',
                    encoding="utf-8",
                )
                (folder / "state_motor.csv").write_text(
                    'type,timestamp,positions,velocities,torques\n'
                    'STATE_MOTOR,0.0,"[0.0]","[0.0]","[0.0]"\n'
                    'STATE_MOTOR,1000.0,"[1.0]","[1.0]","[1.0]"\n',
                    encoding="utf-8",
                )
            output = root / "analysis"

            run_robot_analysis("so101", root / "bench", output, make_end_effector_plots=False, seeds=[42])

            metrics = (output / "metrics" / "so101_metrics.csv").read_text(encoding="utf-8")
            self.assertIn("example,motion,implicit,42,Rotation", metrics)
            self.assertTrue((output / "plots" / "seed_42" / "example" / "motion" / "joints" / "Rotation.png").exists())


if __name__ == "__main__":
    unittest.main()
