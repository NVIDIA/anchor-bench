
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import yaml

from anchor_bench.actuators.loader import load_actuator_file
from anchor_bench.core.paths import repo_root


def _valid_robot_spec() -> dict:
    return {
        "schema_version": 1,
        "name": "demo",
        "asset_path": "assets/robots/demo/files/demo.usd",
        "joint_names": ["JointA"],
    }


def _valid_actuator_spec() -> dict:
    return {
        "name": "demo",
        "joints": ["JointA"],
        "controller": {"type": "pd"},
    }


def _valid_bounds_file() -> dict:
    return {
        "schema_version": 2,
        "mirror": False,
        "joint_types": ["JointA"],
        "parameters": {"armature": {"lower": 0.0, "upper": 1.0}},
        "fixed": {"motor_lag_ms": 0.0},
    }


class NativeActuatorSchemaTests(unittest.TestCase):
    def test_no_legacy_model_type_blocks_remain_in_robot_configs(self):
        for path in (repo_root() / "configs" / "robots").glob("*/actuators.yaml"):
            text = path.read_text()
            self.assertNotIn("model_type:", text, f"{path} still uses legacy model_type")

    def test_so101_uses_newton_component_schema(self):
        spec = load_actuator_file(repo_root() / "configs" / "robots" / "so101" / "actuators.yaml")
        actuator = spec.actuators[0]
        self.assertEqual(spec.backend, "newton")
        self.assertEqual(actuator.controller.type, "pd")
        self.assertGreater(len(actuator.clamps), 0)
        self.assertIn("armature", actuator.physical_parameters)
        self.assertIn("viscous_friction", actuator.physical_parameters)

    def test_delay_schema_rejects_unknown_keys(self):
        from anchor_bench.core.schemas import DelaySpec

        with self.assertRaisesRegex(ValueError, "actuator.delay has unknown keys: typo"):
            DelaySpec.from_dict({"duration_ms": 2.0, "typo": 1.0}, "actuator.delay")

    def test_schema_layers_reject_unknown_keys(self):
        from anchor_bench.core.schemas import (
            ActuatorSpec,
            BoundsRange,
            CMAESConfig,
            ClampingSpec,
            ControllerSpec,
            RobotSpec,
            SysIdBoundsFile,
        )

        actuator = _valid_actuator_spec()
        actuator["typo"] = True
        robot = _valid_robot_spec()
        robot["typo"] = True
        bounds = _valid_bounds_file()
        bounds["typo"] = True

        cases = [
            (lambda: RobotSpec.from_dict(robot), "robot has unknown keys: typo"),
            (lambda: ActuatorSpec.from_dict(actuator, "actuator"), "actuator has unknown keys: typo"),
            (
                lambda: ControllerSpec.from_dict({"type": "pd", "typo": True}, "controller"),
                "controller has unknown keys: typo",
            ),
            (
                lambda: ClampingSpec.from_dict({"type": "effort", "typo": True}, "clamp"),
                "clamp has unknown keys: typo",
            ),
            (
                lambda: BoundsRange.from_dict({"lower": 0.0, "upper": 1.0, "typo": True}, "range"),
                "range has unknown keys: typo",
            ),
            (
                lambda: CMAESConfig.from_dict({"sigma": 0.5, "typo": True}, "cmaes"),
                "cmaes has unknown keys: typo",
            ),
            (
                lambda: SysIdBoundsFile.from_dict(bounds, "bounds.yaml"),
                "bounds.yaml has unknown keys: typo",
            ),
        ]

        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    call()

    def test_bounds_schema_rejects_cmaes_policy(self):
        from anchor_bench.core.schemas import SysIdBoundsFile

        bounds = _valid_bounds_file()
        bounds["cmaes"] = {"sigma": 0.5, "max_iterations": 200, "epsilon": 0.01}

        with self.assertRaisesRegex(ValueError, "bounds.yaml has unknown keys: cmaes"):
            SysIdBoundsFile.from_dict(bounds, "bounds.yaml")

    def test_robot_run_schema_accepts_nested_cmaes_policy(self):
        from anchor_bench.core.schemas import RobotRunFile

        run = RobotRunFile.from_dict(
            {
                "sysid": {
                    "num_envs": 64,
                    "buffer_time": 2.0,
                    "cmaes": {
                        "sigma": 0.25,
                        "max_iterations": 30,
                        "epsilon": 0.001,
                    },
                }
            },
            "run.yaml",
        )

        self.assertEqual(run.sysid["num_envs"], 64)
        self.assertEqual(run.sysid["cmaes"]["sigma"], 0.25)
        self.assertEqual(run.sysid["cmaes"]["max_iterations"], 30)
        self.assertEqual(run.sysid["cmaes"]["epsilon"], 0.001)

    def test_robot_run_schema_rejects_flat_cmaes_policy_keys(self):
        from anchor_bench.core.schemas import RobotRunFile

        for key in ("sigma", "epsilon", "max_iter"):
            with self.subTest(key=key):
                data = {"sysid": {key: 1.0}}
                with self.assertRaisesRegex(ValueError, f"run.yaml.sysid uses unsupported flat CMA-ES key '{key}'"):
                    RobotRunFile.from_dict(data, "run.yaml")



    def test_robot_run_schema_accepts_training_defaults(self):
        from anchor_bench.core.schemas import RobotRunFile

        run = RobotRunFile.from_dict(
            {
                "training": {
                    "model": "hybrid_residual",
                    "real_data_dir": "inputs/training/demo",
                    "output_dir": "outputs/training/demo",
                    "joint_type": "Rotation",
                    "skip_optuna": True,
                    "trials": 3,
                    "trial_epochs": 1,
                    "final_epochs": 2,
                    "n_streams": 4,
                    "noise_std": 0.01,
                    "seed": 7,
                }
            },
            "run.yaml",
        )

        self.assertEqual("hybrid_residual", run.training["model"])
        self.assertEqual("outputs/training/demo", run.training["output_dir"])
        self.assertTrue(run.training["skip_optuna"])
        self.assertEqual(4, run.training["n_streams"])

    def test_robot_run_schema_rejects_unknown_training_keys(self):
        from anchor_bench.core.schemas import RobotRunFile

        with self.assertRaisesRegex(ValueError, "run.yaml.training has unknown keys: typo"):
            RobotRunFile.from_dict({"training": {"model": "gru", "typo": True}}, "run.yaml")

    def test_gru_controller_requires_runtime_network_metadata(self):
        from anchor_bench.core.schemas import ActuatorSpec, ControllerSpec

        with self.assertRaisesRegex(ValueError, "controller.type 'gru' requires network_file"):
            ControllerSpec.from_dict({"type": "gru"}, "controller")

        actuator = _valid_actuator_spec()
        actuator["controller"] = {
            "type": "gru",
            "network_file": "outputs/training/demo/gru_JointA_network.pt",
            "parameters": {
                "stats_file": "outputs/training/demo/gru_JointA_stats.json",
                "input_columns": ["position", "position_error", "velocity"],
            },
        }
        parsed = ActuatorSpec.from_dict(actuator, "actuator")

        self.assertEqual("gru", parsed.controller.type)
        self.assertEqual("outputs/training/demo/gru_JointA_network.pt", parsed.controller.network_file)
        self.assertEqual(["position", "position_error", "velocity"], parsed.controller.parameters["input_columns"])

    def test_gru_residual_metadata_is_validated(self):
        from anchor_bench.core.schemas import ActuatorSpec

        actuator = _valid_actuator_spec()
        actuator["residual"] = {
            "type": "gru",
            "network_file": "outputs/training/demo/hybrid_residual_JointA_network.pt",
            "stats_file": "outputs/training/demo/hybrid_residual_JointA_stats.json",
            "input_columns": ["position", "position_error", "velocity", "solver_pd"],
        }
        parsed = ActuatorSpec.from_dict(actuator, "actuator")

        self.assertIsNotNone(parsed.residual)
        self.assertEqual("gru", parsed.residual.type)
        self.assertEqual("torque_residual", parsed.residual.output_column)
        self.assertEqual(["position", "position_error", "velocity", "solver_pd"], parsed.residual.input_columns)

        actuator["residual"] = {"type": "gru", "network_file": "net.pt", "typo": True}
        with self.assertRaisesRegex(ValueError, "actuator.residual has unknown keys: typo"):
            ActuatorSpec.from_dict(actuator, "actuator")

        actuator["residual"] = {
            "type": "gru",
            "network_file": "net.pt",
            "input_columns": ["position", "qfrc_bias"],
        }
        with self.assertRaisesRegex(ValueError, "unsupported GRU input columns: qfrc_bias"):
            ActuatorSpec.from_dict(actuator, "actuator")

    def test_instantiate_newton_actuators_delegates_pd_to_newton_adapter(self):
        from anchor_bench.actuators import loader
        from anchor_bench.core.schemas import ActuatorFile

        spec = ActuatorFile.from_dict(
            {
                "schema_version": 2,
                "backend": "newton",
                "actuators": [
                    {
                        "name": "demo",
                        "joints": ["JointA"],
                        "controller": {
                            "type": "pd",
                            "parameters": {"stiffness": 20.0, "damping": 1.5},
                        },
                        "clamps": [{"type": "effort", "parameters": {"effort_limit": 3.0}}],
                    }
                ],
            },
            "demo.yaml",
        )
        calls = {}

        class FakeNewtonActuatorAdapter:
            @classmethod
            def from_usd(cls, **kwargs):
                calls["from_usd"] = kwargs
                return SimpleNamespace(kind="newton-adapter")

        fake_package = ModuleType("isaaclab_newton")
        fake_actuators = ModuleType("isaaclab_newton.actuators")
        fake_actuators.NewtonActuatorAdapter = FakeNewtonActuatorAdapter

        stage = object()
        with patch.dict(
            "sys.modules",
            {"isaaclab_newton": fake_package, "isaaclab_newton.actuators": fake_actuators},
        ):
            with patch.object(loader, "author_newton_actuator_prims") as author:
                adapter = loader.instantiate_newton_actuators(
                    spec,
                    stage=stage,
                    articulation_prim_path="/World/Robot",
                    joint_names=["JointA"],
                    num_envs=2,
                    num_joints=1,
                    device="cuda:0",
                )

        self.assertEqual("newton-adapter", adapter.kind)
        author.assert_called_once_with(
            spec,
            stage=stage,
            articulation_prim_path="/World/Robot",
        )
        self.assertEqual(stage, calls["from_usd"]["stage"])
        self.assertEqual(["JointA"], calls["from_usd"]["joint_names"])
        self.assertEqual(2, calls["from_usd"]["num_envs"])
        self.assertEqual(1, calls["from_usd"]["num_joints"])
        self.assertEqual("cuda:0", calls["from_usd"]["device"])
        self.assertEqual("/World/Robot", calls["from_usd"]["articulation_prim_path"])

    def test_instantiate_newton_actuators_rejects_gru_runtime_shim_boundary(self):
        from anchor_bench.actuators.loader import instantiate_newton_actuators
        from anchor_bench.core.schemas import ActuatorFile

        spec = ActuatorFile.from_dict(
            {
                "schema_version": 2,
                "backend": "newton",
                "actuators": [
                    {
                        "name": "demo",
                        "joints": ["JointA"],
                        "controller": {
                            "type": "gru",
                            "network_file": "outputs/training/demo/gru_JointA_network.pt",
                        },
                    }
                ],
            },
            "demo.yaml",
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "GRU actuators are not Newton-native.*anchor_bench.runtime.actuator_net_gru",
        ):
            instantiate_newton_actuators(
                spec,
                stage=object(),
                articulation_prim_path="/World/Robot",
                joint_names=["JointA"],
                num_envs=2,
                num_joints=1,
                device="cuda:0",
            )

    def test_instantiate_newton_actuators_requires_runtime_context(self):
        from anchor_bench.actuators.loader import instantiate_newton_actuators
        from anchor_bench.core.schemas import ActuatorFile

        spec = ActuatorFile.from_dict(
            {
                "schema_version": 2,
                "backend": "newton",
                "actuators": [
                    {
                        "name": "demo",
                        "joints": ["JointA"],
                        "controller": {
                            "type": "gru",
                            "network_file": "outputs/training/demo/gru_JointA_network.pt",
                        },
                    }
                ],
            },
            "demo.yaml",
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "stage, articulation_prim_path, joint_names, num_envs, num_joints, and device",
        ):
            instantiate_newton_actuators(spec)

    def test_robot_registry_loads_validated_run_config(self):
        from anchor_bench.robots.registry import get_robot

        bundle = get_robot("so101")

        self.assertEqual(100, bundle.run_config["sysid"]["cmaes"]["max_iterations"])
        self.assertNotIn("max_iter", bundle.run_config["sysid"])

    def test_bounds_configs_do_not_contain_cmaes_policy(self):
        for path in (repo_root() / "configs" / "robots").glob("*/bounds.yaml"):
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            self.assertNotIn("cmaes", data, f"{path} should keep CMA-ES policy in run.yaml")

    def test_run_configs_use_nested_cmaes_policy(self):
        for path in (repo_root() / "configs" / "robots").glob("*/run.yaml"):
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            sysid = data.get("sysid")
            if not sysid:
                continue
            self.assertNotIn("sigma", sysid, f"{path} should use sysid.cmaes.sigma")
            self.assertNotIn("epsilon", sysid, f"{path} should use sysid.cmaes.epsilon")
            self.assertNotIn("max_iter", sysid, f"{path} should use sysid.cmaes.max_iterations")
            self.assertIn("cmaes", sysid, f"{path} should define sysid.cmaes")

    def test_legacy_migration_utility_removes_model_type(self):
        from anchor_bench.migration.legacy import migrate_legacy_actuator_config

        migrated = migrate_legacy_actuator_config(
            {
                "model_type": "implicit",
                "yaml_file": "so101/so101_implicit.yaml",
                "motor_lag_ms": 2.0,
                "parameters": {
                    "stiffness": 4.0,
                    "damping": 0.1,
                    "effort_limit_sim": 1.5,
                    "velocity_limit_sim": 6.28,
                    "dynamic_friction": {"Rotation": 0.25},
                },
            },
            robot_name="so101",
            joint_names=["Rotation"],
        )
        self.assertNotIn("model_type", yaml.safe_dump(migrated))
        self.assertEqual(migrated["actuators"][0]["controller"]["type"], "pd")
        self.assertEqual(migrated["actuators"][0]["delay"]["duration_ms"], 2.0)
        physical = migrated["actuators"][0]["physical_parameters"]
        self.assertEqual({"Rotation": 0.25}, physical["dry_friction"])
        self.assertNotIn("dynamic_friction", physical)


if __name__ == "__main__":
    unittest.main()
