
import unittest


class ActuatorNetGRUTests(unittest.TestCase):
    def test_cfg_classes_import_without_full_runtime(self):
        from anchor_bench.runtime.actuator_net_gru import ActuatorNetGRUCfg, ActuatorNetGRUResidualCfg

        full = ActuatorNetGRUCfg(
            joint_names_expr=["JointA"],
            network_file="outputs/training/demo/gru_JointA_network.pt",
            stats_file="outputs/training/demo/gru_JointA_stats.json",
        )
        residual = ActuatorNetGRUResidualCfg(
            joint_names_expr=["JointA"],
            network_file="outputs/training/demo/hybrid_residual_JointA_network.pt",
        )

        self.assertEqual(("position", "position_error", "velocity"), tuple(full.input_columns))
        self.assertEqual("torque", full.output_column)
        self.assertEqual("torque_residual", residual.output_column)

    def test_configclass_import_normalizer_accepts_module_shape(self):
        from types import SimpleNamespace

        from anchor_bench.runtime.actuator_net_gru import _coerce_configclass

        direct = lambda cls: cls
        nested = lambda cls: cls

        self.assertIs(_coerce_configclass(direct), direct)
        self.assertIs(_coerce_configclass(SimpleNamespace(configclass=nested)), nested)

    def test_runtime_input_column_validator_rejects_unsupported_columns(self):
        from anchor_bench.runtime.actuator_net_gru import validate_gru_input_columns

        self.assertEqual(
            ("position", "solver_pd"),
            validate_gru_input_columns(["position", "solver_pd"]),
        )
        with self.assertRaisesRegex(ValueError, "unsupported GRU input columns: previous_torque"):
            validate_gru_input_columns(["position", "previous_torque"])


if __name__ == "__main__":
    unittest.main()
