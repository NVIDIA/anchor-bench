"""Built-in training data processors."""

from importlib import import_module

from anchor_bench.train.data_processors.base import BaseDataProcessor

DATA_PROCESSORS = {
    "position_error": (".gru", "PositionErrorDataProcessor"),
    "previous_torque": (".gru", "PreviousTorqueDataProcessor"),
    "qfrc_bias": (".gru", "DynamicBiasDataProcessor"),
    "solver_pd": (".gru", "SolverPDDataProcessor"),
    "torque_residual": (".gru", "ResidualTorqueDataProcessor"),
}


def __getattr__(name: str) -> type[BaseDataProcessor]:
    """Import a built-in data processor only when it is requested."""

    try:
        module_name, class_name = DATA_PROCESSORS[name]
    except KeyError as error:
        raise AttributeError(f"Unknown data processor {name!r}.") from error

    module = import_module(module_name, __name__)
    return getattr(module, class_name)


__all__ = ["BaseDataProcessor"]
