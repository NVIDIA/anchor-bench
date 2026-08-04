"""Built-in training preprocessors."""

from importlib import import_module

from anchor_bench.train.preprocessors.base import BasePreprocessor

PREPROCESSORS = {
    "position_error": (".gru", "PositionErrorPreprocessor"),
    "solver_pd": (".gru", "SolverPDPreprocessor"),
    "torque_residual": (".gru", "ResidualTorquePreprocessor"),
}


def __getattr__(name: str) -> type[BasePreprocessor]:
    """Import a built-in preprocessor only when it is requested."""

    try:
        module_name, class_name = PREPROCESSORS[name]
    except KeyError as error:
        raise AttributeError(f"Unknown preprocessor {name!r}.") from error

    module = import_module(module_name, __name__)
    return getattr(module, class_name)


__all__ = ["BasePreprocessor"]
