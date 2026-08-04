"""Built-in training validators."""

from importlib import import_module

from anchor_bench.train.validators.base import (
    BaseValidator,
    ValidationResult,
    ValidatorType,
)

# Built-in validators are registered here as name: (module, class) entries.
VALIDATORS: dict[str, tuple[str, str]] = {}


def __getattr__(name: str) -> type[BaseValidator]:
    """Import a built-in validator only when it is requested."""

    try:
        module_name, class_name = VALIDATORS[name]
    except KeyError as error:
        raise AttributeError(f"Unknown validator {name!r}.") from error

    module = import_module(module_name, __name__)
    return getattr(module, class_name)


__all__ = ["BaseValidator", "ValidationResult", "ValidatorType"]
