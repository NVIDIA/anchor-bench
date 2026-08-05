"""Instantiate configured training validators."""

from __future__ import annotations

from typing import Any, Iterable, Literal, Mapping, Protocol

from anchor_bench.train.validators import BaseValidator


class _TrainingValidatorConfig(Protocol):
    """Structural view of one training validator configuration."""

    @property
    def name(self) -> str: ...

    @property
    def type(self) -> Literal["offline", "simulation"]: ...

    @property
    def entrypoint(self) -> str: ...

    @property
    def config(self) -> Mapping[str, Any]: ...


class ValidatorRunner:
    """Resolve and instantiate configured built-in validators."""

    @staticmethod
    def instantiate(
        validators: Iterable[_TrainingValidatorConfig],
    ) -> list[BaseValidator]:
        """Construct validators in declaration order without executing them."""

        import anchor_bench.train.validators as validator_registry

        instances = []
        for validator in validators:
            validator_class = getattr(validator_registry, validator.entrypoint)
            if not isinstance(validator_class, type) or not issubclass(
                validator_class,
                BaseValidator,
            ):
                raise TypeError(
                    f"Registered validator {validator.entrypoint!r} must inherit "
                    "from BaseValidator."
                )

            instances.append(
                validator_class(
                    name=validator.name,
                    type=validator.type,
                    config=validator.config,
                )
            )
        return instances
