# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Instantiate and execute configured training validators."""

from __future__ import annotations

import logging
from typing import Any, Iterable, Literal, Mapping, Protocol

from anchor_bench.tasks.train.validators import BaseValidator, ValidationResult

logger = logging.getLogger(__name__)


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

        import anchor_bench.tasks.train.validators as validator_registry

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

    @staticmethod
    def run(
        validators: Iterable[_TrainingValidatorConfig],
        *,
        model: Any,
        validation_dataset: Any,
    ) -> list[ValidationResult]:
        """Validate a trained model against its validation dataset in order."""

        results = []
        for validator in ValidatorRunner.instantiate(validators):
            logger.info(
                "Starting validator name=%s type=%s",
                validator.name,
                validator.type,
            )
            result = validator.validate(model, validation_dataset)
            if not isinstance(result, ValidationResult):
                raise TypeError(
                    f"Validator {validator.name!r} must return a ValidationResult."
                )
            if result.name != validator.name or result.type != validator.type:
                raise ValueError(
                    f"Validator {validator.name!r} returned result metadata for "
                    f"{result.name!r} ({result.type!r})."
                )
            results.append(result)
            logger.info(
                "Completed validator name=%s type=%s metrics=%d",
                validator.name,
                validator.type,
                len(result.metrics),
            )
        return results
