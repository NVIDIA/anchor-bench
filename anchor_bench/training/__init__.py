
"""Neural actuator training workflows.

The package keeps heavyweight ML imports inside execution functions so CLI
dry-runs and schema validation remain host-safe.
"""

from anchor_bench.training.runner import (
    ResolvedTrainActuatorRunConfig,
    build_train_actuator_command,
    resolve_train_actuator_run_config,
    run_train_actuator,
)

__all__ = [
    "ResolvedTrainActuatorRunConfig",
    "build_train_actuator_command",
    "resolve_train_actuator_run_config",
    "run_train_actuator",
]
