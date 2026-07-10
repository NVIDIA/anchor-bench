"""Host-safe public APIs for immutable GRU training and promotion."""

from anchor_bench.training.promotion import TrainingPromotionOutcome, promote_training_run
from anchor_bench.training.runner import (
    ResolvedTrainingRun,
    TrainingInvocation,
    TrainingResolverInputs,
    TrainingRunOutcome,
    execute_training_run,
    resolve_training_run,
)

__all__ = [
    "ResolvedTrainingRun",
    "TrainingInvocation",
    "TrainingPromotionOutcome",
    "TrainingResolverInputs",
    "TrainingRunOutcome",
    "execute_training_run",
    "promote_training_run",
    "resolve_training_run",
]
