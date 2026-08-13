"""Domain-specific errors raised by the task runner."""


class AnchorBenchError(Exception):
    """Base class for expected, user-facing Anchor Bench errors."""


class ConfigParseError(AnchorBenchError):
    """Raised when a configuration file cannot be loaded."""


class UnknownConfigTypeError(ConfigParseError):
    """Raised when no Pydantic model is registered for a config type."""


class TaskNotFoundError(AnchorBenchError):
    """Raised when no task implementation is registered for a config type."""


class DatasetLoadError(AnchorBenchError):
    """Raised when a dataset source cannot be loaded."""


class DatasetAlignmentError(AnchorBenchError):
    """Raised when dataset signals cannot be aligned."""


class MLOpsError(AnchorBenchError):
    """Base class for MLOps backend and tracking failures."""


class MLOpsBackendNotFoundError(MLOpsError):
    """Raised when an MLOps backend is not registered."""


class MLOpsTrackingError(MLOpsError):
    """Raised when an MLOps backend cannot create or update a tracked run."""
