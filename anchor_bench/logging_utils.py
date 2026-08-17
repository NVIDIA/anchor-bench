"""Logging configuration and progress-reporting helpers."""

from __future__ import annotations

import logging
import sys
from typing import Optional

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"
_handler: Optional[logging.Handler] = None


def configure_logging(verbosity: int = 0, *, quiet: bool = False) -> None:
    """Configure process logging for one command-line invocation.

    Package modules only emit records; the CLI calls this function to choose
    their presentation. Reconfiguration replaces the handler from a previous
    invocation so repeated in-process calls do not duplicate output or retain a
    stale stderr stream.
    """

    global _handler

    level = logging.ERROR if quiet else logging.DEBUG if verbosity > 0 else logging.INFO
    root_logger = logging.getLogger()
    if _handler is not None:
        root_logger.removeHandler(_handler)
        _handler.close()

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT))
    handler.setLevel(level)
    root_logger.addHandler(handler)
    root_logger.setLevel(level)
    _handler = handler

    if quiet:
        third_party_level = logging.ERROR
    elif verbosity > 0:
        third_party_level = logging.DEBUG
    else:
        third_party_level = logging.WARNING
    logging.getLogger("mlflow").setLevel(third_party_level)
    logging.getLogger("optuna").setLevel(third_party_level)


def should_log_progress(current: int, total: int, *, updates: int = 20) -> bool:
    """Return whether one-based progress should be reported at INFO level."""

    if total < 1:
        raise ValueError("Progress total must be positive.")
    if current < 1 or current > total:
        raise ValueError("Progress current value must be between one and total.")
    if updates < 1:
        raise ValueError("Progress update count must be positive.")

    interval = max(1, (total + updates - 1) // updates)
    return current == 1 or current == total or current % interval == 0
