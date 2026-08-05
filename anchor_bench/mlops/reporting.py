"""Active-run binding and portable parameter preparation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Generator, Optional

from anchor_bench.mlops.base import MLOpsRun, ParameterValue

_ACTIVE_RUN: ContextVar[Optional[MLOpsRun]] = ContextVar(
    "anchor_bench_mlops_run",
    default=None,
)


def get_active_run() -> Optional[MLOpsRun]:
    """Return the run bound to the current task execution, if any."""

    return _ACTIVE_RUN.get()


@contextmanager
def bind_active_run(run: MLOpsRun) -> Generator[MLOpsRun, None, None]:
    """Bind a run for task-local reporting and restore the prior binding."""

    token = _ACTIVE_RUN.set(run)
    try:
        yield run
    finally:
        _ACTIVE_RUN.reset(token)


def flatten_params(
    values: Mapping[str, Any],
    *,
    separator: str = "/",
) -> Dict[str, ParameterValue]:
    """Flatten JSON-like configuration values into portable run parameters."""

    flattened: Dict[str, ParameterValue] = {}
    for key, value in values.items():
        _flatten_value(
            value,
            prefix=str(key),
            separator=separator,
            destination=flattened,
        )
    return flattened


def _flatten_value(
    value: Any,
    *,
    prefix: str,
    separator: str,
    destination: Dict[str, ParameterValue],
) -> None:
    if isinstance(value, Mapping):
        if not value:
            destination[prefix] = "{}"
            return
        for key, nested_value in value.items():
            _flatten_value(
                nested_value,
                prefix=f"{prefix}{separator}{key}",
                separator=separator,
                destination=destination,
            )
        return

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if not value:
            destination[prefix] = "[]"
            return
        for index, nested_value in enumerate(value):
            _flatten_value(
                nested_value,
                prefix=f"{prefix}{separator}{index}",
                separator=separator,
                destination=destination,
            )
        return

    if value is None:
        destination[prefix] = "null"
    elif isinstance(value, (str, int, float, bool)):
        destination[prefix] = value
    else:
        destination[prefix] = str(value)
