"""Helpers for invoking the Anchor Bench CLI from integration tests."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pytest

from anchor_bench.cli import main


@dataclass(frozen=True)
class CliResult:
    """The observable result of one in-process CLI invocation."""

    exit_code: int
    stdout: str
    stderr: str


def invoke_cli(
    argv: Sequence[str],
    capsys: pytest.CaptureFixture[str],
) -> CliResult:
    """Call ``anchor_bench.cli.main`` and capture its user-visible result."""

    try:
        exit_code = main(argv)
    except SystemExit as error:
        exit_code = error.code if isinstance(error.code, int) else 1

    captured = capsys.readouterr()
    return CliResult(
        exit_code=exit_code,
        stdout=captured.out,
        stderr=captured.err,
    )
