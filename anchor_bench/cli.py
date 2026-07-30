"""Command-line interface for Anchor Bench."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Optional

from pydantic import ValidationError

from anchor_bench.core.task_runner import TaskRunner
from anchor_bench.errors import AnchorBenchError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anchor-bench",
        description="Run an Anchor Bench task from a YAML configuration file.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the task's YAML configuration file.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        TaskRunner().run(args.config)
    except (AnchorBenchError, ValidationError) as error:
        parser.error(str(error))

    return 0
