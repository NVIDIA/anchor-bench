# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command-line interface for Anchor Bench."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Optional

from pydantic import ValidationError

from anchor_bench.core.task_runner import TaskRunner
from anchor_bench.errors import AnchorBenchError
from anchor_bench.logging_utils import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anchor-bench",
        description="Run an Anchor Bench task from a YAML configuration file.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=["mpm_sys_id"],
        help="Optional explicit task subcommand.",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume an interrupted MPM search."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the task's YAML configuration file.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run the task even when its output is already complete.",
    )
    parser.add_argument(
        "--output",
        help="Use this folder for task artifacts instead of a generated hash folder.",
    )
    logging_group = parser.add_mutually_exclusive_group()
    logging_group.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Enable debug logging. Repeat for compatibility with common CLI usage.",
    )
    logging_group.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Only emit errors.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose, quiet=args.quiet)

    try:
        options = {}
        if args.command:
            options["expected_type"] = args.command
        if args.resume:
            options["resume"] = True
        TaskRunner().run(args.config, force=args.force, output=args.output, **options)
    except (AnchorBenchError, ValidationError) as error:
        parser.error(str(error))

    except KeyboardInterrupt:
        return 130

    return 0
