"""Argument parser for the authoritative phase-oriented CLI."""

from __future__ import annotations

import argparse
import json
from typing import Any


def _json_value(value: str) -> Any:
    """Parse one JSON command-line value with an argparse-friendly error."""

    try:
        return json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(f"invalid JSON: {error.msg}") from error


def _json_mapping(value: str) -> dict[str, Any]:
    """Parse a JSON object used as one complete typed-field replacement."""

    parsed = _json_value(value)
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("value must be a JSON object")
    return parsed


def _json_list(value: str) -> list[Any]:
    """Parse a JSON array used as one complete structural replacement."""

    parsed = _json_value(value)
    if not isinstance(parsed, list):
        raise argparse.ArgumentTypeError("value must be a JSON array")
    return parsed


def _csv_names(value: str) -> tuple[str, ...]:
    """Parse a non-empty comma-separated ordered name list."""

    names = tuple(item.strip() for item in value.split(",") if item.strip())
    if not names:
        raise argparse.ArgumentTypeError("value must contain at least one name")
    return names


def _joint_selector(value: str) -> str | tuple[str, ...]:
    """Parse either the literal ``all`` or an ordered joint-name list."""

    return "all" if value.strip() == "all" else _csv_names(value)


def _add_common_run_options(parser: argparse.ArgumentParser) -> None:
    """Add profile, replicate, and dry-run options shared by run phases."""

    parser.add_argument("--profile", default="default")
    parser.add_argument("--replicate")
    parser.add_argument("--dry-run", action="store_true")


def _add_robots_parser(subparsers: Any) -> None:
    """Add robot identity discovery commands."""

    robots = subparsers.add_parser(
        "robots",
        help="Inspect declarative robot identities.",
    )
    robot_sub = robots.add_subparsers(dest="robot_command", required=True)
    robot_sub.add_parser("list", help="List configured robots.")
    describe = robot_sub.add_parser("describe", help="Show one robot identity.")
    describe.add_argument("robot")


def _add_sysid_parser(subparsers: Any) -> None:
    """Add immutable SysID execution arguments."""

    parser = subparsers.add_parser(
        "sysid",
        help="Create or preview one immutable SysID candidate.",
    )
    parser.add_argument("--robot", required=True)
    _add_common_run_options(parser)
    parser.add_argument("--initial-state", default=None)
    parser.add_argument("--data")
    parser.add_argument("--joints", type=_csv_names)
    parser.add_argument("--simulation", type=_json_mapping)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--buffer-time", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--optimizer", type=_json_mapping)
    parser.add_argument("--parameters", type=_json_mapping)
    parser.add_argument("--loss", type=_json_mapping)


def _add_training_parser(subparsers: Any) -> None:
    """Add immutable actuator-training arguments."""

    parser = subparsers.add_parser(
        "train-actuator",
        help="Create or preview one immutable GRU candidate.",
    )
    parser.add_argument("--robot", required=True)
    _add_common_run_options(parser)
    parser.add_argument("--sysid-state")
    parser.add_argument("--data")
    parser.add_argument("--sample-dt-s", type=float)
    mapping = parser.add_mutually_exclusive_group()
    mapping.add_argument("--mappings", type=_json_list)
    mapping.add_argument("--input-joints", type=_joint_selector)
    parser.add_argument("--output-joints", type=_joint_selector)
    parser.add_argument("--network", type=_json_mapping)
    parser.add_argument("--target", choices=("torque", "torque_residual"))
    parser.add_argument("--input-features", type=_json_list)
    parser.add_argument("--fit", type=_json_mapping)
    parser.add_argument(
        "--skip-optuna",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--trials", type=int)
    parser.add_argument("--trial-epochs", type=int)
    parser.add_argument("--final-epochs", type=int)
    parser.add_argument("--noise-std", type=float)
    parser.add_argument("--seed", type=int)


def _add_benchmark_parser(subparsers: Any) -> None:
    """Add immutable candidate-benchmark arguments."""

    parser = subparsers.add_parser(
        "benchmark",
        help="Replay one exact SysID or training candidate.",
    )
    parser.add_argument("--robot", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--profile", default="default")
    parser.add_argument("--data")
    parser.add_argument("--motion", action="append", dest="motions")
    parser.add_argument("--simulation", type=_json_mapping)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--buffer-time", type=float)
    parser.add_argument(
        "--sync-initial-state",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--signals", type=_csv_names)
    parser.add_argument("--output-root")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--dry-run", action="store_true")


def _add_promotion_parser(subparsers: Any) -> None:
    """Add explicit, non-executing promotion commands."""

    promote = subparsers.add_parser(
        "promote",
        help="Explicitly update accepted phase state.",
    )
    phases = promote.add_subparsers(dest="promote_phase", required=True)
    for phase in ("sysid", "train"):
        parser = phases.add_parser(
            phase,
            help=f"Promote one immutable {phase} candidate.",
        )
        parser.add_argument("--robot", required=True)
        parser.add_argument("--run", required=True)
        parser.add_argument("--benchmark-run", action="append", default=[])
        parser.add_argument("--dry-run", action="store_true")
    deploy = phases.add_parser(
        "deploy",
        help="Freeze a deployment profile from accepted sources.",
    )
    deploy.add_argument("--robot", required=True)
    deploy.add_argument("--profile", default="default")
    deploy.add_argument("--dry-run", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    """Build the complete public command tree."""

    parser = argparse.ArgumentParser(prog="anchor-bench")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_robots_parser(subparsers)
    _add_sysid_parser(subparsers)
    _add_training_parser(subparsers)
    _add_benchmark_parser(subparsers)
    _add_promotion_parser(subparsers)
    return parser


__all__ = ["build_parser"]
