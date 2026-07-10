from __future__ import annotations

import sys

from anchor_bench.cli.commands import dispatch
from anchor_bench.cli.parser import build_parser
from anchor_bench.runtime.environment import RuntimeUnavailableError


def main(argv: list[str] | None = None) -> int:
    """Parse one command and convert expected user errors into exit status 2."""

    args = build_parser().parse_args(argv)
    try:
        return dispatch(args)
    except (
        RuntimeUnavailableError,
        KeyError,
        ValueError,
        FileNotFoundError,
        RuntimeError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
