# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render a small, dependency-free SVG badge from a literal or JSON value."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

ColorThreshold = tuple[float, str]


def _json_value(path: Path, dotted_key: str) -> str:
    value: Any = json.loads(path.read_text(encoding="utf-8"))
    for key in dotted_key.split("."):
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"JSON path not found: {dotted_key}")
        value = value[key]
    return str(value)


def _segment_width(text: str) -> int:
    """Estimate a Shields-style text segment width in pixels."""
    return max(40, len(text) * 7 + 10)


def _color_threshold(value: str) -> ColorThreshold:
    try:
        minimum, color = value.split(":", maxsplit=1)
        if not color:
            raise ValueError
        return float(minimum), color
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "color thresholds must use the format MINIMUM:COLOR"
        ) from error


def _select_color(value: str, thresholds: list[ColorThreshold], fallback: str) -> str:
    if not thresholds:
        return fallback

    try:
        numeric_value = float(value)
    except ValueError as error:
        raise ValueError("color thresholds require a numeric badge value") from error

    for minimum, color in sorted(thresholds, reverse=True):
        if numeric_value >= minimum:
            return color
    return fallback


def render_badge(label: str, value: str, color: str) -> str:
    label_width = _segment_width(label)
    value_width = _segment_width(value)
    title = html.escape(f"{label}: {value}")
    label = html.escape(label)
    value = html.escape(value)
    total_width = label_width + value_width
    label_center = label_width // 2
    value_center = label_width + value_width // 2

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{total_width}" height="20" role="img" aria-label="{title}">
  <title>{title}</title>
  <linearGradient id="s" x2="0" y2="100%">
    <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
    <stop offset="1" stop-opacity=".1"/>
  </linearGradient>
  <clipPath id="r">
    <rect width="{total_width}" height="20" rx="3" fill="#fff"/>
  </clipPath>
  <g clip-path="url(#r)">
    <rect width="{label_width}" height="20" fill="#555"/>
    <rect x="{label_width}" width="{value_width}" height="20" fill="{html.escape(color)}"/>
    <rect width="{total_width}" height="20" fill="url(#s)"/>
  </g>
  <g fill="#fff" text-anchor="middle" font-family="Verdana,Geneva,DejaVu Sans,sans-serif" font-size="11">
    <text x="{label_center}" y="15" fill="#010101" fill-opacity=".3">{label}</text>
    <text x="{label_center}" y="14">{label}</text>
    <text x="{value_center}" y="15" fill="#010101" fill-opacity=".3">{value}</text>
    <text x="{value_center}" y="14">{value}</text>
  </g>
</svg>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    value_source = parser.add_mutually_exclusive_group(required=True)
    value_source.add_argument("--value")
    value_source.add_argument("--json-file", type=Path)
    parser.add_argument("--json-path")
    parser.add_argument("--suffix", default="")
    parser.add_argument("--color", default="#007ec6")
    parser.add_argument(
        "--color-threshold",
        action="append",
        default=[],
        type=_color_threshold,
        metavar="MINIMUM:COLOR",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if args.json_file:
        if not args.json_path:
            parser.error("--json-path is required with --json-file")
        value = _json_value(args.json_file, args.json_path)
    else:
        value = args.value

    try:
        color = _select_color(value, args.color_threshold, args.color)
    except ValueError as error:
        parser.error(str(error))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        render_badge(args.label, f"{value}{args.suffix}", color),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
