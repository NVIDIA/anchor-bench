# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validation for the user-facing task configurations."""

from pathlib import Path

import pytest

from anchor_bench.schema.base import BaseConfig
from anchor_bench.schema.parse import parse

REPOSITORY = Path(__file__).parents[1]
EXAMPLES_DIRECTORY = REPOSITORY / "examples"
EXAMPLE_CONFIGS = sorted(EXAMPLES_DIRECTORY.rglob("*.yaml"))


@pytest.mark.parametrize(
    "config_path",
    EXAMPLE_CONFIGS,
    ids=lambda path: str(path.relative_to(EXAMPLES_DIRECTORY)),
)
def test_example_config_parses(config_path: Path) -> None:
    """Keep every distributed example compatible with its registered schema."""

    assert isinstance(parse(config_path), BaseConfig)
