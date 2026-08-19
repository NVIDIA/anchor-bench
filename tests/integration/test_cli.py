# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration coverage for the Anchor Bench command-line interface."""

from collections.abc import Callable, Sequence

import pytest

from tests.integration.cli_runner import CliResult

pytestmark = pytest.mark.integration


def test_run_cli_invokes_main(
    run_cli: Callable[[Sequence[str]], CliResult],
) -> None:
    result = run_cli(["--help"])

    assert result.exit_code == 0
    assert "usage: anchor-bench" in result.stdout
    assert result.stderr == ""
