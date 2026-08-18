# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared pytest fixtures for integration tests."""

from collections.abc import Callable, Sequence
from functools import partial

import pytest

from tests.integration.cli_runner import CliResult, invoke_cli


@pytest.fixture
def run_cli(
    capsys: pytest.CaptureFixture[str],
) -> Callable[[Sequence[str]], CliResult]:
    """Return a helper that invokes the CLI with caller-provided arguments."""

    return partial(invoke_cli, capsys=capsys)
