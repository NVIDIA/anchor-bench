# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

import pytest

from anchor_bench import logging_utils
from anchor_bench.logging_utils import configure_logging, should_log_progress


def test_configure_logging_replaces_its_previous_handler() -> None:
    root_logger = logging.getLogger()
    third_party_loggers = [logging.getLogger("mlflow"), logging.getLogger("optuna")]
    previous_handlers = list(root_logger.handlers)
    previous_level = root_logger.level
    previous_third_party_levels = [logger.level for logger in third_party_loggers]
    previous_handler = logging_utils._handler
    try:
        logging_utils._handler = None

        configure_logging()
        first_handler = logging_utils._handler
        configure_logging(verbosity=1)

        assert first_handler not in root_logger.handlers
        assert logging_utils._handler in root_logger.handlers
        assert root_logger.level == logging.DEBUG
        assert logging_utils._handler is not None
        assert logging_utils._handler.level == logging.DEBUG
        assert all(logger.level == logging.DEBUG for logger in third_party_loggers)

        configure_logging(quiet=True)

        assert root_logger.level == logging.ERROR
        assert logging_utils._handler is not None
        assert logging_utils._handler.level == logging.ERROR
        assert all(logger.level == logging.ERROR for logger in third_party_loggers)
    finally:
        if logging_utils._handler is not None:
            root_logger.removeHandler(logging_utils._handler)
        root_logger.handlers = previous_handlers
        root_logger.setLevel(previous_level)
        for logger, level in zip(third_party_loggers, previous_third_party_levels):
            logger.setLevel(level)
        logging_utils._handler = previous_handler


@pytest.mark.parametrize(
    ("current", "expected"),
    [(1, True), (2, False), (5, True), (99, False), (100, True)],
)
def test_progress_is_rate_limited(current: int, expected: bool) -> None:
    assert should_log_progress(current, 100) is expected


@pytest.mark.parametrize(
    ("current", "total", "updates"),
    [(0, 10, 20), (11, 10, 20), (1, 0, 20), (1, 10, 0)],
)
def test_progress_rejects_invalid_values(
    current: int,
    total: int,
    updates: int,
) -> None:
    with pytest.raises(ValueError):
        should_log_progress(current, total, updates=updates)
