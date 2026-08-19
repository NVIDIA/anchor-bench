# Repository agent instructions

## Licensing header

Every newly created code file must begin with the repository's licensing
header, using the current calendar year and the comment syntax appropriate for
the file's language:

```text
SPDX-FileCopyrightText: Copyright (c) <current year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
```

For every existing code file touched by a change, verify that its licensing
header uses the current calendar year and update the year when necessary.
Adding or updating the header is a mandatory step before committing or handing
off the change.

## Pre-commit gate

Run this gate only when a change modifies Python files or project libraries and
dependencies, including `pyproject.toml` or `uv.lock`. Do not run it for
documentation-only, workflow-only, or unrelated configuration changes.

Before committing or handing off an applicable change, agents must run all of
the following checks from the repository root:

```bash
uv run ruff check
uv run ruff format --check
uv run pyright
uv run pytest <targeted-test-paths>
```

Choose the targeted test paths that exercise the behavior affected by the
change. Include directly modified tests and tests for any source modules whose
behavior changed. For example:

```bash
uv run pytest tests/test_dataset_loader.py tests/test_datasets.py
```

The default pytest configuration excludes tests marked `integration`. If a
change affects an integration workflow, run its relevant integration tests
explicitly as an additional gate:

```bash
uv run pytest -m integration tests/integration/test_dataset_loader_huggingface.py
```

When a change is broad, affects shared infrastructure, or has no clearly
isolated test coverage, run the complete unit test suite with
`uv run pytest`. Run the complete unit and integration suite with
`uv run pytest -o "addopts="` when integration behavior may also be affected.

All applicable gate commands must pass. Do not skip a check or suppress a
failure without documenting the reason in the handoff.

### Fixing gate failures

- For `ruff check` failures, first apply safe automatic fixes with
  `uv run ruff check --fix <affected-python-paths>`, fix any remaining
  violations manually, and rerun `uv run ruff check`.
- For `ruff format --check` failures, format the affected paths with
  `uv run ruff format <affected-python-paths>`, then rerun
  `uv run ruff format --check`.
- For `pyright` failures, correct the implementation or type annotations and
  rerun `uv run pyright`. Do not add an ignore solely to make the gate pass;
  use one only when the reported issue is intentionally safe and document why.
- For pytest failures, diagnose whether the implementation or the test is
  incorrect, make the scoped correction, and rerun the same targeted test
  command. Do not weaken assertions merely to make a failure pass.
- If a library or dependency change leaves `uv.lock` stale, run `uv lock` and
  repeat the applicable gate commands.

Only fix failures related to the current change. If a gate exposes a pre-existing
or unrelated failure, leave unrelated code untouched and report the command and
failure in the handoff.
