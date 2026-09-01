# Contributing to Anchor Bench

Thank you for contributing to Anchor Bench. Run all commands from the
repository root.

## Development setup

Install the project and its development dependencies with
[uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev
```

## Validate changes

For Python, library, or dependency changes, run the static checks and the tests
that cover the affected behavior:

```bash
uv run ruff check
uv run ruff format --check
uv run pyright
uv run pytest <targeted-test-paths>
```

The default pytest configuration excludes integration tests. If a change
affects an integration workflow, run the relevant tests explicitly:

```bash
uv run pytest -m integration <targeted-test-paths>
```

## Sign off commits

All commits must include a
[Developer Certificate of Origin](https://developercertificate.org/) sign-off.
The sign-off certifies that you have the right to submit the contribution under
the project's license. Add it with Git's `--signoff` (or `-s`) option:

```bash
git commit --signoff -m "Describe the change"
```

Git appends a trailer in this form:

```text
Signed-off-by: Your Name <your@email.com>
```
