# Anchor Bench

[![Tests](https://github.com/NVIDIA-dev/anchor-bench/actions/workflows/quality.yml/badge.svg?branch=refactor-main)](https://github.com/NVIDIA-dev/anchor-bench/actions/workflows/quality.yml)
[![Coverage](/../badges/badges/coverage.svg)](https://github.com/NVIDIA-dev/anchor-bench/actions/workflows/quality.yml)

Anchor Bench is an early-stage, configuration-driven toolkit for robot actuator
workflows. It validates a YAML task, runs the corresponding implementation, and
writes reproducible artifacts to a dedicated output directory.

The current task types are:

| Task type | Purpose |
| --- | --- |
| `trajectory_v1` | Generate piecewise actuator command trajectories. |
| `actuator_sys_id` | Fit actuator parameters by replaying recorded data in replicated Newton worlds. |
| `training` | Prepare mapped trajectories and train recurrent actuator models. |

Anchor Bench also provides a Parquet dataset loader, a Newton USD adapter, and
optional run tracking through the bundled MLflow backend.

## Quick start

Anchor Bench requires Python 3.10 or newer and uses
[uv](https://docs.astral.sh/uv/) for its environment and dependencies:

```bash
uv sync
```

Run the included trajectory example:

```bash
uv run anchor-bench \
  --config examples/so101_motion_pack.yaml \
  --output outputs/so101_motion_pack
```

The command validates the configuration and writes `config.yaml`,
`trajectories.parquet`, and a `.completed` marker under the selected output
directory. See [Running tasks](docs/running-tasks.md) for output selection,
rerunning completed tasks, configuration behavior, and logging controls.

Task lifecycle and rate-limited progress logs are written to standard error.
Pass `--verbose` (`-v`) for debug output or `--quiet` (`-q`) for errors only.

Install the simulation components for workflows that import or run Newton
models:

```bash
uv sync --extra simulation
```

Install contributor tools with:

```bash
uv sync --extra dev
```

## Documentation

The [documentation index](docs/README.md) organizes the supported guides by
workflow. Good starting points are:

- [Trajectory generation](docs/trajectory.md)
- [Actuator system identification](docs/system-identification.md)
- [Training actuator models](docs/training.md)
- [Dataset loading](docs/datasets.md)
- [MLOps backends](docs/mlops.md)

## Python API

Load and uniformly resample a long-form Parquet dataset:

```python
from anchor_bench.data import DatasetLoader

dataset = DatasetLoader.load("recording.parquet", dt=0.02)
```

After installing the `simulation` extra, load a USD model and its initial state:

```python
from anchor_bench.core import NewtonSimulationContext

context = NewtonSimulationContext("robot.usda")
model, state = context.load()

context.capture_step_graph(dt=0.01, substeps=2)
context.step(dt=0.01, substeps=2)
```

`capture_step_graph` captures a reusable Warp graph for the requested interval
when the solver supports capture. `step` then advances every world with that
graph when its `dt` and `substeps` match, or executes the interval eagerly.

The `state` property is the canonical input and current-result buffer. The
`state_next` property is the stable output buffer used while computing a step;
after each complete interval, the current result is available through `state`.

`NewtonSimulationContext.load()` constructs the model and state; it does not
run a simulation.

## Development

Run the default unit suite and static checks from the repository root:

```bash
uv run pytest
uv run ruff check
uv run ruff format --check
uv run pyright
```

Tests marked `integration` are excluded by default. Run them explicitly when
the required external data and runtime are available:

```bash
uv run pytest -m integration
```
