# Anchor Bench

[![Tests](https://github.com/NVIDIA-dev/anchor-bench/actions/workflows/quality.yml/badge.svg?branch=refactor-main)](https://github.com/NVIDIA-dev/anchor-bench/actions/workflows/quality.yml)
[![Coverage](/../badges/badges/coverage.svg)](https://github.com/NVIDIA-dev/anchor-bench/actions/workflows/quality.yml)

Anchor Bench is an early-stage, configuration-driven Python package for robot
actuator tooling.

The currently implemented functionality is limited to:

- parsing and strictly validating `actuator_sys_id`, `trajectory_v1`, and
  `training` YAML configurations;
- dispatching validated configurations through the task runner;
- generating piecewise actuator command trajectories as Parquet datasets;
- loading local or HTTP(S) Parquet datasets and resampling their signals onto a
  uniform time grid;
- preparing mapped training trajectories as Parquet artifacts;
- loading a USD model and its initial state through the optional Newton runtime;
  and
- tracking portable run parameters, metrics, tags, and artifacts through
  bundled MLflow or W&B plugins.

The actuator system-identification pipeline fits actuator parameters by
evaluating CMA-ES populations in replicated Newton worlds and writes the best
parameters and optimization history to the task output directory.

## Installation

Anchor Bench requires Python 3.10 or newer because Newton requires Python
3.10+. Python 3.12 is recommended because it matches Isaac Lab's recommended
Python version. The repository's `.python-version` configures uv to use Python
3.12 by default.

Anchor Bench uses [uv](https://docs.astral.sh/uv/) for environment and
dependency management. From the repository root, create the environment and
install the package:

```bash
uv sync
```

Install the optional Newton dependency to use `NewtonSimulationContext`:

```bash
uv sync --extra simulation
```

For development, install the test and quality-check dependencies:

```bash
uv sync --extra dev
```

Install only the tracking SDK you use. For the default MLflow backend, run:

```bash
uv sync --extra mlflow
```

Or, for the W&B plugin, run:

```bash
uv sync --extra wandb
```

The tracking SDKs are loaded lazily and are not needed for tasks that do not
use MLOps tracking. See the [MLOps backend guide](docs/mlops.md) for the shared
API, opt-in task-run lifecycle, task configuration, and plugin contract.

## Command-line usage

The command-line interface accepts one YAML configuration file:

```bash
uv run anchor-bench --config path/to/config.yaml
```

Successful tasks write a `.completed` marker in their output directory. A
later invocation with the same configuration skips the task while that marker
is present. Pass `--force` to run it again:

```bash
uv run anchor-bench --config path/to/config.yaml --force
```

By default, task artifacts are written to a generated hash folder. Pass
`--output` to use a specific folder instead:

```bash
uv run anchor-bench --config path/to/config.yaml --output path/to/artifacts
```

Use `--help` to display the available option:

```bash
uv run anchor-bench --help
```

The `actuator_sys_id`, `trajectory_v1`, and `training` configuration types are
registered. A
minimal actuator configuration looks like this:

```yaml
type: actuator_sys_id
joint_parameters:
  - names: [hip]
    controller:
      type: pd
      clamps:
        effort_limit: 120.0
        velocity_limit: 30.0
      stiffness: {lower: 1.0, upper: 100.0, initial: 20.0}
      damping: {lower: 0.0, upper: 10.0}
    parameters:
      dry_friction: {lower: 0.0, upper: 1.0}
      viscous_friction: {lower: 0.0, upper: 1.0}
      armature: {lower: 0.0, upper: 0.1}
optimizer:
  type: python-cmaes
  sigma: 0.5
  max_iterations: 100
  epsilon: 0.001
simulation:
  type: newton
  usd_path: assets/robots/test.usd
  physics_freq: 1000
  control_freq: 200
  render_freq: 60
  solver:
    type: warp_mujoco
    solver_iterations: 10
    line_search_iterations: 4
dataset:
  type: parquet
  source: runs/actuator.parquet
  resampling:
    type: uniform
    dt: 0.005
```

Configuration objects are strict: unknown fields, coerced values, invalid
bounds, and unsupported `type` values are rejected.

## Trajectory generation

The `trajectory_v1` task generates ordered hold, transition, sine, and linear
chirp pieces. The supplied actuator order defines every pose array and the
output field order. For example:

```bash
uv run anchor-bench \
  --config examples/so101_motion_pack.yaml \
  --output outputs/so101_motion_pack
```

The task writes `trajectories.parquet` in the long-form actuator dataset shape:
`time_ns`, `time_utc`, `experiment`, `field`, and `value`. It contains only
`<actuator>/command_q` fields. Synthetic time begins at the Unix epoch for each
trajectory, making both timestamp columns deterministic. Piece durations are
converted to sample counts with Python's round-to-nearest, ties-to-even
behavior, and transition endpoints are excluded so the next piece begins at
the exact target pose. A nonzero `ramp_s` applies a linear fade-in and fade-out
envelope to sine and chirp offsets.

## Dataset loading

`DatasetLoader` reads Parquet data from a local path or an HTTP(S) URL and
linearly resamples each experiment over the time range shared by all its
signals:

```python
from anchor_bench.data import DatasetLoader

dataset = DatasetLoader.load("recording.parquet", dt=0.02)
```

Input data must contain `time_ns`, `experiment`, `field`, and `value` columns.
The returned data frame uses `(experiment, time_ns)` as its index and one column
per signal field.

The training task maps those signal columns into recurrent input and target
trajectories and writes them to `trajectories.parquet` in its output directory.
The artifact is long-form, with `sequence`, `mapping`, `kind`, `experiment`,
`time_ns`, `feature`, and `value` columns.

## Newton model loading

After installing the `simulation` extra, a USD model can be loaded through the
Newton adapter:

```python
from anchor_bench.core import NewtonSimulationContext

model, state = NewtonSimulationContext().load("robot.usda")
```

This adapter only constructs the Newton model and initial state; it does not run
a simulation.

## Testing and quality checks

Run the unit test suite with coverage:

```bash
uv run pytest --cov=anchor_bench --cov-report=term-missing
```

Tests marked `integration` are excluded by default because they download an
external dataset and exercise the dataset-loading workflow. Run them explicitly
with network access:

```bash
uv run pytest -m integration
```

Run the same static checks used by CI:

```bash
uv run ruff check anchor_bench tests
uv run ruff format --check anchor_bench tests
uv run pyright
```

## Documentation status

The guides in [`docs/`](docs/) describe an earlier architecture and are
outdated. Do not use them as instructions for the current package. Until they
are rewritten, this README, the source code, and the tests are the sources of
truth for implemented behavior.
