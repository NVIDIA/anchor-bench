# Anchor Bench

[![Tests](https://github.com/NVIDIA-dev/anchor-bench/actions/workflows/quality.yml/badge.svg?branch=refactor-main)](https://github.com/NVIDIA-dev/anchor-bench/actions/workflows/quality.yml)
[![Coverage](/../badges/badges/coverage.svg)](https://github.com/NVIDIA-dev/anchor-bench/actions/workflows/quality.yml)

Anchor Bench is an early-stage, configuration-driven Python package for robot
actuator tooling.

The currently implemented functionality is limited to:

- parsing and strictly validating `actuator_sys_id` YAML configurations;
- dispatching validated configurations through the task runner;
- loading local or HTTP(S) Parquet datasets and resampling their signals onto a
  uniform time grid; and
- loading a USD model and its initial state through the optional Newton runtime.

The actuator system-identification optimizer and simulator execution pipeline
is not wired in yet. The training module does not contain a training task.
Running a valid `actuator_sys_id` configuration currently validates and
dispatches it, then exits successfully without running optimization or writing
result artifacts.

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

Use `--help` to display the available option:

```bash
uv run anchor-bench --help
```

Only the `actuator_sys_id` configuration type is currently registered. A
minimal complete configuration looks like this:

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
  physics_freq: 1000
  control_freq: 200
  render_freq: 60
  solver:
    type: warp_mujoco
    solver_iterations: 10
    line_search_iterations: 4
dataset:
  type: parquet
  commanded_state: {path: runs/commanded.parquet}
  measured_state: {path: runs/measured.parquet}
```

Configuration objects are strict: unknown fields, coerced values, invalid
bounds, and unsupported `type` values are rejected.

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
