# Actuator system identification

The `actuator_sys_id` task fits actuator and PD-controller parameters by
replaying recorded joint commands across a CMA-ES population of replicated
Newton worlds. The objective minimizes position mean squared error against the
recorded state.

Newton's simulation components are included in the default Anchor Bench
installation.

The Newton dependency is pinned to revision
[`1e5036c48ec92d44782dfe0f448dc25d64984b58`](https://github.com/skulkar22/newton/commit/1e5036c48ec92d44782dfe0f448dc25d64984b58)
of the `dev/rigid-vbd-integration` branch in `skulkar22/newton`. Installing
Anchor Bench resolves that exact revision rather than the moving branch head.
The current sysID configuration continues to use MuJoCo.

## Run the included example

The maintained SO-101 example includes its source USD and three recorded
training trajectories. From the repository root, run:

```bash
uv run anchor-bench \
  --config examples/system_identification/so101.yaml \
  --output outputs/so101_actuator_sysid
```

The configuration evaluates up to 100 CMA-ES generations across 64 replicated
worlds and may stop earlier when it reaches the configured convergence target.

## Configuration

A single-joint configuration has this shape (replace the example paths with
real assets):

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
  usd_path: path/to/robot.usda
  physics_freq: 1000
  control_freq: 200
  render_freq: 60
  solver:
    type: warp_mujoco
    solver_iterations: 10
    line_search_iterations: 4

dataset:
  type: parquet
  source: path/to/recording.parquet
  joint_name_mapping:
    recorded_hip: hip
  resampling:
    type: uniform
    dt: 0.005
```

The recording must provide `<joint>/command_q` and `<joint>/actual_q` fields for
every configured joint and at least two aligned samples per experiment. The
dataset resampling interval must equal `1 / control_freq`; in this example,
both are 0.005 seconds. Joint names must be unique across parameter groups and
must match the canonical names produced by the dataset loader and authored in
the USD.

Optional top-level settings include `num_envs`, `buffer_time`, `seed`,
`output_dir`, descriptive tags, and an [`mlops` block](mlops.md). The CMA-ES
population contains at least two environments; `num_envs` defaults to 64.
`buffer_time` excludes an aligned prefix of each recorded trajectory from the
objective while still replaying its commands, allowing the simulated state to
settle without advancing it ahead of the measurements.

See [Dataset loading](datasets.md) for the Parquet contract and alias behavior.

## Run a custom task

```bash
uv run anchor-bench \
  --config path/to/sysid.yaml \
  --output outputs/my-sysid-run
```

The runtime loads the configured trajectories, evaluates optimizer candidates
with the selected simulator backend, and stops when the optimizer reaches its
iteration or convergence condition.

## Register another simulator

A backend supplies its own strict simulation configuration, resource context,
and population evaluator. Its configuration extends
`anchor_bench.schema.simulation.SysIdSimulationConfig`, retaining the shared
required `usd_path` and timing fields, and declaring its own `type` and simulator
settings. Its context extends `SimulationContext(usd_path, *, fixed_base=False)`.
The shared context resolves and validates the USD path and protects reuse by
backend class, resolved path, and base mode. Its `_load(usd_path)` receives
the resolved USD path and returns the model and initial state.

Register these components through the Python API before validating a task:

```python
from anchor_bench.core.simulation.registry import (
    SimulationBackend,
    SysIdBackend,
    register_simulation_backend,
)

registration = SimulationBackend(
    name="my_simulator",
    context_type=MySimulationContext,
    sysid=SysIdBackend(
        config_model=MySimulationConfig,
        create_context=create_context,
        create_evaluator=create_evaluator,
        export_model=export_model,
    ),
)
register_simulation_backend(registration)
```

The classes and callbacks in this example are supplied by the plugin.
`create_context(config)` returns the registered context type.
`create_evaluator(config, parameter_space, context)` receives that same context
and returns an object implementing `evaluate(population, trajectories)`: one
finite position-MSE score per decoded candidate. The evaluator owns its
simulator's candidate application, resets, command delay, buffer exclusion,
actuation, and stepping. The task retains dataset loading and optimization.

An installed package can expose a zero-argument provider that returns the
`SimulationBackend` registration, using this entry-point group:

```toml
[project.entry-points."anchor_bench.sysid_backends"]
my_simulator = "my_package.backend:provide_backend"
```

The entry-point name, registration name, and configuration `type` must agree.
Discovery runs before direct configuration validation and YAML parsing;
duplicate names and invalid providers fail before optimization. Select the
backend using `simulation.type: my_simulator` and provide its configuration
fields. Plugin fields survive configuration serialization and hashing.

Model export is optional. An exporter receives `(config, best_parameters,
context, output_directory)` and returns an `IdentifiedModelArtifact(path,
format)` or `None`. The result exposes the returned path through
`SysIdResult.identified_model_path`; only an explicit `format="usd"` also
supplies the compatibility `identified_usd_path`. An optional
`describe_runtime(evaluator)` callback returns
JSON-compatible runtime settings for `run_summary.yaml`.

## Artifacts

In addition to the common `config.yaml` and `.completed` files, a successful
run writes:

| Artifact | Contents |
| --- | --- |
| `best_params.yaml` | Best candidate, position MSE, and result schema metadata. |
| `optimization_log.csv` | Best and population-mean MSE for each generation. |
| `run_summary.yaml` | Timestamp, best MSE, metric name, and generation count. |
| `identified.usda` | Newton's USD layer containing the fitted actuator opinions. Other backends can export a different format or omit model export. |

When MLOps tracking is enabled, the task also reports per-generation metrics
and uploads the finalized directory as the `sysid-output` artifact collection.
