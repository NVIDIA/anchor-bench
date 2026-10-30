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

The maintained SO-101 example includes its source USD and selects three
revision-pinned training trajectories from Anchor-Lab. From the repository
root, run:

```bash
uv run anchor-bench \
  --config examples/system_identification/so101.yaml \
  --output outputs/so101_actuator_sysid
```

The configuration evaluates up to 100 CMA-ES generations across 64 replicated
worlds and may stop earlier when it reaches the configured convergence target.
The first run downloads the trajectories into the Hugging Face cache; later
runs reuse the cached files.

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
in Newton, and stops when the optimizer reaches its iteration or convergence
condition.

## Artifacts

In addition to the common `config.yaml` and `.completed` files, a successful
run writes:

| Artifact | Contents |
| --- | --- |
| `best_params.yaml` | Best candidate, position MSE, and result schema metadata. |
| `optimization_log.csv` | Best and population-mean MSE for each generation. |
| `run_summary.yaml` | Timestamp, best MSE, metric name, and generation count. |
| `identified.usda` | USD layer containing the fitted actuator opinions. |

When MLOps tracking is enabled, the task also reports per-generation metrics
and uploads the finalized directory as the `sysid-output` artifact collection.
