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
    settings:
      solver: newton
      integrator: implicitfast
      iterations: 10
      ls_iterations: 4

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

`simulation.solver.type` selects the Newton solver (`warp_mujoco` or `vbd`).
`simulation.solver.settings` is passed to that solver's constructor. A key
that is not a constructor parameter is skipped with a warning and recorded in
`run_summary.yaml` as `"NOT SET: <reason>"`. Solver-specific joint, body, and
shape attributes are read from the USD (for example
`newton:mujoco:solreflimit`); author them in the asset or in an override layer
that sublayers it.

`controller.implementation` selects how candidate gains are applied:

- `native` (default): the solver's joint position drive, with the configured
  effort and velocity limits.
- `external_implicit`: Newton's implicit external PD actuator. The PD law is
  solved against the coupled joint-space response with the gravity and
  Coriolis bias at the current state, and effort is clamped to
  `clamps.effort_limit`. `clamps.velocity_limit` is not applied. Fitted gains
  are exported to the same USD drive attributes; consumers that use a native
  drive may behave slightly differently.

VBD's native joint drive enforces neither `joint_effort_limit` nor
`joint_velocity_limit`. Use `controller.implementation: external_implicit` to
clamp effort on VBD.

`examples/system_identification/so101_vbd.yaml` fits the SO-101 with VBD and
`external_implicit` actuators. It sets `rigid_articulation_solve:
block_sparse_joints`: VBD's default per-body solve diverges on this articulated
arm and ignores revolute armature, which the coupled articulation solve
simulates.

See [Dataset loading](datasets.md) for the Parquet contract and alias behavior.

## Run a custom task

```bash
uv run anchor-bench \
  --config path/to/sysid.yaml \
  --output outputs/my-sysid-run
```

The runtime loads the configured trajectories, evaluates optimizer candidates
in the simulator selected by `simulation.type`, and stops when the optimizer
reaches its iteration or convergence condition.

## Add another simulator

System identification creates the `SimulationContext` registered for
`simulation.type` in `anchor_bench.core.simulation.registry.SIMULATION_CONTEXTS`
and asks it for an evaluator. To support another simulator:

1. Subclass `SimulationContext` and override `create_sysid_evaluator()` to
   return a `PopulationEvaluator` that scores candidates in that simulator.
2. Optionally override `PopulationEvaluator.export_model()` to write the fitted
   model; the default writes nothing.
3. Add the context class to `SIMULATION_CONTEXTS` and its configuration to the
   `simulation` field of `ActuatorSysIdConfig`.

## Artifacts

In addition to the common `config.yaml` and `.completed` files, a successful
run writes:

| Artifact | Contents |
| --- | --- |
| `best_params.yaml` | Best candidate, position MSE, and result schema metadata. |
| `optimization_log.csv` | Best and population-mean MSE for each generation. |
| `run_summary.yaml` | Timestamp, best MSE, metric name, and generation count. |
| `run_summary.yaml` `solver` | Solver type and the settings applied. |
| `identified.usda` | USD layer containing the fitted actuator opinions (Newton). |

When MLOps tracking is enabled, the task also reports per-generation metrics
and uploads the finalized directory as the `sysid-output` artifact collection.
