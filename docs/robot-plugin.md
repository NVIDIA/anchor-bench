# Adding a Robot

Anchor Bench robot support starts as declarative configuration. A new robot
should be understandable from `configs/robots/<robot>/`, with measured run data
under `inputs/`, assets under `assets/` or `ANCHOR_BENCH_ASSET_DIR`, and generated
results under `outputs/`.

Use SO-101 as the reference layout while keeping robot-specific runtime code
isolated from host-safe command planning.

## Minimum File Set

Create these files for every robot:

```text
configs/
  robots/
    <robot>/
      robot.yaml        # identity, asset manifest path, runtime joint order
      actuators.yaml    # canonical/default actuator model
      bounds.yaml       # SysID parameter search bounds
      run.yaml          # default SysID, benchmark, and analysis settings
assets/
  robots/
    <robot>/
      manifest.yaml     # expected asset names and source notes
      files/            # optional repo-local USD/MJCF/meshes for small assets
inputs/
  datasets/
    <robot>/
      <dataset>/
        <motion>/
          control.csv
          state_motor.csv
          joint_list.txt
          metadata.yaml
```

Optional curated input parameter files can live under:

```text
inputs/
  actuator_params/
    <robot>/
      fitted_seed42.yaml
```

Only put actuator params there when they are deliberate inputs for replay or
publication. Fresh SysID results belong in `outputs/sysid/...` until a fitted
file is intentionally promoted.

## Config Files

### `robot.yaml`

This is the robot registry source of truth. It must include:

```yaml
name: my_robot
display_name: My Robot
fixed_base: true
asset_path: assets/robots/my_robot/manifest.yaml
joint_names: [joint_1, joint_2]
joint_aliases:
  shoulder: joint_1
actuator_file: actuators.yaml
bounds_file: bounds.yaml
ee_frame: tool_link
```

`joint_names` is the runtime order. Data loading, actuator parameter maps,
SysID population mapping, benchmark writing, and analysis should all preserve
this order. Keep aliases as convenience names only; actuator maps and datasets
should use the canonical names.

Do not add separate joint-list files for the same robot. `get_robot()` validates
that actuator joints and SysID bounds refer to `robot.yaml` joint names or
aliases, so keeping joint names in one file prevents drift.

If a robot needs named joint groups later, add an explicit schema field rather
than reintroducing another copy of the base joint order.

`anchor-bench robots describe <robot>` validates the robot bundle returned by
`get_robot()`.

### `actuators.yaml`

Use the canonical schema-v2 actuator document. Do not add unsupported `model_type` keys.

```yaml
schema_version: 2
backend: newton
actuators:
  - name: my_robot_all
    joints: [joint_1, joint_2]
    controller:
      type: pd
      source: "baseline controller values"
      parameters:
        stiffness:
          joint_1: 40.0
          joint_2: 30.0
        damping:
          joint_1: 2.0
          joint_2: 1.5
    delay:
      duration_ms: 0.0
    clamps:
      - type: effort
        parameters:
          effort_limit: 3.0
      - type: velocity
        parameters:
          velocity_limit: 20.0
    physical_parameters:
      armature:
        joint_1: 0.01
        joint_2: 0.01
      dry_friction:
        joint_1: 0.1
        joint_2: 0.1
      viscous_friction:
        joint_1: 0.1
        joint_2: 0.1
```

Every per-joint parameter map used by a runtime path should contain every
configured joint. If a robot has multiple actuator groups, keep the groups
explicit and ensure the runtime for that robot knows how to apply them.

### `bounds.yaml`

Bounds define what SysID may tune. Keep fixed values outside `parameters`:

```yaml
schema_version: 2
mirror: false
joint_types: [joint_1, joint_2]
parameters:
  armature:
    lower: 0.0
    upper: 0.1
  dry_friction:
    lower: 0.0
    upper: 1.0
  viscous_friction:
    lower: 0.0
    upper: 2.0
  stiffness:
    lower: 1.0
    upper: 80.0
  damping:
    lower: 0.01
    upper: 5.0
fixed:
  motor_lag_ms: 0.0
```

Actual SysID runs require `cmaes`. Host-safe tests may opt into deterministic
random fallback, but production execution should fail early if `cmaes` is not
available. Configure CMA-ES runtime policy in `run.yaml`, not `bounds.yaml`.

### `run.yaml`

Use this for default workflow settings, not measured data:

```yaml
simulation:
  physics_freq: 500
  control_freq: 500
  render_freq: 500
  integrator: implicitfast
  solver_iterations: 10
  line_search_iterations: 10
benchmark:
  fix_root: true
  buffer_time: 2.0
  motion_files: inputs/datasets/my_robot/default/motion_001
  motion_name: my_robot_motion_001
  output_folder: outputs/sim2real_benchmark
analysis:
  output_dir: outputs/sim2real_analysis
  sample_dt: 0.005
  motion_names: "*"
sysid:
  output_dir: outputs/sysid/my_robot/full
  real_data_dir: inputs/datasets/my_robot/default/motion_001
  joints: full
  num_envs: 64
  buffer_time: 2.0
  cmaes:
    sigma: 0.5
    max_iterations: 100
    epsilon: 0.001
training:
  model: gru
  real_data_dir: inputs/datasets/my_robot/default/motion_001
  output_dir: outputs/training/my_robot
  shared_siso_joints: all
  input_columns: [position, position_error, velocity]
  target_column: torque
  skip_optuna: true
  trials: 20
  trial_epochs: 5
  final_epochs: 50
  n_streams: 4
  noise_std: 0.0
  seed: 42
```

Use paths under `inputs/` for datasets and paths under `outputs/` for generated
results. Keep `training:` neural defaults separate from `sysid:` CMA-ES policy.
CLI arguments override these workflow defaults for one run. SysID records the
resolved values and whether they came from CLI, `run.yaml`, or a fallback in
`run_summary.yaml`.

## Assets

Add a manifest at `assets/robots/<robot>/manifest.yaml`:

```yaml
name: my_robot
description: My Robot asset manifest.
expected_assets:
  - my_robot.usd
source_notes:
  - Set ANCHOR_BENCH_ASSET_DIR to a directory containing my_robot.usd.
```

Small public smoke-test assets may be checked into
`assets/robots/<robot>/files/`. Large, private, generated, or licensed assets
should stay outside git and be provided by setting `ANCHOR_BENCH_ASSET_DIR`.

Asset lookup checks:

1. `$ANCHOR_BENCH_ASSET_DIR/<asset>`
2. `$ANCHOR_BENCH_ASSET_DIR/<robot>/<asset>`
3. `assets/robots/<robot>/files/<asset>`

Do not put USDs, meshes, FMUs, or other robot assets under `inputs/`.

## Input Datasets

SysID, benchmark, analysis, and training use the SAGE-style folder layout:

```text
inputs/datasets/<robot>/<dataset>/<motion>/
  control.csv
  state_motor.csv
  joint_list.txt
  metadata.yaml
```

`joint_list.txt` must use canonical joint names. The loader can reorder columns
onto `robot.yaml` joint order, but every requested joint must be present.

`metadata.yaml` must declare supported units:

```yaml
schema: sage
time_unit: microseconds  # or seconds
position_unit: rad
velocity_unit: rad/s
torque_unit: N*m
source: measured run or synthetic fixture
```

All workflows load this layout into the same canonical motion model. Timestamp
units come only from `metadata.yaml`; joint selection is compiled once from
`joint_list.txt`; vector widths, finite values, duplicate joints, and duplicate
timestamps are validated at the boundary.

When `target_dt` is used, the loader normalizes timestamps to seconds and
interpolates command position, state position, velocity, and torque over their
common time range. Training derives per-joint `position`, `position_error`,
`velocity`, and `torque` columns from the same in-memory motion, so robot
plugins should not generate a second set of training CSVs. If traces do not
overlap, fix the dataset rather than forcing a comparison of misaligned signals.

## Outputs From First Runs

After the first SysID run, inspect:

```text
outputs/sysid/<robot>/<run>/
  best_params.yaml
  optimization_log.csv
  run_summary.yaml
```

`best_params.yaml` is a complete schema-v2 actuator document. Its actuator
structure matches `configs/robots/<robot>/actuators.yaml`, while fit metrics and
provenance are recorded under its optional top-level `metadata` mapping. Use it
directly for local replay:

```bash
anchor-bench benchmark \
  --robot <robot> \
  --motion inputs/datasets/<robot>/<dataset>/<motion> \
  --actuator-params outputs/sysid/<robot>/<run>/best_params.yaml \
  --output outputs/sim2real_benchmark \
  --headless
```

Only copy a fitted file into `inputs/actuator_params/<robot>/` when it becomes a
published or frozen input. Generated benchmark and analysis files should remain
under `outputs/`.

## Runtime Support

Host-safe registration and dry-run planning work for any robot with valid
config files. Full simulation execution uses the shared robot-parameterized
runtime:

- SysID dispatch lives in `anchor_bench/sysid/runner.py`.
- Benchmark dispatch lives in `anchor_bench/benchmark/newton_runner.py`.
- Analysis dispatch lives in `anchor_bench/analysis/runner.py`.
- Runtime implementation is split across `anchor_bench/sysid/runtime.py`,
  `anchor_bench/benchmark/runtime.py`, and
  `anchor_bench/analysis/runtime.py`.

New robots should not add robot-named runtime modules. Declare the robot under
`configs/robots/<robot>/`; the current runtime requires a runtime USD and one
PD actuator group. Keep heavy Isaac Sim, Isaac Lab, and Newton imports inside
the non-dry-run path so dry-run tests continue to work on host-only Python.

The runtime should derive joint names from `get_robot("<robot>").spec.joint_names`
instead of hardcoding them. Use those names consistently for scene construction,
actuator application, data loading, output writing, and optimizer population
mapping.

## Validation Checklist

Run these from the repo root:

```bash
anchor-bench robots list
anchor-bench robots describe <robot>
anchor-bench sysid --robot <robot> --dry-run
anchor-bench benchmark --robot <robot> --dry-run
anchor-bench analyze --robot <robot> --dry-run
```

For actual runs, use the full container:

```bash
./setup.sh --shell
anchor-bench sysid --robot <robot> --headless
```

Before opening a PR or publishing a robot plugin, run:

```bash
python3 -m pytest -q
python3 -m ruff check anchor_bench tests scripts
python3 -m pyright
git diff --check
```

Add focused tests for any robot-specific runtime helper, asset preflight,
dataset loader behavior, and output layout introduced by the new robot.

## Common Mistakes

- Putting measured datasets or fitted run outputs under `configs/`.
- Putting USDs, meshes, FMUs, or third-party checkouts under `inputs/`.
- Adding `model_type` to `actuators.yaml`.
- Reintroducing duplicate joint-list files that can drift from `robot.yaml`.
- Hardcoding runtime joint names instead of reading the robot config.
- Treating dry-run success as proof that the full Isaac Sim/Newton runtime path
  exists for a new robot.
- Committing large generated plots, benchmark outputs, datasets, weights, or
  binary artifacts.
