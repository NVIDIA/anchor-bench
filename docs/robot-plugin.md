# Adding a robot

Robot support begins with declarative identity and phase files. Keep runtime
code robot-agnostic: canonical joints, assets, recipes, accepted values, and
actuator ownership all come from configuration.

## File layout

Start with:

```text
configs/robots/<robot>/
  robot.yaml
  sysid/
    profiles/default.yaml
    default.yaml
  train/
    profiles/default.yaml       # when neural training is supported
    default.yaml
  benchmark/
    profiles/default.yaml       # when replay data is available
  deploy/
    profiles/default.yaml       # when a deploy composition is defined

assets/robots/<robot>/
  manifest.yaml
  files/<runtime-usd>            # optional for small public assets

inputs/datasets/<robot>/<dataset>/<motion>/
  control.csv
  state_motor.csv
  joint_list.txt
  metadata.yaml
```

Do not create workflow-wide robot files. Each phase owns its recipe, and only
promotion writes accepted or frozen state.

## Robot identity

`robot.yaml` contains stable identity only:

```yaml
schema_version: 1
name: my_robot
display_name: My Robot
fixed_base: true
asset_path: assets/robots/my_robot/manifest.yaml
runtime_usd: my_robot.usd
joint_names: [joint_1, joint_2]
joint_aliases:
  shoulder: joint_1
ee_frame: tool_link
```

`joint_names` is the canonical runtime order. Profile and CLI selectors may use
aliases, then resolve to canonical names before hashing or writing artifacts.
Datasets should use canonical names. Alias keys cannot collide with canonical
names, and every target must be canonical.

`asset_path` must be `assets/robots/<robot>/manifest.yaml`. `runtime_usd` names
the USD below the resolved robot asset root. The identity file contains no
optimizer settings, actuator values, ownership, training architecture, or
accepted artifact paths.

Check it without loading Newton or the USD:

```bash
anchor-bench robots describe my_robot
```

## Asset manifest

Add `assets/robots/<robot>/manifest.yaml`:

```yaml
name: my_robot
description: My Robot runtime asset.
expected_assets:
  - my_robot.usd
source_notes:
  - Set ANCHOR_BENCH_ASSET_DIR when the asset is not checked in.
```

Small redistributable assets may live under `assets/robots/<robot>/files/`.
Otherwise set `ANCHOR_BENCH_ASSET_DIR` to a directory containing either
`<runtime-usd>` or `<robot>/<runtime-usd>`. Actual runs hash the resolved asset;
dry runs validate the manifest without resolving the USD.

## SysID

### Initial accepted state

Create `sysid/default.yaml` at revision zero with complete canonical values for
the joints and parameters needed by the robot's first SysID or PD deployment:

```yaml
schema_version: 1
revision: 0
parameters:
  stiffness: {joint_1: 40.0, joint_2: 30.0}
  damping: {joint_1: 2.0, joint_2: 1.5}
  armature: {joint_1: 0.01, joint_2: 0.01}
  dry_friction: {joint_1: 0.1, joint_2: 0.1}
  viscous_friction: {joint_1: 0.1, joint_2: 0.1}
provenance:
  history: []
```

This is accepted physical state, not a search recipe. Later edits happen through
`anchor-bench promote sysid`.

### SysID profile

Create `sysid/profiles/default.yaml`:

```yaml
schema_version: 1
data: inputs/datasets/my_robot/default/motion_001
joints: [joint_1, joint_2]
simulation:
  physics_freq: 500
  control_freq: 500
  render_freq: 500
  integrator: implicitfast
  solver_iterations: 10
  line_search_iterations: 10
num_envs: 64
buffer_time: 2.0
seed: 42
optimizer:
  type: cmaes
  sigma: 0.5
  max_iterations: 100
  epsilon: 0.001
parameters:
  stiffness:
    mode: per_joint
    bounds: {lower: 1.0, upper: 80.0}
  damping:
    mode: per_joint
    bounds: {lower: 0.01, upper: 5.0}
  armature:
    mode: shared
    bounds: {lower: 0.0, upper: 0.1}
  dry_friction:
    mode: per_joint
    bounds: {lower: 0.0, upper: 1.0}
  viscous_friction:
    mode: per_joint
    bounds: {lower: 0.0, upper: 2.0}
loss:
  position: 1.0
  velocity: 0.0
```

`per_joint` creates one optimizer variable per selected joint. `shared` creates
one variable expanded across the selected joints. Run and promote separately:

```bash
anchor-bench sysid --robot my_robot --dry-run
anchor-bench sysid --robot my_robot
anchor-bench promote sysid --robot my_robot --run <completed-sysid-run>
```

## Neural training

Initialize `train/default.yaml` as an empty accepted catalog:

```yaml
schema_version: 1
revision: 0
models: {}
```

One training profile produces one artifact and one set of weights and pooled
normalization statistics. Each mapping is one physical application of those
weights. All mappings must have equal input and output widths; each independently
requires its output joints to be a subset of its input joints.

Example `train/profiles/default.yaml`:

```yaml
schema_version: 1
sysid_state: default
data: inputs/datasets/my_robot/default/train
sample_dt_s: 0.002
network:
  type: gru
  hidden_size: 64
  num_layers: 1
  dropout: 0.0
target: torque_residual
mappings:
  - input_joints: [joint_1, joint_2]
    output_joints: [joint_2]
input_features:
  - {name: position, domain: input, components: [value]}
  - {name: position_error, domain: input, components: [value]}
  - {name: velocity, domain: input, components: [value]}
  - {name: dynamic_bias, domain: input, components: [value]}
  - {name: previous_torque, domain: output, components: [value]}
fit:
  skip_optuna: true
  trials: 20
  trial_epochs: 5
  final_epochs: 50
  noise_std: 0.0
  seed: 42
```

The example is MISO: two input joints and one output joint. Multiple 1-to-1
mappings naturally train a shared SISO artifact. A single multi-output mapping
is MIMO. Use separate profiles for separate artifacts. The training-only
`input_joints: all` selector means one coupled all-joint mapping.

Residual targets freeze PD and friction baselines from the selected SysID state.
Neural effort limits come from the runtime asset for each output joint, so
actual training loads that asset through pinned Newton. If `dynamic_bias` is
selected, episode assembly additionally consumes the MuJoCo solver's
`eval_dynamic_bias` result. Dry-run remains host-safe.

The generated `network.pt` embeds schema-v3 metadata, so benchmark and deploy
can recover exact mappings, features, normalization, target semantics, timing,
baselines, effort limits, and provenance. Promote the run only after review:

```bash
anchor-bench train-actuator --robot my_robot
anchor-bench promote train --robot my_robot --run <completed-training-run>
```

## Benchmark

Benchmark profiles describe held-out replay, not a candidate. The candidate is
always supplied with `--run`:

```yaml
schema_version: 1
data: inputs/datasets/my_robot/default/heldout
motions: "*"
simulation:
  physics_freq: 500
  control_freq: 500
  render_freq: 500
  integrator: implicitfast
  solver_iterations: 10
  line_search_iterations: 10
num_envs: 1
buffer_time: 2.0
sync_initial_state: true
signals: [position, velocity]
output_root: outputs/sim2real_benchmark
```

```bash
anchor-bench benchmark \
  --robot my_robot \
  --run <completed-sysid-or-training-run> \
  --dry-run
```

Required candidate joints must exist in the selected trajectories. A neural
candidate's `sample_dt_s` and a SysID candidate's source control timestep must
match benchmark control frequency. Benchmark evidence is immutable, optional
during promotion, and never an automatic acceptance decision.

## Deployment

A deployment profile explicitly assigns output ownership. It refers to accepted
model keys and accepted SysID state, never arbitrary candidate paths:

```yaml
schema_version: 1
simulation:
  physics_freq: 500
  control_freq: 500
  render_freq: 500
  integrator: implicitfast
  solver_iterations: 10
  line_search_iterations: 10
composition:
  neural:
    - name: joint_2_residual
      model: default
      mappings:
        - index: 0
          input_joints: [joint_1, joint_2]
          output_joints: [joint_2]
  pd:
    - name: joint_1_pd
      joints: [joint_1]
      effort_limits: {joint_1: 3.0}
```

Joint lists must be explicit; composition keywords such as `all`, `remaining`,
and `exclude` are rejected. Neural and PD inputs may overlap, but no output joint
may have more than one owner. Omitted joints retain asset-defined behavior.

PD values are read from accepted `sysid/default.yaml`. Neural mappings and
physics come from the accepted archive's schema-v3 metadata. A full-torque GRU
owns and clamps total effort. A residual GRU adds its output to its frozen
baseline and clamps only the residual. Every active network timestep must equal
`1 / control_freq`.

Freeze the composition explicitly:

```bash
anchor-bench promote deploy --robot my_robot --dry-run
anchor-bench promote deploy --robot my_robot
```

This creates `deploy/default.yaml`. Runtime consumes that frozen file and does
not reread mutable accepted state until another deploy promotion.

## Dataset contract

Each SAGE motion contains `control.csv`, `state_motor.csv`, `joint_list.txt`,
and `metadata.yaml`. Metadata declares timestamp and physical units. Requested
joint columns must be finite, unique, and present.

Training treats recorded state timestamps as authoritative and linearly
interpolates commands onto them. It splits train and validation by recorded
episode before expanding mappings, preventing shared-weight mappings from
leaking the same episode across the split.

## Validation checklist

```bash
anchor-bench robots describe my_robot
anchor-bench sysid --robot my_robot --dry-run
anchor-bench train-actuator --robot my_robot --dry-run
anchor-bench benchmark --robot my_robot --run <completed-run> --dry-run
python3 -m pytest -q
python3 -m ruff check anchor_bench tests scripts
python3 -m pyright
git diff --check
```

Keep imports of Isaac Sim, Isaac Lab, Newton, Warp, and Torch inside actual
execution paths. A successful dry run proves configuration and resolution, not
runtime availability or physical correctness.
