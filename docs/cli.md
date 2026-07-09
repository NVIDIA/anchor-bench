# Anchor Bench CLI

Run these commands inside the full container shell opened by `./setup.sh --shell`.
From the host, prefix commands with `./setup.sh --shell`; for example,
`./setup.sh --shell anchor-bench robots list`.

## Container Workflow

The public persistent-container entrypoint is:

```bash
./setup.sh --shell
```

Container wrappers run processes as your host UID/GID by default so generated
artifacts under bind-mounted paths such as `outputs/` remain writable from both
the host and container. Set `ANCHOR_BENCH_RUN_AS_ROOT=1` only when you
intentionally need a root container process.

By default the full Docker image uses the pinned Isaac Lab checkout baked into
the image at `/opt/IsaacLab`. To develop against a host-editable checkout,
bootstrap and use `third_party/IsaacLab` explicitly:

```bash
ANCHOR_BENCH_ISAACLAB_SOURCE=third_party ./setup.sh --shell
```

That mode clones `third_party/IsaacLab` if needed, checks out the pinned Isaac
Lab commit from `versions/compatibility.toml`, and points the container at the
mounted checkout.

## Commands

| Command | Purpose |
| --- | --- |
| `anchor-bench --help` | Show top-level CLI commands |
| `anchor-bench robots list` | List registered robots (`g1`, `h1`, `so101`, `teststand`, `ur10e`) |
| `anchor-bench robots describe so101` | Print robot metadata and validation errors as JSON |
| `anchor-bench sysid --robot so101 --dry-run` | Build a SysID command plan from `run.yaml` defaults without launching simulation |
| `anchor-bench train-actuator --robot so101 --dry-run` | Build a neural actuator training plan from `run.yaml` defaults without importing ML libraries |
| `anchor-bench benchmark --robot so101 --dry-run` | Build a benchmark playback plan from `run.yaml` defaults without launching simulation |
| `anchor-bench analyze --robot so101 --dry-run` | Build an analysis plan from `run.yaml` defaults |
| `anchor-bench analyze --robot so101 --motion <name-or-glob> --dry-run` | Override the `run.yaml` analysis motion filter |
| `anchor-bench analyze --robot so101 --result-folder <benchmark_dir> --require-end-effector` | Fail instead of skipping when SO-101 end-effector FK is unavailable |
| `anchor-bench experiment benchmark configs/experiments/so101_anchor_lab_figure_eight.yaml --dry-run` | Expand an experiment config into batched benchmark runs |
| `anchor-bench experiment analyze configs/experiments/so101_anchor_lab_figure_eight.yaml --dry-run` | Build the analysis plan for an experiment config |

Use `--dry-run` for host-safe command construction checks. Actual SysID and
benchmark runs require the full Isaac Sim/Newton runtime and, for SysID, the
`cmaes` package. Neural actuator training requires `torch`; it imports Newton
only for `dynamic_bias`. Omit `--skip-optuna` only in an environment
with `optuna` installed.

Robot `run.yaml` provides default SysID data/output paths, training
data/output/model settings, benchmark motion/output paths, simulation timing,
and analysis output/filter settings. Passing CLI flags overrides those defaults
for that run; SysID writes the resolved values and their sources into
`run_summary.yaml`.


## Neural Actuator Training

The training entrypoint is separate from CMA-ES SysID. Start with a dry run:

```bash
anchor-bench train-actuator \
  --robot so101 \
  --model gru \
  --data inputs/datasets/so101/so101_arm_50motion/train \
  --input-joints Rotation \
  --output outputs/training/so101/rotation \
  --input-columns position,position_error,velocity \
  --skip-optuna \
  --dry-run
```

Training consumes whole SAGE motion folders. Recorded episodes are split into
training and validation sets before joint mappings are expanded, so different
mappings of one recording cannot leak across the split.

### Groups and mappings

One training group produces one artifact and owns one set of weights and
normalization statistics. Each mapping in the group describes one simultaneous
application of that model. All mappings in a group must have the same input and
output widths; their physical joint names may differ.

For example, SO-101 shares one scalar model across all six joints:

```yaml
training:
  groups:
    - name: so101_actuator
      mappings:
        - input_joints: [Rotation]
          output_joints: [Rotation]
        - input_joints: [Pitch]
          output_joints: [Pitch]
        - input_joints: [Elbow]
          output_joints: [Elbow]
        - input_joints: [Wrist_Pitch]
          output_joints: [Wrist_Pitch]
        - input_joints: [Wrist_Roll]
          output_joints: [Wrist_Roll]
        - input_joints: [Jaw]
          output_joints: [Jaw]
```

This is ordinary SISO at each mapping: the model still has one input-joint axis
and one scalar output. The six source/mapping sequences share weights, and their
samples are pooled into the same scalar normalization statistics. Separate
groups produce separate artifacts and do not share weights or statistics.

Top-level `input_joints` and `output_joints`, including the corresponding CLI
flags, are shorthand for one default group containing one mapping. Omit outputs
to use the same joints as inputs. Supplying CLI joint flags replaces the
declarative group selection for that invocation.

Use ordered joint lists for coupled mappings. Joint aliases are resolved through
`robot.yaml`, and outputs must be a subset of inputs independently in every
mapping. `input_joints: all` continues to mean one coupled all-joint input
mapping; it never expands into several shared SISO mappings. `output_joints:
all` means all inputs from that same mapping.

Artifact basenames contain the group name and a digest of the complete ordered
mapping contract.

```bash
anchor-bench train-actuator \
  --robot g1 \
  --model gru \
  --data <g1-sage-motion-or-dataset-root> \
  --input-joints right_shoulder_pitch,right_shoulder_roll,right_shoulder_yaw,right_elbow \
  --output-joints right_elbow \
  --input-columns position,position_error,velocity,previous_torque \
  --output outputs/training/g1/right_arm_to_elbow \
  --skip-optuna \
  --dry-run
```

Topology describes one mapping, not how many mappings share the artifact:

| Topology | Input joints per mapping | Output joints per mapping | Application |
| --- | ---: | ---: | --- |
| SISO | 1 | 1 | One joint predicts one torque |
| MISO | More than 1 | 1 | Joint context predicts one selected torque |
| MIMO | More than 1 | More than 1 | One coupled mapping predicts an ordered torque vector |

Each feature block has shape `[T, A, C]`, where `A` is `I` for an
input-joint block or `O` for an output-joint block, and `C` is its ordered
channel count. `I` and `O` come from one mapping, never from the total number
of physical joints across the group. For example, a three-channel velocity
block is `[T, I, 3]`, while `previous_torque` is `[T, O, 1]`. The torque
target is `[T, O]`. Normalization is pooled across mappings by logical tensor
coordinate.

Blocks are flattened in `input_columns` order, then axis-major/channel-minor
within each block. The GRU input width is `D = sum(A * C)`, and exported
`input_feature_specs` records each feature domain and channel order. The built-in
SAGE assembler emits scalar blocks with channel `value`; another data producer
must define channels explicitly for vector-valued blocks.

### Features, targets, and timing

Supported inputs are:

| Feature | Training value |
| --- | --- |
| `position` | Measured joint position |
| `position_error` | Linearly interpolated command position minus measured position |
| `velocity` | Measured joint velocity |
| `solver_pd` | `kp * position_error - kd * velocity` |
| `dynamic_bias` | MuJoCo `qfrc_bias` evaluated by Newton without stepping simulation |
| `previous_torque` | Previous target, with physical zero at the start of each episode |

`dynamic_bias` uses every recorded joint that maps into the runtime model.
Unrecorded non-input joints retain their model-default position and zero
velocity. It is evaluated only during real training, so dry runs remain
host-safe. Execution with this feature requires the full Newton/MuJoCo runtime,
a declared `runtime_usd`, and
`SolverMuJoCo.eval_dynamic_bias(state, out)`. Use `dynamic_bias`, not the
legacy name `qfrc_bias`.

For a full-torque `--model gru`, the target is measured torque. For
`--model hybrid_residual`, the target is measured torque minus the shared PD
and friction baselines. In a residual model, `previous_torque` therefore
means the previous residual target. It means the previous measured torque in a
full-torque model.
Training uses that previous target as supervised recurrence. Each recorded
episode/mapping pair is an independent recurrent sequence: previous torque and
hidden state reset between mappings and episodes, while hidden state continues
across TBPTT chunks. A runtime consumer feeds back the previous raw network
output and keeps independent recurrent state for each physical mapping.

PD values are required by residual training and whenever `solver_pd` is an
input. Friction values are required by residual training. Supply a complete
pair with `--pd-kp/--pd-kd` or
`--friction-dry/--friction-viscous`. Resolution order is CLI, then
`training:` in `run.yaml`, then uniform values inferred from
`actuators.yaml`. Each group/artifact has one shared baseline. Fallback
inference considers every relevant joint across that group's mappings and fails
when those actuator values differ. Separate groups may infer different
baselines.

The expected sample period is `1 / simulation.control_freq`. Every recorded
state interval must match it or training fails. Training does not resample an
episode or apply zero-order hold; commands continue to be linearly interpolated
onto the recorded state timestamps.

### Artifacts

Each group writes a raw checkpoint, normalization stats JSON, and a TorchScript
network under `outputs/training/<robot>/...`. The TorchScript archive embeds
schema-version-2 `metadata.json`, so a runtime consumer needs only that file.
The metadata records the group name, every ordered physical joint mapping,
feature and target semantics, sample period, group-pooled normalization, delay
handling, applicable PD/friction baselines, and derivation provenance for
dynamic bias and previous torque. Model dimensions are validated against one
mapping's axes, not against the union of physical joint names.

Anchor Bench training does not install or modify Newton. A runtime loading an
enriched or coupled artifact must support this metadata contract; the legacy
Anchor Bench actuator-schema compatibility path still accepts only its older
runtime feature set. Do not commit generated weights, datasets, or stats.
