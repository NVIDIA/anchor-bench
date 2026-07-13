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
`cmaes` package. Neural actuator training requires `torch`;
`dynamic_bias` additionally requires upstream Newton 1.4 and the robot USD.
Dry runs import neither dependency. Omit `--skip-optuna` only in an
environment with `optuna` installed.

Robot `run.yaml` provides default SysID data/output paths, training
data/output/model settings, benchmark motion/output paths, simulation timing,
and analysis output/filter settings. Passing CLI flags overrides those defaults
for that run; SysID writes the resolved values and their sources into
`run_summary.yaml`.


## Neural Actuator Training

The native training entrypoint is intentionally separate from CMA-ES SysID:

```bash
anchor-bench train-actuator \
  --robot so101 \
  --model gru \
  --data inputs/datasets/so101/so101_arm_50motion/train \
  --input-joints Rotation \
  --output-joints Rotation \
  --output outputs/training/so101/rotation \
  --input-columns position,position_error,velocity \
  --target-column torque \
  --hidden-size 192 \
  --num-layers 2 \
  --dropout 0.225 \
  --skip-optuna \
  --dry-run
```

Training consumes SAGE motion folders. Command position, measured position,
velocity, torque, and `position_error` are aligned directly in memory.
Recorded episodes are split before joint mappings are expanded.

Training architecture, optimization, TBPTT, output-head, normalization, and
PD-baseline settings can live under `training:` in the robot's `run.yaml`.
The matching CLI option overrides the YAML value and appears in the dry-run's
reproducible command:

```yaml
training:
  hidden_size: 192
  num_layers: 2
  dropout: 0.225
  learning_rate: 0.0004949926657319083
  weight_decay: 0.0000003376044479094932
  huber_delta: 0.3149430512234676
  chunk_length: 100
  output_activation: tanh
  output_scale: 15.0
  target_normalization: identity
  pd_kp: 40.0
  pd_kd: 1.0
```

The corresponding flags are `--hidden-size`, `--num-layers`, `--dropout`,
`--learning-rate`, `--weight-decay`, `--huber-delta`, `--chunk-length`,
`--output-activation`, `--output-scale`, `--target-normalization`,
`--pd-kp`, and `--pd-kd`. Explicit architecture, optimizer, and chunk
settings stay fixed during Optuna; omitted searchable values are tuned.
Dropout must be zero when explicitly selecting a one-layer GRU.

A `linear` head returns the scaled linear output. A `tanh` head returns
`tanh(linear_output) * output_scale`, so the example is bounded to ±15.
`standard` normalizes training targets by their measured mean and standard
deviation; `identity` trains in physical torque units. PD gains accept either
one broadcast scalar or an ordered comma-separated/YAML vector, for example
`--pd-kp 40` or `--pd-kp 40,35,30,25`. Vector order follows the active PD
domain: input-joint order for `solver_pd`, otherwise residual output-joint
order. Shared mappings must still have identical values by model slot.

Use `training.mappings` when one artifact shares weights across more than one
physical application:

```yaml
training:
  mappings:
    - input_joints: [Rotation]
      output_joints: [Rotation]
    - input_joints: [Elbow]
      output_joints: [Elbow]
```

Every mapping in one run has the same input and output width. A single 1-to-1
mapping is SISO, several 1-to-1 mappings share one SISO model, an N-to-1 mapping
is MISO, and an N-to-M mapping is MIMO. `output_joints` defaults to
`input_joints`. `--input-joints`/`--output-joints` is shorthand for one
mapping and replaces
the YAML mapping selection for one run. Shared SISO must list each physical
mapping explicitly.

Use `--model gru` for full-torque GRU prediction from
`[position, position_error, velocity]`. Use `--model hybrid_residual`
when the native PD/friction actuator remains active and the GRU should
learn residual torque. `dynamic_bias` derives MuJoCo `qfrc_bias` from the
recorded state without stepping. `previous_torque` is the previous raw,
denormalized network output before clamping, initialized to physical zero for
every episode and mapping. Training and validation roll it autoregressively;
shifted targets are used only for its normalization statistics. The single PD
baseline follows `input_joints` when `solver_pd` is enabled and
`output_joints` for residual-only models; friction always follows `output_joints`.
Shared mappings must have identical values by model slot.

Training writes a raw checkpoint, stats JSON, and TorchScript network under
`outputs/training/<robot>/...`. The TorchScript `.pt` embeds
`metadata.json` with mappings, feature order and domains, dimensions,
normalization, the configured control timestep, target semantics, and
derivation/baseline metadata.
Schema-v3 artifacts target the future Newton `GRUActuator`; the legacy Anchor
Isaac Lab GRU shim is not a supported consumer.
