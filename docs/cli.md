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
  --joint-type Rotation \
  --output outputs/training/so101/rotation \
  --input-columns position,position_error,velocity \
  --target-column torque \
  --skip-optuna \
  --dry-run
```

Training consumes SAGE motion folders. Command position, measured position,
velocity, torque, and `position_error` are aligned directly in memory.
Recorded episodes are split before joint mappings are expanded.

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
`input_joints`. The existing `--joint-type full` spelling remains shorthand
for one shared SISO mapping per canonical robot joint. CLI
`--input-joints/--output-joints` replaces the YAML mapping selection for one
run.

Use `--model gru` for full-torque GRU prediction from
`[position, position_error, velocity]`. Use `--model hybrid_residual`
when the native PD/friction actuator remains active and the GRU should
learn residual torque. `dynamic_bias` derives MuJoCo `qfrc_bias` from the
recorded state without stepping. `previous_torque` is the previous final
training target, initialized to zero for every episode and mapping. Shared
mappings using `solver_pd` or residual targets must have identical baseline
values by model slot; otherwise train separate artifacts.

Training writes a raw checkpoint, stats JSON, and TorchScript network under
`outputs/training/<robot>/...`. The TorchScript `.pt` embeds
`metadata.json` with mappings, feature order and domains, dimensions,
normalization, the median recorded state timestep, target semantics, and
derivation/baseline metadata.
The current upstream Newton 1.4 controller does not yet consume this schema.
