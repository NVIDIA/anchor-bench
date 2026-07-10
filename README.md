# Anchor Bench

Anchor Bench is a focused system identification (SysID) and sim-to-real benchmark package for robot
actuator models on the Isaac Sim, Newton, and Isaac Lab versions pinned in
[`versions/compatibility.toml`](versions/compatibility.toml).

The repo provides robot configs, data loaders, SysID command builders, Newton
benchmark command builders, and analysis helpers around the actuator-identification workflow.

## Compatibility

[`versions/compatibility.toml`](versions/compatibility.toml) is the single
source of truth for supported Isaac Sim, Newton, Isaac Lab, and Python versions.
`setup.sh` and the Python environment checks consume that manifest.

## Installation

Use the full Docker image for SysID and benchmark playback with Isaac Sim, Isaac Lab, and Newton.

```bash
./setup.sh --full
./setup.sh --check
./setup.sh --shell
```

| Command | Purpose |
| --- | --- |
| `./setup.sh --shell` | Open the persistent full container; the environment is checked automatically |
| `./docker/run-gui.sh` | Run a single GUI container |
| `./docker/run-headless.sh` | Run a single headless container |
| `anchor-bench --help` | Show top-level CLI commands inside the container |

Container wrappers run as your host UID/GID so generated outputs stay writable.
For local Isaac Lab development with `third_party/IsaacLab`, see
[CLI commands](docs/cli.md).

## Quick Start: SO-101

SO-101 includes `assets/robots/so101/files/so101.usd` for quick checks. Set
`ANCHOR_BENCH_ASSET_DIR` only when you want to override it with your own USD
assets.

Download and convert NVIDIA's
[Anchor-Lab SO-101 motion folder](https://huggingface.co/datasets/nvidia/Anchor-Lab/tree/main/data/so101_arm_50motion)
before the first run:

```bash
# Open the full container.
./setup.sh --shell

# Download, verify, and convert every parquet in the folder.
python scripts/prepare_huggingface_parquets.py \
  https://huggingface.co/datasets/nvidia/Anchor-Lab/tree/main/data/so101_arm_50motion

# Optional: validate one epoch of real-data actuator training.
anchor-bench train-actuator --robot so101 --final-epochs 1 --skip-optuna

# SysID uses the converted training figure-eight motion by default.
anchor-bench sysid \
  --robot so101 \
  --headless --num-envs 2 --max-iter 1 --max-trajectory-len 200

# Benchmark
anchor-bench benchmark \
  --robot so101 \
  --actuator-model implicit \
  --seed 42 \
  --actuator-params inputs/actuator_params/so101/fitted_seed42.yaml \
  --headless

# Analyze
anchor-bench analyze --robot so101
```

The importer infers `so101` from the folder name and writes converted motions
under `inputs/datasets/so101/so101_arm_50motion/`. The 42 upstream
`train-*` motions remain under `train/`; the 8 `heldout-*` motions remain
under `heldout/` and must not influence normalization or model selection.

Robot `run.yaml` supplies the default SysID dataset/output, benchmark
motion/output, simulation timing, and analysis output/filter settings. CLI
arguments override those defaults when you need a one-off run.

## Common Commands

```bash
anchor-bench robots list
anchor-bench robots describe so101

# Full experiments (multiple configs)
anchor-bench experiment benchmark configs/experiments/so101_anchor_lab_figure_eight.yaml
anchor-bench experiment analyze configs/experiments/so101_anchor_lab_figure_eight.yaml
```

See [CLI commands](docs/cli.md) for the full command reference.

## Repo Layout

```text
anchor_bench/   Python package: CLI, loaders, runners, runtime helpers
configs/        Declarative robot and experiment configuration
assets/         Robot USD/MJCF/mesh assets and manifests
inputs/         Measured datasets and curated frozen actuator params
outputs/        Generated SysID, training, benchmark, and analysis artifacts
third_party/    Ignored local artifacts and optional vendor checkouts
```

See [repo structure](docs/repo-structure.md) for the detailed rules.

## Docs

- [CLI commands](docs/cli.md): full `anchor-bench` command reference.
- [Repo structure](docs/repo-structure.md): what belongs in `configs/`, `assets/`, `inputs/`, `outputs`, and `third_party/`.
- [Artifacts](docs/artifacts.md): large files, SO-101 USD handling, and generated output locations.
- [Host-safe end-to-end test](docs/end-to-end-test.md): SysID to benchmark to analysis contract coverage.
- [Robot plugins](docs/robot-plugin.md): files required to add a new robot.
- [G1 neural actuator recipes](docs/g1-neural-actuator-recipes.md): supported and experimental G1 training paths.

## Acknowledgements

Anchor Bench uses SAGE-compatible trajectory and analysis conventions for
sim-to-real benchmark data, including `control.csv`, `state_motor.csv`,
`joint_list.txt`, and `metadata.yaml` files. See
[SAGE](https://github.com/isaac-sim2real/sage) for the upstream sim-to-real
benchmarking work.

The SysID optimizer follows a PACE Sim2Real-style CMA-ES workflow, with parallel
candidate evaluation and normalized parameter bounds. See
[PACE Sim2Real](https://github.com/leggedrobotics/pace-sim2real) from ETH Zurich.
