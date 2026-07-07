# Anchor Bench

Anchor Bench is a focused system identification (SysID) and sim-to-real benchmark package for robot
actuator models on Isaac Sim 6.0, a pinned Newton 1.4 development revision, and Isaac Lab `develop`.

The repo provides robot configs, data loaders, SysID command builders, Newton
benchmark command builders, and analysis helpers around the actuator-identification workflow.

## Compatibility

- Isaac Sim: `6.0.0.0`
- Newton: `1.4.0.dev0` at `b6402cf6630df90234cdf3a2c72f9b08ac72472e`
- Warp: `1.15.0.dev20260626`
- Isaac Lab: `develop` at `5f2db9902baa724427b7aa654835aa559fce64e8`
- Python: `3.12`

The Newton Git SHA is pinned because the required GRU and dynamic-bias APIs are
not yet available together in a stable release.

Docker setup supplies the NVIDIA package index required by the pinned Warp build.
For a direct full-extra install, use
`python -m pip install --extra-index-url https://pypi.nvidia.com -e ".[full]"`.

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

```bash
# Open container
./setup.sh --shell

# SysID
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

Robot `run.yaml` supplies the default SysID dataset/output, benchmark
motion/output, simulation timing, and analysis output/filter settings. CLI
arguments override those defaults when you need a one-off run.

## Common Commands

```bash
anchor-bench robots list
anchor-bench robots describe so101

# Full experiments (multiple configs)
anchor-bench experiment benchmark configs/experiments/so101_custom_figure8_with_camera.yaml
anchor-bench experiment analyze configs/experiments/so101_custom_figure8_with_camera.yaml
```

See [CLI commands](docs/cli.md) for the full command reference.

## Repo Layout

```text
anchor_bench/   Python package: CLI, loaders, runners, runtime helpers
configs/        Declarative robot and experiment configuration
assets/         Robot USD/MJCF/mesh assets and manifests
inputs/         Measured datasets and curated frozen actuator params
outputs/        Generated SysID, benchmark, and analysis artifacts
third_party/    Ignored local artifacts and optional vendor checkouts
```

See [repo structure](docs/repo-structure.md) for the detailed rules.

## Docs

- [CLI commands](docs/cli.md): full `anchor-bench` command reference.
- [Repo structure](docs/repo-structure.md): what belongs in `configs/`, `assets/`, `inputs/`, `outputs`, and `third_party/`.
- [Artifacts](docs/artifacts.md): large files, SO-101 USD handling, and generated output locations.
- [Host-safe end-to-end test](docs/end-to-end-test.md): SysID to benchmark to analysis contract coverage.
- [Robot plugins](docs/robot-plugin.md): files required to add a new robot.
- [Newton 1.2 migration](docs/migration-newton-1.2.md): actuator schema migration notes.

## Acknowledgements

Anchor Bench uses SAGE-compatible trajectory and analysis conventions for
sim-to-real benchmark data, including `control.csv`, `state_motor.csv`,
`joint_list.txt`, and `metadata.yaml` files. See
[SAGE](https://github.com/isaac-sim2real/sage) for the upstream sim-to-real
benchmarking work.

The SysID optimizer follows a PACE Sim2Real-style CMA-ES workflow, with parallel
candidate evaluation and normalized parameter bounds. See
[PACE Sim2Real](https://github.com/leggedrobotics/pace-sim2real) from ETH Zurich.
