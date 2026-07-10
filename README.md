# Anchor Bench

Anchor Bench identifies robot actuator parameters, trains neural actuator
models, replays exact candidates, and freezes accepted actuator compositions for
Newton. Supported Isaac Sim, Newton, Isaac Lab, and Python versions are pinned
in [`versions/compatibility.toml`](versions/compatibility.toml).

## The workflow

Anchor Bench keeps four kinds of state separate:

1. **Profiles** under `configs/robots/<robot>/<phase>/profiles/` are editable
   recipes.
2. **Runs** under `outputs/` are immutable, content-addressed candidates.
3. **Accepted state** in `sysid/default.yaml` and `train/default.yaml` records
   deliberate promotion decisions.
4. **Frozen deployment state** in `deploy/default.yaml` is the exact actuator
   composition consumed at runtime.

The phases are explicit:

```text
SysID profile -> immutable SysID run -> optional benchmark -> promote sysid
Training profile + SysID state -> immutable training run -> optional benchmark -> promote train
Deploy profile + accepted states -> promote deploy -> frozen deploy/default.yaml
```

Running or benchmarking a candidate never promotes it. Benchmark evidence is
optional and is not an automatic quality gate.

## Installation

Use the full Docker image for Newton, Isaac Sim, and GPU-backed execution:

```bash
./setup.sh --full
./setup.sh --check
./setup.sh --shell
```

Container wrappers run as your host UID/GID, so generated `outputs/` remain
writable from the host. Host-safe commands such as `--dry-run`, profile
validation, and artifact inspection keep Torch, Newton, Isaac Lab, and robot
runtime imports lazy.

## Quick start

Inspect the checked-in SO-101 recipes without launching the runtime:

```bash
anchor-bench robots describe so101
anchor-bench sysid --robot so101 --dry-run
anchor-bench train-actuator --robot so101 --dry-run
```

An actual phase run prints its immutable run directory. Use that exact directory
for benchmarking or promotion:

```bash
anchor-bench sysid --robot so101

anchor-bench benchmark \
  --robot so101 \
  --run outputs/sysid/so101/default/seed-42/<run-input-sha256> \
  --headless

anchor-bench promote sysid \
  --robot so101 \
  --run outputs/sysid/so101/default/seed-42/<run-input-sha256>
```

Training may use the accepted SysID state or an explicit completed SysID
candidate. Deployment may use only promoted training and SysID state:

```bash
anchor-bench train-actuator --robot so101
anchor-bench promote train --robot so101 --run outputs/train/so101/default/seed-42/<run-input-sha256>
anchor-bench promote deploy --robot so101
```

Use `--dry-run` on execution and promotion commands before a runtime or state
change. See [CLI commands](docs/cli.md) for override syntax and full examples.

## SO-101 data

SO-101 includes a small runtime USD. Download and convert NVIDIA's
[Anchor-Lab SO-101 motions](https://huggingface.co/datasets/nvidia/Anchor-Lab/tree/main/data/so101_arm_50motion)
before an actual run:

```bash
python scripts/prepare_huggingface_parquets.py \
  https://huggingface.co/datasets/nvidia/Anchor-Lab/tree/main/data/so101_arm_50motion
```

Converted SAGE motions are written under
`inputs/datasets/so101/so101_arm_50motion/`. Training and held-out data must
remain separate so validation and normalization do not leak across the split.

## Repository layout

```text
anchor_bench/   Phase schemas, resolvers, execution, promotion, and runtime adapters
configs/        Robot identity, editable profiles, accepted state, and frozen deploy state
assets/         Robot asset manifests and small public runtime assets
inputs/         Measured datasets
outputs/        Immutable runs, benchmark evidence, checkpoints, and generated results
third_party/    Ignored local artifacts and optional vendor checkouts
```

Do not commit large datasets, neural weights, FMUs, generated benchmark output,
or plots.

## Documentation

- [CLI commands](docs/cli.md)
- [Artifacts and immutable runs](docs/artifacts.md)
- [Repository structure](docs/repo-structure.md)
- [Adding a robot](docs/robot-plugin.md)
- [Pipeline verification](docs/end-to-end-test.md)
- [G1 neural actuator recipes](docs/g1-neural-actuator-recipes.md)

## Acknowledgements

Anchor Bench uses SAGE-compatible trajectory conventions. See
[SAGE](https://github.com/isaac-sim2real/sage) for the upstream sim-to-real
benchmarking work. Its SysID optimizer follows the parallel CMA-ES workflow used
by [PACE Sim2Real](https://github.com/leggedrobotics/pace-sim2real).
