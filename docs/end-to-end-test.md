# Verifying the phase pipeline

The authoritative workflow has two verification layers:

- host-safe contract tests exercise resolution, immutable outputs, promotion,
  metadata, and runtime boundaries with fakes;
- full-runtime checks exercise the pinned Newton, Isaac, MuJoCo, Torch, and GPU
  stack with real assets and trajectories.

Keep host-safe coverage in the default suite. Full-runtime execution belongs in
the full container and must not be required by ordinary unit tests.

## Host-safe contract suite

Run the phase-focused suite from the repository root:

```bash
python3 -m pytest -q \
  tests/test_robot_identity.py \
  tests/test_robot_phase_configs.py \
  tests/test_core_hashing.py \
  tests/test_core_profiles.py \
  tests/test_core_manifest_schema.py \
  tests/test_core_artifacts.py \
  tests/test_phase_paths.py \
  tests/test_sysid_schema.py \
  tests/test_sysid_state.py \
  tests/test_sysid_execution.py \
  tests/test_sysid_runtime_backend.py \
  tests/test_training_contract.py \
  tests/test_training_episode.py \
  tests/test_training_engine.py \
  tests/test_training_phase.py \
  tests/test_benchmark_schema.py \
  tests/test_benchmark_phase.py \
  tests/test_benchmark_replay.py \
  tests/test_deploy_schema.py \
  tests/test_deploy_promotion.py \
  tests/test_deploy_runtime.py \
  tests/test_authoritative_cli.py
```

Then run the full suite and static checks:

```bash
python3 -m pytest -q
python3 -m ruff check anchor_bench tests scripts
python3 -m pyright
git diff --check
```

Torch-dependent tests skip when Torch is not installed. Tests that inspect a
TorchScript ZIP directly remain host-safe and do not import Torch.

## What the contracts prove

### Cross-cutting artifact protocol

The core tests cover strict YAML, duplicate-key rejection, canonical semantic
hashes, repository path containment, source fingerprints, stable locks, atomic
installation, manifest-last completion, verified no-ops, and corruption
failures.

### SysID

The SysID tests cover profile validation, shared and per-joint optimizer layout,
accepted revision-zero state, authenticated parent-state selection, the
`parent + fitted patch = candidate` equation, immutable run output, sparse
promotion, and optional exact benchmark evidence.

The backend tests use injected simulation and optimizer boundaries. They prove
that execution consumes the resolved profile and frozen parent state without
requiring Isaac Sim or Newton on the host.

### Training and metadata

Training tests cover SISO, shared mappings, MISO, and MIMO dimensions without a
separate topology tag. They verify episode splitting before mapping expansion,
linear command interpolation, recurrent resets, previous-target shifting,
full-torque and residual targets, multi-component feature ordering, pooled
normalization, dynamic-bias provider behavior, and training-asset effort limits.

Schema-v3 tests authenticate mapping axes, tensor dimensions, feature domains,
normalization, target-specific clamping, residual baselines, dynamic-bias and
previous-torque derivations, and source hashes embedded in `network.pt`.

### Benchmark

Benchmark tests prove that `--run` selects one exact completed SysID or training
candidate. They validate source manifests and artifacts before runtime, enforce
joint and timestep contracts, ignore output root for semantic identity, produce
real and simulated trajectories plus metrics, and return a pre-runtime no-op for
valid existing evidence.

Replay tests use fake Newton/Torch boundaries to cover PD candidates, schema-v3
GRUs, mapping-local recurrent state, dynamic bias, candidate-only metrics, and
omitted-joint behavior.

### Deployment

Deployment tests cover explicit PD and neural ownership, accepted-source
resolution, mapping and artifact authentication, output-overlap rejection,
frozen baseline and effort-limit values, atomic promotion, and no-op behavior.

Runtime tests prove that the frozen state—not mutable SysID or training YAML—is
the runtime authority. They cover native explicit PD, full-torque and residual
neural semantics, mapping-local recurrent state, shared dynamic-bias evaluation,
and leaving omitted asset joints untouched.

### CLI and lazy imports

`tests/test_authoritative_cli.py` verifies the public command tree, typed
overrides, exact-candidate forwarding, dry-run behavior, explicit promotion, and
host-safe phase exports. Dry runs must not resolve a runtime USD or import Torch,
Newton, Isaac Lab, or simulation modules.

## Full-runtime check

Run this in a disposable worktree or clone: promotion intentionally modifies
accepted YAML under `configs/robots/<robot>/`.

Start with the full environment and a real SAGE dataset:

```bash
./setup.sh --shell
./setup.sh --check
anchor-bench robots describe so101
anchor-bench sysid --robot so101 --dry-run
anchor-bench train-actuator --robot so101 --dry-run
```

Then exercise each boundary, using the exact run directory printed by the prior
command:

```bash
# 1. Produce an immutable physical candidate.
anchor-bench sysid --robot so101 --num-envs 2 \
  --optimizer '{"type":"cmaes","sigma":0.5,"max_iterations":1,"epsilon":0.001}'

# 2. Replay that exact candidate on held-out data.
anchor-bench benchmark --robot so101 --run <completed-sysid-run> --headless

# 3. Preview, then accept the candidate if desired.
anchor-bench promote sysid --robot so101 --run <completed-sysid-run> --dry-run
anchor-bench promote sysid --robot so101 --run <completed-sysid-run>

# 4. Train and replay one exact schema-v3 network artifact.
anchor-bench train-actuator --robot so101 --final-epochs 1 --skip-optuna
anchor-bench benchmark --robot so101 --run <completed-training-run> --headless

# 5. Preview and accept the model, then freeze deployment.
anchor-bench promote train --robot so101 --run <completed-training-run> --dry-run
anchor-bench promote train --robot so101 --run <completed-training-run>
anchor-bench promote deploy --robot so101 --dry-run
anchor-bench promote deploy --robot so101
```

The checked-in SO-101 deploy profile is PD-only. To test neural deployment, add
a temporary deploy profile that explicitly references the promoted model key and
its exact metadata mapping; do not overlap its output joints with a PD owner.

## Inspecting results

For every completed run:

1. verify that `manifest.yaml` exists and was written last;
2. compare its `run_input_sha256` with the directory name;
3. confirm generated-file hashes match the bytes on disk;
4. rerun the identical command and expect `status: no-op`;
5. confirm no artifact was copied under `configs/`.

For promotion, inspect the revision, exact source path and hashes, affected
parameter cells or model key, and optional benchmark provenance. For deployment,
confirm every output has exactly one owner and every frozen value came from
accepted SysID state or authenticated neural metadata.

## What these checks do not prove

Contract completion does not establish optimizer convergence, model quality,
sim-to-real accuracy, hardware safety, or a benchmark acceptance threshold.
Those are experiment and review decisions, not implicit pipeline side effects.
