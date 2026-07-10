# Anchor Bench CLI

Run runtime commands inside the full container opened by `./setup.sh --shell`.
Profiles, dry runs, and promotion validation are also designed to work on a
host-only Python installation.

## Configuration precedence

Each execution phase reads one profile from
`configs/robots/<robot>/<phase>/profiles/`:

- Omitting `--profile` selects `default.yaml`.
- A named profile replaces `profiles/default.yaml`; profiles are not merged.
- Typed CLI fields replace fields in the selected profile.
- Structural overrides such as `--joints`, `--mappings`, and
  `--input-features` replace the complete list.

`--dry-run` validates and prints the resolved plan without writing a run or
importing Torch, Newton, Isaac Lab, or a robot runtime asset. It does not prove
that those execution dependencies are installed. The printed identity is a
preview: real execution replaces manifest/declared-path fingerprints with the
actual runtime asset and dataset bytes, so its authoritative run hash may
differ.

## Robot identity

```bash
anchor-bench robots list
anchor-bench robots describe so101
```

`describe` prints the identity in `robot.yaml` plus validation errors. Robot
identity supplies canonical joint order, aliases, the runtime USD name, and the
end-effector frame; it does not contain workflow or actuator settings.

## SysID

Preview or create one immutable SysID candidate:

```bash
anchor-bench sysid \
  --robot g1 \
  --profile default \
  --dry-run
```

Useful overrides are:

```text
--initial-state default|<completed-run>/sysid_state.yaml
--data <dataset-directory>
--joints joint_a,joint_b
--simulation '<complete JSON object>'
--num-envs <integer>
--buffer-time <seconds>
--seed <integer>
--replicate <label>
--optimizer '<complete JSON object>'
--parameters '<complete JSON object>'
--loss '<complete JSON object>'
```

For example:

```bash
anchor-bench sysid \
  --robot g1 \
  --joints right_shoulder_pitch,right_elbow \
  --optimizer '{"type":"cmaes","sigma":0.3,"max_iterations":40,"epsilon":0.001}' \
  --loss '{"position":1.0,"velocity":0.25}' \
  --seed 7
```

The default initial state is `configs/robots/<robot>/sysid/default.yaml`. An
explicit state must be `sysid_state.yaml` from a completed, authenticated SysID
run. A run never updates accepted state.

## Neural actuator training

Preview or train one shared-weight GRU artifact:

```bash
anchor-bench train-actuator --robot so101 --dry-run
```

The profile defines one artifact. Each mapping defines one physical application
of those weights. Multiple mappings with the same dimensions pool training data
and normalization statistics; separate profiles produce separate artifacts.

Training overrides are:

```text
--sysid-state default|<completed-run>/sysid_state.yaml
--data <dataset-directory>
--sample-dt-s <seconds>
--mappings '<complete JSON list>'
--input-joints all|joint_a,joint_b [--output-joints all|joint_b]
--network '<complete JSON object>'
--target torque|torque_residual
--input-features '<complete JSON list>'
--fit '<complete JSON object>'
--skip-optuna|--no-skip-optuna
--trials <integer>
--trial-epochs <integer>
--final-epochs <integer>
--noise-std <float>
--seed <integer>
--replicate <label>
```

`--mappings` and `--input-joints` are mutually exclusive. The shorthand
`--input-joints` creates one mapping; omitted output joints default to its input
joints. `all` means one coupled all-joint mapping, never independent SISO
training.

Example MISO override:

```bash
anchor-bench train-actuator \
  --robot g1 \
  --input-joints right_shoulder_pitch,right_shoulder_roll,right_shoulder_yaw,right_elbow \
  --output-joints right_elbow \
  --target torque_residual \
  --input-features '[
    {"name":"position","domain":"input","components":["value"]},
    {"name":"position_error","domain":"input","components":["value"]},
    {"name":"velocity","domain":"input","components":["value"]},
    {"name":"dynamic_bias","domain":"input","components":["value"]},
    {"name":"previous_torque","domain":"output","components":["value"]}
  ]'
```

Actual training loads the runtime asset through pinned Newton to freeze output
effort limits. `dynamic_bias` additionally consumes
`SolverMuJoCo.eval_dynamic_bias()` during episode assembly. `previous_torque` is
the previous denormalized training target, with physical zero at every
episode/mapping boundary. Training writes a TorchScript archive with
self-contained metadata schema v3; deployment does not need the training SysID
snapshot.

## Benchmark

Benchmark one exact completed SysID or training run:

```bash
anchor-bench benchmark \
  --robot g1 \
  --run outputs/train/g1/default/seed-42/<run-input-sha256> \
  --profile default \
  --headless
```

Available overrides are:

```text
--data <held-out-dataset-directory>
--motion <motion-name>              # repeat for more motions
--simulation '<complete JSON object>'
--num-envs <integer>
--buffer-time <seconds>
--sync-initial-state|--no-sync-initial-state
--signals position,velocity
--output-root <directory>
--headless
--dry-run
```

The candidate directory is required. Benchmark does not read deployment state
or choose the currently accepted model. A SysID benchmark reconstructs the exact
candidate state from that run. A training benchmark authenticates `network.pt`
and uses its embedded schema-v3 contract. Timing and required-joint mismatches
fail before simulation.

Changing `--output-root` changes storage only, not benchmark identity. Repeating
an identical completed benchmark is a verified no-op. Position and velocity are
the supported comparable signals; torque is intentionally not exposed yet.

## Promotion

Promotion is always a separate command. It does not rerun SysID, training, or
benchmarking.

```bash
anchor-bench promote sysid \
  --robot g1 \
  --run outputs/sysid/g1/default/seed-42/<run-input-sha256> \
  [--benchmark-run <output-root>/g1/sysid/default/<source-hash>/<benchmark-hash>] \
  [--dry-run]

anchor-bench promote train \
  --robot g1 \
  --run outputs/train/g1/default/seed-42/<run-input-sha256> \
  [--benchmark-run <output-root>/g1/train/default/<source-hash>/<benchmark-hash>] \
  [--dry-run]

anchor-bench promote deploy \
  --robot g1 \
  [--profile default] \
  [--dry-run]
```

Repeat `--benchmark-run` to attach more than one exact evidence record.
Benchmark evidence is optional; promotion validates identity but does not apply
a metric threshold.

SysID promotion applies only the fitted parameter cells to
`sysid/default.yaml`. Training promotion adds or replaces the profile's model
entry in `train/default.yaml` and rejects output overlap with another accepted
model. Deploy promotion resolves a human-authored deployment profile against
both accepted states and atomically freezes `deploy/default.yaml`.

## Runtime requirements

- SysID execution requires the pinned Newton/Isaac runtime and CMA-ES.
- Training execution requires Torch and the pinned Newton runtime asset path to
  resolve neural effort limits. `dynamic_bias` additionally consumes the MuJoCo
  solver's `eval_dynamic_bias` result.
- Fresh benchmark replay requires the pinned Newton runtime; neural candidates
  additionally require Torch.
- Promotion and valid no-op detection are host-safe and do not start simulation.
- Frozen neural deployment requires a mapping-aware schema-v3 actuator factory;
  it is never silently routed through a schema-v1, equal-axis controller.

Paths accepted by the public CLI must remain inside the repository. Escaping
paths, cross-robot artifacts, corrupt manifests, and stale hashes are hard
errors.
