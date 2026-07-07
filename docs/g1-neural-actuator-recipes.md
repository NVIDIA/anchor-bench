# G1 Neural Actuator Recipes

Anchor Bench keeps G1 neural actuator support declarative and artifact-light.
Do not commit G1 `.pt`, `.pth`, `.onnx`, raw datasets, stats, or generated
benchmark outputs. Store generated runs under `outputs/training/g1/...`.

The configured G1 robot contains the four right-arm joints:

```text
right_shoulder_pitch
right_shoulder_roll
right_shoulder_yaw
right_elbow
```

Use those canonical names in artifact metadata. The shorter aliases from
`robot.yaml` are accepted on the CLI and resolved before training.

## Full-Torque Arm MIMO

This enriched full-torque model uses five four-joint feature blocks plus the
previous four-joint output:

```text
[position, position_error, velocity, solver_pd, dynamic_bias, previous_torque]
```

That is a 24-value input and a four-value torque output. The feature name is
`dynamic_bias`; `qfrc_bias` is the MuJoCo quantity used to derive it, not a
training-column name.

```bash
anchor-bench train-actuator \
  --robot g1 \
  --model gru \
  --data <g1-sage-motion-or-dataset-root> \
  --input-joints right_shoulder_pitch,right_shoulder_roll,right_shoulder_yaw,right_elbow \
  --output-joints right_shoulder_pitch,right_shoulder_roll,right_shoulder_yaw,right_elbow \
  --input-columns position,position_error,velocity,solver_pd,dynamic_bias,previous_torque \
  --pd-kp <shared-training-kp> \
  --pd-kd <shared-training-kd> \
  --target-column torque \
  --output outputs/training/g1/right_arm_mimo_full_torque \
  --skip-optuna \
  --dry-run
```

Remove `--dry-run` only in the full runtime environment. Dynamic-bias
generation resolves `g1.usd` through the normal external-asset search path and
calls `SolverMuJoCo.eval_dynamic_bias`. The USD is not stored in this
repository.

A full-torque artifact predicts total effort. A consumer must apply full-torque
actuator semantics rather than adding that output to an implicit PD result.

## Arm MISO Variants

Keep the same four input joints and select one output joint to train a MISO
artifact. For example:

```text
--output-joints right_elbow
```

With the same six features, this contract has 21 inputs: five blocks of four
joint values and one previous elbow torque. A SISO artifact uses one input and
one output joint. Ordering is part of the artifact contract, and every output
must also appear in the input list.

## Hybrid Residual Arm GRU

Residual training subtracts the shared PD and friction baselines from measured
torque before fitting the network. The previous-torque feature is shifted after
that subtraction, so it contains the previous residual target and starts at
physical zero for every episode.

```bash
anchor-bench train-actuator \
  --robot g1 \
  --model hybrid_residual \
  --data <g1-sage-motion-or-dataset-root> \
  --input-joints right_shoulder_pitch,right_shoulder_roll,right_shoulder_yaw,right_elbow \
  --output-joints right_shoulder_pitch,right_shoulder_roll,right_shoulder_yaw,right_elbow \
  --input-columns position,position_error,velocity,dynamic_bias,previous_torque \
  --pd-kp <shared-training-kp> \
  --pd-kd <shared-training-kd> \
  --friction-dry <shared-training-dry-friction> \
  --friction-viscous <shared-training-viscous-friction> \
  --target-column torque \
  --output outputs/training/g1/right_arm_mimo_residual \
  --skip-optuna \
  --dry-run
```

Each network stores one copy of each baseline. G1's current
`actuators.yaml` has different fitted values for its arm joints, so coupled
baseline inference intentionally fails. Supply the exact shared values used by
the real-world controller during data collection through the CLI or
`training.pd_baseline` and `training.friction_baseline` in `run.yaml`.
Do not average per-joint values silently.

At runtime, residual output is added as feed-forward effort while the matching
implicit PD/friction baseline remains active.

## Data and Artifact Checks

G1's configured control rate is 500 Hz, so current recipes expect state samples
every 0.002 seconds. Training fails on a different or irregular cadence; it does
not resample the trajectory. Command positions remain linearly interpolated
onto the state timestamps.

Every configured input joint must be present in each SAGE trajectory. Other
recorded joints are included when deriving dynamic bias if they map into the
Newton model. Unrecorded non-input model joints use model-default position and
zero velocity.

The exported TorchScript file carries its own `metadata.json`. Check it for:

- ordered `joint_names` and `output_joint_names`;
- `input_columns` and the single `torque` or `torque_residual` semantic;
- `sample_dt_s: 0.002`;
- per-feature and per-output normalization;
- the shared baselines required by the model;
- dynamic-bias and previous-target derivation records.

The raw checkpoint and stats JSON remain useful for training diagnostics, but a
runtime consumer should need only the TorchScript archive. Anchor Bench does
not update Newton as part of this workflow; use a runtime that supports the
exported metadata and feature contract.

## Leg Experiments

The current declarative G1 plugin covers only the right arm. Add any leg joints
and their runtime asset mapping to the robot plugin before adapting these
recipes. Keep leg actuator comparisons experimental until they have their own
validated data and configuration.
