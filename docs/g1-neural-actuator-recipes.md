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

General topology, feature, timing, and artifact semantics are documented in the
[CLI guide](cli.md#neural-actuator-training).

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
  --output outputs/training/g1/right_arm_mimo_full_torque \
  --skip-optuna \
  --dry-run
```

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

G1 runs at 500 Hz, so these recipes require `sample_dt_s: 0.002`. Every
configured input joint must be present in each SAGE trajectory. See
[the CLI guide](cli.md#neural-actuator-training) for cadence validation,
dynamic-bias state completion, and the exported metadata contract.

## Leg Experiments

The current declarative G1 plugin covers only the right arm. Add any leg joints
and their runtime asset mapping to the robot plugin before adapting these
recipes. Keep leg actuator comparisons experimental until they have their own
validated data and configuration.
