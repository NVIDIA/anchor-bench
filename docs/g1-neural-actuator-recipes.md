# G1 Neural Actuator Recipes

Anchor Bench keeps G1 neural actuator support declarative and artifact-light.
Do not commit G1 `.pt`, `.pth`, `.onnx`, raw datasets, or generated benchmark
outputs. Store generated runs under `outputs/training/g1/...`; promote only
small non-weight metadata under `inputs/actuator_models/g1/`.

## Production Arm Full-Torque Enriched GRU

```text
[q, position_error, velocity, SDK_PD_hint, qfrc_bias, previous_torque]
```

For the four right-arm joints this is a 24-input recurrent model. The actuator
YAML disables solver PD, so the network predicts total torque. This path should
remain a documented recipe until the richer feature-generation trainer source is
available; do not reconstruct it from checkpoints or stats artifacts alone.

## Arm Hybrid Residual GRU

The native v1 `train-actuator --model hybrid_residual` path matches the hybrid
idea: keep the native solver PD/friction actuator active, compute residual torque
by subtracting those terms from measured torque, and train a GRU on
`[position, position_error, velocity]` to add the remaining correction.
The runtime side should use the shared GRU network with residual actuator
semantics: preserve position targets for implicit PD and add the predicted
`torque_residual` as feed-forward effort.

```bash
anchor-bench train-actuator \
  --robot g1 \
  --model hybrid_residual \
  --data <g1-arm-training-csv-dir> \
  --joint-type right_shoulder_pitch \
  --output outputs/training/g1/right_shoulder_pitch \
  --input-columns position,position_error,velocity \
  --target-column torque \
  --skip-optuna
```

## Leg Full-Torque GRU Experiment

The G1 leg full-torque enriched experiment uses the same enriched family as the
arm path, with six leg joints and a 36-input contract. It mechanically works,
but should remain secondary to the lag10 SysID actuator recipe because the branch
results showed position and velocity regressions even when torque improved
slightly. Treat this as an experimental benchmark comparison, not the default G1
leg actuator path.
