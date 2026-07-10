# G1 neural actuator recipes

G1 neural actuator experiments are training profiles. One profile produces one
immutable artifact, one weight set, and one pooled set of normalization
statistics. The profile filename becomes its accepted model key when promoted.

Keep G1 datasets, checkpoints, TorchScript archives, benchmark output, and plots
under `inputs/` or `outputs/` as appropriate; do not commit generated weights or
runs.

## Checked-in baseline

`configs/robots/g1/train/profiles/default.yaml` is a one-joint residual SISO
baseline for `right_shoulder_pitch`. It uses:

```text
features: position, position_error, velocity
target:   torque_residual
mapping:  1 input joint -> 1 output joint
```

It is deliberately small. To train another independent SISO model, create a
separate named profile. To share one scalar model across several physical
joints, put several 1-to-1 mappings of identical shape in one profile.

## Coupled right-arm recipes

For the four configured right-arm joints, one mapping determines topology:

- MISO: four input joints and one output joint;
- MIMO: four input joints and four output joints.

Example MIMO mapping:

```yaml
mappings:
  - input_joints:
      - right_shoulder_pitch
      - right_shoulder_roll
      - right_shoulder_yaw
      - right_elbow
    output_joints:
      - right_shoulder_pitch
      - right_shoulder_roll
      - right_shoulder_yaw
      - right_elbow
```

Example MISO mapping:

```yaml
mappings:
  - input_joints:
      - right_shoulder_pitch
      - right_shoulder_roll
      - right_shoulder_yaw
      - right_elbow
    output_joints: [right_elbow]
```

Outputs must be a subset of inputs. Runtime tensor dimensions come from this one
mapping, not the number of physical mappings in the profile.

## Enriched feature contract

The supported enriched feature family is:

```yaml
input_features:
  - {name: position, domain: input, components: [value]}
  - {name: position_error, domain: input, components: [value]}
  - {name: velocity, domain: input, components: [value]}
  - {name: solver_pd, domain: input, components: [value]}
  - {name: dynamic_bias, domain: input, components: [value]}
  - {name: previous_torque, domain: output, components: [value]}
```

`dynamic_bias` is Newton/MuJoCo `qfrc_bias` evaluated from synchronized position
and velocity without stepping the simulation. Anchor Bench calls
`SolverMuJoCo.eval_dynamic_bias()` lazily when the feature is requested.

`previous_torque` lives on the output axis. It is the previous denormalized
target with physical zero at the start of each recorded-episode/mapping
sequence. For a residual model it therefore means previous residual torque.

With five scalar input-domain blocks and `previous_torque`, the flattened width
is:

```text
D = 5 * I + O
```

The four-joint MIMO recipe has `D = 24`; the four-to-one MISO recipe has
`D = 21`. Feature order in YAML is runtime order and is frozen in schema-v3
metadata.

## Full-torque versus residual

Use `target: torque` when the network should own total output torque. Deployment
disables solver-native PD on those output joints and clamps total network
output.

Use `target: torque_residual` when the network should learn:

```text
measured torque - PD baseline - friction baseline
```

The selected SysID state supplies per-joint stiffness, damping, dry friction,
and viscous friction. Those mapping-specific values are embedded in the
artifact and remain authoritative after promotion. At runtime the frozen
baseline and network residual are added, while only the residual is subject to
the neural clamp.

Neural per-output effort limits come from the training-time G1 asset and are
also frozen in metadata. A deploy profile cannot replace either the residual
baseline or neural effort limits.

## Run, benchmark, and promote

Create a named profile such as
`configs/robots/g1/train/profiles/right_arm_mimo.yaml`, then preview it:

```bash
anchor-bench train-actuator --robot g1 --profile right_arm_mimo --dry-run
```

Actual training may use accepted SysID values or a specific unpromoted SysID
candidate:

```bash
anchor-bench train-actuator \
  --robot g1 \
  --profile right_arm_mimo \
  --sysid-state outputs/sysid/g1/default/seed-42/<hash>/sysid_state.yaml
```

Benchmark and promotion always name the exact completed training run:

```bash
anchor-bench benchmark \
  --robot g1 \
  --run outputs/train/g1/right_arm_mimo/seed-42/<run-input-sha256> \
  --headless

anchor-bench promote train \
  --robot g1 \
  --run outputs/train/g1/right_arm_mimo/seed-42/<run-input-sha256>
```

Training promotion adds the `right_arm_mimo` model key to
`train/default.yaml`. It fails if another accepted model already owns any of the
same output joints.

## Deployment ownership

A neural G1 deploy profile must select the accepted mapping index and repeat its
exact axes:

```yaml
composition:
  neural:
    - name: right_arm_network
      model: right_arm_mimo
      mappings:
        - index: 0
          input_joints:
            - right_shoulder_pitch
            - right_shoulder_roll
            - right_shoulder_yaw
            - right_elbow
          output_joints:
            - right_shoulder_pitch
            - right_shoulder_roll
            - right_shoulder_yaw
            - right_elbow
  pd: []
```

PD and neural actuators may coexist only on disjoint output joints. Context
inputs may overlap. Omitted joints retain asset-defined behavior. Freeze the
composition only after the model is accepted:

```bash
anchor-bench promote deploy --robot g1 --profile <deploy-profile> --dry-run
anchor-bench promote deploy --robot g1 --profile <deploy-profile>
```

The network's `sample_dt_s` must equal `1 / control_freq`; deployment does not
resample recurrent inference.
