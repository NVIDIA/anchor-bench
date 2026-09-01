# Trajectory generation

The `trajectory_v1` task generates deterministic commanded actuator positions
from ordered hold, transition, sine, and chirp pieces. It does not load a robot
model or run a simulation.

## Run the included example

From the repository root, run the example into a named output directory:

```bash
uv run anchor-bench \
  --config examples/trajectory/so101_motion_pack.yaml \
  --output outputs/so101_motion_pack
```

A successful run creates:

```text
outputs/so101_motion_pack/
  config.yaml
  trajectories.parquet
  .completed
```

`config.yaml` and `.completed` follow the common task artifact contract. See
[Running tasks](running-tasks.md) for default output paths and rerun behavior.

## Configuration

A trajectory configuration has this top-level shape:

```yaml
type: trajectory_v1
name: two_joint_example
sample_rate_hz: 100

actuators:
  - {name: shoulder, unit: rad}
  - {name: gripper, unit: normalized}

poses:
  home: [0.0, 0.5]
  reach: [0.8, 0.25]

trajectories:
  - name: reach_and_wave
    initial_pose: home
    pieces:
      - {type: hold, duration_s: 0.5}
      - type: transition
        duration_s: 1.0
        to: reach
        interpolation: quintic_smoothstep
      - type: sine
        duration_s: 2.0
        center: reach
        ramp_s: 0.25
        components:
          - actuator: shoulder
            amplitude: 0.1
            frequency_hz: 0.5
            phase_rad: 0.0
      - {type: hold, duration_s: 0.5, pose: home}
```

The configuration is strict: unknown fields and invalid references are errors.
The important rules are:

- `sample_rate_hz` is a positive integer.
- Actuator names and trajectory names are unique.
- Every pose array follows `actuators` order and contains exactly one value per
  actuator.
- `poses` must define `home`, even if another pose is used initially.
- `initial_pose`, hold `pose`, transition `to`, and oscillation `center` may be
  a named pose or an inline pose array.
- Every duration is positive. Frequencies and `ramp_s` values are nonnegative.
- Sine and chirp components must reference configured actuator names.

Each piece supports the following fields:

| Piece | Required fields | Optional fields |
| --- | --- | --- |
| `hold` | `duration_s` | `pose`; otherwise hold the current pose |
| `transition` | `duration_s`, `to` | `interpolation`: `linear` or the default `quintic_smoothstep` |
| `sine` | `duration_s`, nonempty `components` | `center`, `ramp_s` (default `1.0`); each component accepts `actuator`, `amplitude`, `frequency_hz`, and optional `phase_rad` |
| `chirp` | `duration_s`, nonempty `components` | `center`, `ramp_s` (default `1.0`); each component accepts `actuator`, `amplitude`, `start_frequency_hz`, `end_frequency_hz`, and optional `phase_rad` |

Pieces are evaluated in order. A piece without an explicit pose or center starts
from the current pose left by the preceding piece.

## `trajectories.parquet` contract

The generated Parquet file is long-form and contains exactly these columns:

| Column | Meaning |
| --- | --- |
| `time_ns` | Synthetic nanoseconds from zero at the configured sample rate |
| `time_utc` | UTC timestamp representation of `time_ns` |
| `experiment` | Trajectory `name` |
| `field` | `<actuator>/command_q` |
| `value` | Commanded scalar actuator position |

There is one row per sample and actuator. Fields appear in configured actuator
order, and the file contains command positions only—no measured state, effort,
or temperature fields. Each configured trajectory starts its own synthetic time
at zero.

The artifact can be loaded through the package's dataset loader when a wide,
indexed representation is needed:

```python
from anchor_bench.data import DatasetLoader

dataset = DatasetLoader.load(
    "outputs/so101_motion_pack/trajectories.parquet",
    dt=0.02,
)
```

Choose `dt` to match the generated period (`1 / sample_rate_hz`) when no further
resampling is desired.
