# Dataset loading

`DatasetLoader` reads a local file, a local glob, or a Hugging Face HTTP(S)
Parquet URL and linearly resamples each experiment onto a uniform time grid:

```python
from anchor_bench.data import DatasetLoader

dataset = DatasetLoader.load("recording.parquet", dt=0.02)
```

The returned data frame uses `(experiment, time_ns)` as its index and one
column per signal field.

## Input contract

Input data must contain these long-form columns:

| Column | Meaning |
| --- | --- |
| `time_ns` | Sample timestamp in nanoseconds. |
| `experiment` | Sequence or experiment identifier. |
| `field` | Signal name, such as `hip/actual_q`. |
| `value` | Scalar signal value. |

Each field must have unique timestamps within an experiment. The loader
resamples over the time range shared by all fields in that experiment; it does
not extrapolate missing endpoints.

Task configurations express the same operation with a `dataset` block:

```yaml
dataset:
  type: parquet
  source: recordings/*.parquet
  resampling:
    type: uniform
    dt: 0.005
```

Only intersection-based time ranges and linear interpolation are currently
supported.

## Joint-name mapping

When Parquet and USD joint names differ, map each source field prefix to its USD
joint name:

```yaml
dataset:
  type: parquet
  source: recordings/*.parquet
  joint_name_mapping:
    recorded_hip: hip
    HIP: hip
    Hip: hip
  resampling:
    type: uniform
    dt: 0.005
```

The mapping is partial; unlisted prefixes remain unchanged. Mapping values are
the canonical names used inside Anchor Bench and must match references such as
`joint_parameters[].names`.

Multiple aliases may map to the same canonical name across files or
experiments. Within one experiment, two source aliases may not produce the same
canonical field. The loader reports that conflict rather than merging samples
silently.

## Generated datasets

The [`trajectory_v1` task](trajectory.md) writes command positions in the same
long-form format. The `training` task consumes loaded signals and writes its
mapped sequences to `trajectories.parquet` with these columns:

`sequence`, `mapping`, `kind`, `experiment`, `time_ns`, `feature`, and `value`.
