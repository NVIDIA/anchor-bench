# Dataset loading

`DatasetLoader` reads a local file, a local glob, a Hugging Face HTTP(S)
Parquet URL, or a structured Hugging Face source and linearly resamples each
experiment onto a uniform time grid:

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

## Hugging Face sources

A structured source selects one or more paths from a dataset repository at an
explicit revision:

```yaml
dataset:
  type: parquet
  source:
    provider: huggingface
    repo_id: nvidia/Anchor-Lab
    revision: be6010113e7b769e68f2779fbdf321908725f929
    paths:
      - data/g1_arm_multijoint/*.parquet
  resampling:
    type: uniform
    dt: 0.002
```

`paths` accepts repository-relative file paths and glob patterns. `*` matches
within one directory; `**` matches zero or more directories. For example,
`data/*.parquet` excludes nested directories, while `data/**/*.parquet` includes
them. The loader lists repository paths before downloading only the matching
files through the Hugging Face cache, orders them by repository path, and reuses
cached files on later runs. Pin a commit hash when the input data must be
reproducible.

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

## Feature-name mapping

Feature suffixes can be normalized without rewriting the source files:

```yaml
dataset:
  type: parquet
  source: recordings/*.parquet
  feature_name_mapping:
    dq: actual_dq
    tau_est: actual_effort
  resampling:
    type: uniform
    dt: 0.005
```

Feature mapping is partial and can be combined with joint-name mapping. The
loader rejects mappings that make two fields in the same experiment collide.

## Generated datasets

The [`trajectory_v1` task](trajectory.md) writes command positions in the same
long-form format. The `training` task consumes loaded signals and writes its
mapped sequences to `trajectories.parquet` with these columns:

`sequence`, `mapping`, `kind`, `experiment`, `time_ns`, `feature`, and `value`.
