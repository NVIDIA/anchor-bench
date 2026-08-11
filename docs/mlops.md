# MLOps backends

Anchor Bench exposes one tracking interface for run parameters, scalar metrics,
tags, and local artifacts. Applications choose and construct backends at their
integration boundary.

## Installation

Install the tracking SDK required by the backend implementation you use.

## Tracking a run

Construct a backend directly for standalone tracking:

```python
from my_anchor_backend import CustomBackend

backend = CustomBackend(endpoint="https://tracking.example")

with backend.start_run(
    "actuator-training",
    run_name="gru-seed-42",
    tags={"robot": "so101"},
) as run:
    run.log_params({"hidden_size": 128, "seed": 42})
    run.log_metrics({"train/loss": 0.31}, step=1)
    run.log_artifact(
        "outputs/model.pt",
        collection="training-output",
        artifact_type="model",
    )
```

The run context owns backend completion and failure handling. Do not retain a
run and log to it after leaving the context.

Tracking is opt-in for both training and actuator system identification. When an
`mlops` block is present, its backend name is required:

```yaml
mlops:
  backend: custom
  experiment_name: actuator-training
  run_name: gru-seed-42
  upload_dataset: true
  options:
    endpoint: https://tracking.example
```

### Dataset inputs

Training and system-identification runs automatically log their source Parquet
file as a dataset input. Tracking always records the configured source, a
SHA-256 digest of the acquired file, its size, raw row count and columns, and
the dataset context. Resampling and preprocessing remain part of the flattened
task configuration, so the source fingerprint and transformations can be
inspected together.

Dataset bytes are not uploaded by default. Enable payload upload explicitly
when the tracking store is an appropriate home for the source data:

```yaml
mlops:
  backend: custom
  experiment_name: actuator-training
  upload_dataset: true
```

Backends receive the same portable dataset parameters. Upload-enabled runs also
log the Parquet file as an artifact with type `dataset`. The upload occurs
while remote downloads are still materialized, before their temporary directory
is removed.

`TaskRunner` constructs the selected backend automatically before calling the
task. It logs the validated task configuration as flattened parameters, starts
the configured experiment, and binds the active run to the task:

```python
def run(self, output_directory):
    # Report exactly where the metric or artifact is produced.
    mlops_run = get_active_run()
    if mlops_run is not None:
        mlops_run.log_metrics(
            {"validation/loss": validation_loss},
            step=epoch,
        )
        mlops_run.log_artifact(
            checkpoint_path,
            collection="training-checkpoints",
            artifact_type="model",
        )
```

Metric steps are run-global and must increase monotonically for portability
across backends. During hyperparameter search, Anchor Bench reports one
`search/validation_loss` value per trial and suppresses the individual trial
epoch curves; this prevents different trials from reusing the same run steps.

An artifact `collection` is a logical output group. Backends decide how to
map collections into their artifact store. Files and directories
must exist before they are logged. Missing or failed artifact uploads fail the
task, so the local `.completed` marker is not written for an incompletely
tracked run.

Backend-specific `mlops.options` are used to construct the backend but are
deliberately excluded from logged parameters. Do not put credentials, access
tokens, or other secrets in the run YAML, including under `mlops.options`.
Anchor Bench persists the validated configuration in the output directory, and
successful tasks upload that directory as a run artifact. Configure credentials
through the backend SDK's supported environment variables or credential store
instead. The backend context records successful or failed completion and the
active run is cleared even if the task raises an exception.

`mlops.upload_dataset` defaults to `false`. Set it to `true` to upload dataset
bytes in addition to recording lineage. The portable
`MLOpsRun.log_dataset()` implementation always records lineage parameters and
logs the materialized file as a `dataset` artifact only when upload is enabled.
Backends can override the method to add native dataset relationships.

The training task reports per-epoch training and validation loss, search and
validator metrics, interval checkpoints, and the completed `training-output`
artifact collection. Interval checkpoints remain useful if a later training or
validation stage fails; successful runs also include them in the final output
directory.

## System-identification runs

System identification benefits from the same tracking abstraction: controller
bounds, optimizer settings, simulation settings, dataset identity, and random
seed are run parameters; per-generation MSE values are metrics; and the fitted
parameters, optimization history, summary, and identified USD layer are
artifacts. Enable it explicitly:

```yaml
type: actuator_sys_id
description: Identify the SO-101 arm actuators.
tags: [so101, baseline]
mlops:
  backend: custom
  experiment_name: actuator-identification
  run_name: so101-baseline
  options:
    endpoint: https://tracking.example
```

The task reports `optimization/best_mse`,
`optimization/iteration_best_mse`, and
`optimization/population_mean_mse` for every generation. Its completed local
output directory is logged in the `sysid-output` collection with artifact type
`actuator-sysid`.

## Adding another backend

Implement `MLOpsBackend` and `MLOpsRun`, then publish the factory as an entry
point. No Anchor Bench registry change is required:

```toml
[project.entry-points."anchor_bench.mlops_backends"]
custom = "my_anchor_plugin:CustomBackend"
```

The entry-point object must be a callable that accepts backend-specific keyword
options and returns an `MLOpsBackend`. For application-local integrations, the
same factory can be registered at runtime:

```python
from anchor_bench.mlops import register_backend

register_backend("custom", CustomBackend)
```

Backend names are case-insensitive. Duplicate names are rejected so an
installed plugin cannot silently replace another integration.
