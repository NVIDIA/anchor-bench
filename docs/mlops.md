# MLOps backends

Anchor Bench exposes one tracking interface for run parameters, scalar metrics,
tags, datasets, and local artifacts. Applications choose and construct the backend
at their integration boundary.

## Installation

Install the tracking SDK required by the backend implementation you use.

## Tracking a task

Tracking is opt-in for training and actuator system identification. When an
`mlops` block is present, its backend name is required:

```yaml
mlops:
  backend: custom
  experiment_name: actuator-training
  run_name: gru-seed-42
  options:
    endpoint: https://tracking.example
```

`TaskRunner` constructs the selected backend, starts the configured experiment,
logs the validated task configuration as flattened parameters, and binds the
active run to task execution. The task type, description, and configured tags
become searchable run tags.

Backend-specific `mlops.options` are forwarded to the backend constructor and
persisted in the local configuration for reproducibility, but are deliberately
excluded from logged parameters. Do not put credentials or access tokens in the
run YAML; use the backend SDK's supported environment variables or credential
store instead.

The backend context owns success and failure finalization. Anchor Bench writes
the local `.completed` marker only after task output validation succeeds and
removes it if task execution or backend finalization fails.

## Reporting from task code

The active run is available through `BaseTask.mlops_run` and the context-local
`get_active_run()` helper:

```python
def run(self, output_directory):
    if self.mlops_run is not None:
        self.mlops_run.log_metrics(
            {"validation/loss": validation_loss},
            step=epoch,
        )
        self.mlops_run.log_artifact(
            checkpoint_path,
            collection="training-checkpoints",
            artifact_type="model",
        )
```

Metric steps are run-global and should increase monotonically for portability
across backends. An artifact `collection` is a logical output group; backends
decide how to map collections into their artifact store.

## Dataset lineage

`DatasetLoader.load()` can receive an active run, a dataset name and context,
and an upload policy. It records the configured source, SHA-256 digest, file
size, raw row count and columns before resampling. Remote datasets are recorded
while their temporary materialization still exists, allowing explicitly enabled
payload uploads to complete before cleanup.

Dataset bytes are not uploaded by default. The portable
`MLOpsRun.log_dataset()` implementation always records lineage parameters and
logs the materialized file as a `dataset` artifact only when upload is enabled.
Backends can override the method to add native dataset relationships.

## Using a backend directly

Construct the backend directly:

```python
from my_anchor_backend import CustomBackend

backend = CustomBackend(endpoint="https://tracking.example")

with backend.start_run("actuator-training") as run:
    run.log_params({"hidden_size": 128, "seed": 42})
    run.log_metrics({"train/loss": 0.31}, step=1)
```

Do not retain a run and log to it after leaving its context.

## Adding another backend

Implement `MLOpsBackend` and `MLOpsRun`, then construct that backend at the
application integration boundary. Backend-specific constructor options remain
outside the portable interfaces.
