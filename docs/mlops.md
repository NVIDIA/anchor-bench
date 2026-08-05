# MLOps backends

Anchor Bench exposes one tracking interface for run parameters, scalar metrics,
tags, and local artifacts. MLflow is the default backend. W&B is an optional
plugin discovered through the `anchor_bench.mlops_backends` Python entry-point
group.

## Installation

Install only the SDK used by the current environment:

```bash
uv sync --extra mlflow
```

or:

```bash
uv sync --extra wandb
```

Importing `anchor_bench.mlops` does not import either SDK. The selected SDK is
loaded when its backend is constructed, and a missing extra produces an
installation-specific error.

## Tracking a run

`get_backend()` selects MLflow when no name is passed:

```python
from anchor_bench.mlops import get_backend

backend = get_backend(tracking_uri="file:./mlruns")

with backend.start_run(
    "actuator-training",
    run_name="gru-seed-42",
    tags={"robot": "so101"},
) as run:
    run.log_params({"hidden_size": 128, "seed": 42})
    run.log_metrics({"train/loss": 0.31}, step=1)
    run.log_artifact("outputs/model.pt", artifact_path="models")
```

The run context owns backend completion and failure handling. Do not retain a
run and log to it after leaving the context.

Training configuration has the same default:

```yaml
mlops:
  backend: mlflow
  experiment_name: actuator-training
  options:
    tracking_uri: file:./mlruns
```

`TaskRunner` constructs the selected backend automatically before calling the
task. It logs the validated task configuration as flattened parameters, starts
the configured experiment, and binds the active run to the task:

```python
def run(self):
    # Report exactly where the metric or artifact is produced.
    if self.mlops_run is not None:
        self.mlops_run.log_metrics(
            {"validation/loss": validation_loss},
            step=epoch,
        )
        self.mlops_run.log_artifact(
            checkpoint_path,
            artifact_path="models",
        )
```

Backend-specific `mlops.options` are used to construct the backend but are
deliberately excluded from logged parameters because they can contain access
tokens. The backend context records successful or failed completion and the
active run is cleared even if the task raises an exception.

The current training module only prepares data and is not registered as an
executable task yet. Consequently, automatic epoch metrics and checkpoints will
begin when that training loop is implemented; the runner integration and
task-local reporting path are already in place.

## W&B plugin

The bundled W&B adapter is registered as an entry-point plugin rather than in
the core backend registry. Select it entirely through configuration:

```yaml
mlops:
  backend: wandb
  experiment_name: actuator-training
  options:
    entity: robotics-team
    mode: online
```

MLflow experiment names map to W&B project names. Anchor Bench's key/value tags
are encoded as `key:value` W&B tags.

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
installed plugin cannot silently replace the default or another integration.
