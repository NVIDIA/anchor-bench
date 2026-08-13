# MLOps backends

Anchor Bench exposes one tracking interface for run parameters, scalar metrics,
tags, and local artifacts. MLflow is a bundled backend included in every task
runner by default. Applications can inject custom backends per runner.

## Installation

The default Anchor Bench installation includes the MLflow SDK and imports the
MLflow backend when `anchor_bench.mlops` is imported.

## Tracking a run

Construct the bundled MLflow backend directly for standalone tracking:

```python
from anchor_bench.mlops import MLflowBackend

backend = MLflowBackend(tracking_uri="sqlite:///outputs/mlflow.db")

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
`mlops` block is present, MLflow is the default backend:

```yaml
mlops:
  backend: mlflow
  experiment_name: actuator-training
  run_name: gru-seed-42
  upload_dataset: true
  options:
    # Tracking Server or backend store for experiment and run metadata.
    tracking_uri: sqlite:///outputs/mlflow.db
    # Optional separate model-registry backend. When omitted, MLflow keeps its
    # current registry setting, which normally follows the tracking URI.
    # registry_uri: sqlite:///outputs/registry.db
    # Per-attempt timeout in seconds for MLflow HTTP operations.
    # http_request_timeout: 30
    # Number of retries after the initial MLflow HTTP request.
    # http_request_max_retries: 2
```

URI overrides apply only for the lifetime of the Anchor Bench run. The MLflow
backend restores the SDK's previous tracking and registry URIs when the run
finishes or fails, so sequential runs do not inherit each other's stores.

Anchor Bench defaults MLflow HTTP operations to a 30-second timeout and two
retries, reducing the wait for an unreachable server from MLflow's much longer
SDK defaults. Existing `MLFLOW_HTTP_REQUEST_TIMEOUT` and
`MLFLOW_HTTP_REQUEST_MAX_RETRIES` environment values are honored unless the
corresponding options above are set explicitly. The environment is restored
after each run. Increase the timeout when a remote service or artifact store
legitimately needs longer than 30 seconds without making progress.

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
  backend: mlflow
  experiment_name: actuator-training
  upload_dataset: true
```

MLflow records native dataset-input lineage and, when upload is enabled, stores
the Parquet file as a dataset artifact. Because MLflow limits its native dataset
digest to 36 characters, that field contains a prefix of the SHA-256; the full
digest remains available in the portable dataset parameters and the
`anchor_bench.dataset_digest` input tag. Third-party backends receive the same
portable dataset parameters; upload-enabled runs additionally log an artifact
with type `dataset`. The upload occurs while remote
downloads are still materialized, before their temporary directory is removed.

For local inspection, start the UI against the same store:

```bash
uv run mlflow ui --backend-store-uri sqlite:///outputs/mlflow.db --port 5000
```

`TaskRunner` constructs the selected backend automatically before calling the
task. It logs the validated task configuration as flattened parameters, starts
the configured experiment, and binds the active run to the task:

```python
from anchor_bench.mlops import get_active_run


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

An artifact `collection` is a logical output group. MLflow maps it to a
subdirectory beneath the run artifact root. Files and directories
must exist before they are logged. Missing or failed artifact uploads fail the
task, so the local `.completed` marker is not written for an incompletely
tracked run.

Backend-specific `mlops.options` are used to construct the backend and are
logged alongside the rest of the flattened task configuration. Anchor Bench
does not filter sensitive values from YAML configuration. It persists the
validated configuration in the output directory, and successful tasks upload
that directory as a run artifact. Users are responsible for ensuring their
configuration does not disclose values they do not want stored by the selected
backend. The backend context records successful or failed completion and the
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
  backend: mlflow
  experiment_name: actuator-identification
  run_name: so101-baseline
  options:
    tracking_uri: sqlite:///outputs/mlflow.db
```

The task reports `optimization/best_mse`,
`optimization/iteration_best_mse`, and
`optimization/population_mean_mse` for every generation. Its completed local
output directory is logged in the `sysid-output` collection with artifact type
`actuator-sysid`.

## Bundled backend

The bundled MLflow adapter is included in every task runner and is
selected by default.

## Adding another backend

Implement `MLOpsBackend` and `MLOpsRun`, then pass the backend factory to the
runner. The factory receives backend-specific keyword options from
`mlops.options` and must return an `MLOpsBackend`:

```python
from anchor_bench.core.task_runner import TaskRunner
from my_anchor_backend import CustomBackend

runner = TaskRunner(mlops_backends={"custom": CustomBackend})
runner.run("config.yaml")
```

Custom mappings extend the bundled defaults for that runner only. Passing the
name `mlflow` intentionally overrides the bundled backend for that runner.
