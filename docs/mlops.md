# MLOps backends

Anchor Bench exposes one tracking interface for run parameters, scalar metrics,
tags, datasets, and local artifacts. Backend plugins are discovered through the
`anchor_bench.mlops_backends` Python entry-point group and selected explicitly
by name.

## Installation

Install the plugin and tracking SDK for the backend you use. Importing
`anchor_bench.mlops` does not import a backend-specific SDK; the selected plugin
owns its optional dependencies.

## Tracking a run

`get_backend()` selects an installed or runtime-registered plugin by name:

```python
from anchor_bench.mlops import get_backend

backend = get_backend("custom", endpoint="https://tracking.example")

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

## Portable data and artifacts

`MLOpsRun.log_dataset()` records portable lineage parameters including the
source, digest, shape, column names, and optional metadata. When upload is
enabled, its default implementation also logs the materialized dataset through
`log_artifact()` with artifact type `dataset`. Backends can override this method
to add native dataset relationships while retaining the portable fields.

An artifact `collection` is a logical output group. Backends decide how to map
collections into their artifact store. Files and directories must exist before
they are logged.

Metric steps are run-global and should increase monotonically for portability
across backends. Backend implementations must propagate failures so callers can
keep their local completion state consistent with the tracking service.

Backend constructor options can contain service endpoints and other
backend-specific settings. Do not put credentials or access tokens in persisted
configuration; use the backend SDK's supported environment variables or
credential store instead.

## Reporting helpers

`flatten_params()` converts nested JSON-like configuration into portable scalar
parameters. `bind_active_run()` and `get_active_run()` provide a context-local
run binding for application code that cannot conveniently receive a run as an
explicit argument.

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
