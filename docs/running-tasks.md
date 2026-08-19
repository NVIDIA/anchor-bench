# Running tasks

Anchor Bench runs one YAML task at a time:

```bash
uv run anchor-bench --config path/to/config.yaml
```

The top-level `type` selects the implementation. The registered types are
`actuator_sys_id`, `training`, and `trajectory_v1`. Configuration models are
strict: unknown fields, coerced values, invalid references, and unsupported
type values are rejected before the task starts.

Use `uv run anchor-bench --help` to inspect the CLI options.

## Logging

Task lifecycle and rate-limited progress logs are written to standard error at
the `INFO` level. Pass `--verbose` (`-v`) for per-step debug output or `--quiet`
(`-q`) to emit errors only. The two options are mutually exclusive:

```bash
uv run anchor-bench --verbose --config path/to/config.yaml
uv run anchor-bench --quiet --config path/to/config.yaml
```

## Output directories

By default, Anchor Bench writes to `output/<config-hash>/` at the repository
root. The hash is derived from the validated configuration, so the same
configuration resolves to the same directory.

Choose a directory explicitly with `--output`:

```bash
uv run anchor-bench \
  --config path/to/config.yaml \
  --output outputs/my-run
```

Some task schemas define an `output_dir`. When `--output` is omitted, that
directory replaces the default `output/` root but still receives a generated
hash subdirectory.

Every run persists its validated configuration as `config.yaml`. A successful
task also writes `.completed`. If that marker already exists, running the same
command returns the existing directory without executing the task again.

Use `--force` to rerun into that directory:

```bash
uv run anchor-bench \
  --config path/to/config.yaml \
  --output outputs/my-run \
  --force
```

The completion marker is removed before a forced run and is restored only
after the task and any configured tracking backend finish successfully.

## Run tracking

Tracking is opt-in. Tasks that accept an `mlops` block can log their validated
parameters, metrics, dataset lineage, and artifacts to MLflow or a custom
backend. See [MLOps backends](mlops.md).

## Next steps

- [Generate actuator command trajectories](trajectory.md)
- [Run actuator system identification](system-identification.md)
- [Train actuator models](training.md)
- [Load and resample Parquet datasets](datasets.md)
