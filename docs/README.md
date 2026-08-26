# Anchor Bench documentation

These guides document the interfaces implemented by the current codebase.

## Start here

- [Project overview and installation](../README.md)
- [Running tasks](running-tasks.md): invoke the CLI, select an output directory,
  configure local or revision-pinned Newton assets, and understand completion
  markers.

## Task guides

- [Trajectory generation](trajectory.md): configure hold, transition, sine, and
  chirp pieces. A complete configuration is available in
  [`examples/so101_motion_pack.yaml`](../examples/so101_motion_pack.yaml).
- [Actuator system identification](system-identification.md): configure a
  Newton replay and inspect its fitted parameters and optimization artifacts.
- [Training actuator models](training.md): reproduce the maintained G1 GPU
  training and Newton replay workflows.

## Data and integrations

- [Dataset loading](datasets.md): long-form Parquet requirements, resampling,
  and joint-name mapping.
- [MLOps backends](mlops.md): enable MLflow tracking or inject a custom backend.
- [Newton model loading](../README.md#python-api): construct a Newton model and
  initial state from USD.
