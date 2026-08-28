# Anchor Bench documentation

These guides describe the supported command-line workflows and Python
interfaces. Run commands from the repository root unless a guide says
otherwise.

## Start here

- [Project overview and installation](../README.md): install Anchor Bench and
  run the included example.
- [Running tasks](running-tasks.md): invoke the CLI, select an output directory,
  and understand completion markers.

## Task guides

- [Trajectory generation](trajectory.md): configure hold, transition, sine, and
  chirp pieces. A complete configuration is available in
  [`examples/trajectory/so101_motion_pack.yaml`](../examples/trajectory/so101_motion_pack.yaml).
- [Actuator system identification](system-identification.md): configure a
  Newton replay and inspect its fitted parameters and optimization artifacts.
  The maintained SO-101 configuration is
  [`examples/system_identification/so101.yaml`](../examples/system_identification/so101.yaml).
- [Training actuator models](training.md): reproduce the maintained G1 GPU
  training and Newton replay workflows from
  [`examples/training/`](../examples/training/).

## Data and integrations

- [Dataset loading](datasets.md): long-form Parquet requirements, resampling,
  and joint-name mapping.
- [MLOps backends](mlops.md): enable MLflow tracking or inject a custom backend.
- [Python API](../README.md#python-api): load Parquet data or construct a Newton
  model and initial state from USD.
