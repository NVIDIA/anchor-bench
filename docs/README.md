# Anchor Bench documentation

These pages describe the configuration-driven Anchor Bench package on the
current `main` branch.

## Supported guides

- [Project README](../README.md): installation, the `anchor-bench` command,
  registered task types, dataset loading, Newton model loading, and development
  checks.
- [Trajectory generation](trajectory.md): configure and run a `trajectory_v1`
  task and consume its `trajectories.parquet` artifact.
- [MLOps backends](mlops.md): configure MLflow tracking or inject a custom
  backend into `TaskRunner`.

The complete trajectory example is
[`examples/so101_motion_pack.yaml`](../examples/so101_motion_pack.yaml).
Integration-specific G1 training fixtures are documented beside their tests in
[`tests/integration/g1/README.md`](../tests/integration/g1/README.md).

## Archived pages

The following filenames are retained as archive markers because earlier
revisions linked to them. They do not describe supported interfaces:

- [Artifact layout](artifacts.md)
- [Legacy command-line interface](cli.md)
- [Legacy end-to-end tests](end-to-end-test.md)
- [Legacy G1 actuator recipes](g1-neural-actuator-recipes.md)
- [Legacy repository layout](repo-structure.md)
- [Legacy robot plugin system](robot-plugin.md)

Use the supported guides above for executable commands. Historical contents of
an archived page remain available through Git history.
