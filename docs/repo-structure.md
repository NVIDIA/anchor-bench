# Repository Structure

Anchor Bench separates robot definitions, runtime assets, run inputs, generated
outputs, and third-party dependencies. Keeping those boundaries clear makes runs
reproducible and keeps generated or private files out of robot config.

```text
anchor_bench/          Python package: CLI, loaders, runners, runtime helpers
configs/
  robots/<robot>/      Declarative robot defaults: joints, actuators, bounds, run defaults
  experiments/         Reusable experiment recipes that reference inputs and outputs
assets/robots/         Robot asset manifests plus the SO-101 quick-check USD
inputs/                Measured datasets, curated params, and non-weight actuator metadata
outputs/               Generated SysID, training, benchmark, and analysis artifacts
third_party/           Ignored local artifacts and optional vendor checkouts
docker/                Container entrypoints and GUI/headless/persistent run helpers
docs/                  CLI, plugin, artifact, and test guidance
examples/              Small smoke-test data/results
tests/                 Host-safe unit tests and dry-run command checks
versions/              Runtime compatibility pins
```

## Configs

`configs/` contains declarative robot and experiment configuration. These files
describe what a robot is and how a workflow should be assembled; they should not
contain measured datasets, generated results, or private/binary assets.

```text
configs/
  robots/
    so101/
      robot.yaml        # robot identity, asset manifest, joint names, aliases
      actuators.yaml    # canonical/default actuator model
      bounds.yaml       # SysID parameter search bounds
      run.yaml          # default workflow settings for this robot
  experiments/
    so101_anchor_lab_figure_eight.yaml
```

Use `robot.yaml` as the single source of truth for supported joint names and
aliases. Use `configs/robots/<robot>/actuators.yaml` as the canonical/default
actuator model for that robot. Use `configs/robots/<robot>/bounds.yaml` to
define what SysID is allowed to tune. `configs/experiments/*.yaml` should
reference `configs/robots/...` for robot defaults, `inputs/datasets/...` for
measured data, and `outputs/...` for generated artifacts.

## Assets

SO-101 includes `assets/robots/so101/files/so101.usd` for quick checks.
For other robots, provide USD/MJCF/mesh files either by placing them under
`assets/robots/<robot>/files/` or by setting `ANCHOR_BENCH_ASSET_DIR` to a
directory containing the assets. Use `ANCHOR_BENCH_ASSET_DIR` when assets are
local, private, large, or shared across repos.

Asset lookup checks `ANCHOR_BENCH_ASSET_DIR` first, then the repo-local
`assets/robots/<robot>/files/` directory. Do not put robot USDs or meshes under
`inputs/`.

## Inputs

Inputs are run data or frozen parameter files, not robot definitions.

```text
inputs/
  datasets/
    <robot>/
      <dataset>/
        train/<motion>/
          control.csv
          state_motor.csv
          joint_list.txt
          metadata.yaml
        heldout/<motion>/
  actuator_params/
    so101/
      fitted_seed42.yaml  # optional frozen known params for publishing/replay
  actuator_models/
    <robot>/
      README.md           # non-weight neural actuator metadata and recipes
```

Only put actuator params under `inputs/actuator_params/` when they are deliberate
run inputs, such as a published fitted file. Generated SysID results should stay
in `outputs/sysid/...` unless you intentionally promote one into
`inputs/actuator_params/`. Neural checkpoints remain generated artifacts; only
small metadata belongs under `inputs/actuator_models/`.

## Outputs

Outputs are generated run artifacts and should not be treated as robot config.

```text
outputs/
  sysid/
    <robot>/<run>/
      best_params.yaml
      run_summary.yaml
  training/
    <robot>/<run>/
      *_best.pt           # ignored generated checkpoint
      *_stats.json        # normalization and metric metadata
      *_network.pt        # ignored TorchScript runtime network
  sim2real_benchmark/
    sim/<robot>/<dataset>/<actuator_model>/<motion>/
    real/<robot>/<dataset>/<motion>/
  sim2real_analysis/
    metrics/
    plots/
```

Use `outputs/sysid/.../best_params.yaml` directly for local replay, or copy a
chosen frozen file into `inputs/actuator_params/<robot>/` when it becomes a
published input. Fitted files use the same schema-v2 actuator document as
`configs/robots/<robot>/actuators.yaml`, with run provenance under the optional
top-level `metadata` mapping. Benchmark and analysis outputs can be regenerated
and should stay separate from `configs/`.

## Third Party

Use `third_party/` for local binary artifacts and optional vendor checkouts:

```text
third_party/
  artifacts/h1/nvidia-motor.fmu
  IsaacLab/        # optional local checkout
```

The actual FMU and checkout contents are ignored by git.
The full container image uses /opt/IsaacLab by default; `third_party/IsaacLab`
is only used when explicitly selected with the local development workflow in
`docs/cli.md`.
