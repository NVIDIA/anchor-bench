# Repository structure

The repository separates robot identity, editable recipes, accepted state,
runtime assets, measured inputs, and generated evidence.

```text
anchor_bench/          Schemas, resolution, execution, promotion, and runtime adapters
configs/robots/        Robot identity, phase profiles, accepted state, frozen deploy state
assets/robots/         Asset manifests and small public assets
inputs/datasets/       Measured SAGE trajectories
outputs/               Immutable candidates and benchmark evidence
third_party/           Ignored binary artifacts and optional source checkouts
docker/                Container entrypoints
docs/                  Workflow and contributor guidance
tests/                 Host-safe contracts and opt-in runtime tests
versions/              Supported runtime pins
```

## Robot configuration

Each robot is self-contained under `configs/robots/<robot>/`:

```text
configs/robots/<robot>/
  robot.yaml
  sysid/
    profiles/default.yaml
    profiles/<recipe>.yaml
    default.yaml
  train/
    profiles/default.yaml
    profiles/<recipe>.yaml
    default.yaml
  benchmark/
    profiles/default.yaml
    profiles/<recipe>.yaml
  deploy/
    profiles/default.yaml
    profiles/<recipe>.yaml
    default.yaml             # created only by promote deploy
```

Not every robot needs every optional named profile. A robot must have the
identity and accepted state needed by the phases it supports.

`robot.yaml` contains only stable identity:

- robot and display names;
- fixed-base setting;
- asset manifest and runtime USD name;
- canonical joint order and aliases;
- optional end-effector frame.

Phase recipes live in their own `profiles/` directories. A selected named
profile replaces `profiles/default.yaml`; it is not layered on top. SysID and
training CLI overrides then replace typed fields in that selected profile.

`sysid/default.yaml` is the accepted physical-parameter state.
`train/default.yaml` is the accepted neural-model catalog. They are written only
by their explicit promotion commands. `deploy/default.yaml` is a frozen runtime
composition materialized from accepted state; it is not a mutable recipe.

Generated files and neural weights never belong in `configs/`.

## Assets

Asset metadata lives at `assets/robots/<robot>/manifest.yaml`. Small public
assets may be checked into `assets/robots/<robot>/files/`. Large, licensed,
private, or generated assets should be supplied through
`ANCHOR_BENCH_ASSET_DIR`.

Runtime asset lookup checks the configured override and repository asset
locations. `robot.yaml` names the runtime USD; profiles do not select a different
robot asset. Actual runs hash the resolved asset as part of semantic identity.
Dry runs use the manifest fingerprint and do not resolve or import the runtime
asset.

Do not place USDs, meshes, or FMUs under `inputs/`.

## Inputs

Measured data uses the SAGE layout:

```text
inputs/datasets/<robot>/<dataset>/<split>/<motion>/
  control.csv
  state_motor.csv
  joint_list.txt
  metadata.yaml
```

`joint_list.txt` should use canonical robot names. Profile and CLI selectors may
use configured aliases, but resolved runs and artifacts persist canonical names.
Keep training and held-out motions separate. Datasets are fingerprinted as
semantic run inputs and are not copied into output directories.

Inputs are measured data, not a place for accepted actuator parameters or
neural model descriptors. Accepted state lives with the robot config; generated
candidate state and weights live in immutable runs.

## Outputs

SysID and training candidates have deterministic directories:

```text
outputs/sysid/<robot>/<profile>/seed-<seed>/<run-input-sha256>/
outputs/train/<robot>/<profile>/seed-<seed>/<run-input-sha256>/
```

Benchmark evidence binds an exact source-run manifest:

```text
<benchmark-output-root>/<robot>/<sysid|train>/<benchmark-profile>/
  <source-manifest-sha256>/<benchmark-input-sha256>/
```

The checked-in benchmark profiles use `outputs/sim2real_benchmark` as this
root. `--output-root` may choose another repository-contained location without
changing benchmark identity.

Every completed run writes `manifest.yaml` last. The manifest authenticates the
semantic input, generated files, upstream state, source tree, assets, and
dependencies. Reusing a valid complete directory is a no-op; a partial or
corrupt directory is an error rather than an overwrite target.

Output paths referenced by accepted training state, accepted promotion
provenance, benchmark evidence, or frozen deployment must be retained. Anchor
Bench does not currently garbage-collect them.

## Third-party files

Use `third_party/` for local binary artifacts and optional vendor checkouts:

```text
third_party/
  artifacts/<robot>/...
  IsaacLab/
```

These contents are ignored. The full container uses its pinned Isaac Lab by
default; select a host-editable checkout only through the documented setup
workflow.

## Repository rules

- Keep robot support declarative under `configs/robots/<robot>/`.
- Keep heavy Newton, Isaac Lab, Isaac Sim, Warp, and Torch imports behind actual
  execution boundaries.
- Use `anchor-bench ... --dry-run` for host-safe construction tests.
- Do not commit datasets, neural weights, FMUs, generated runs, plots, or
  third-party checkouts.
- Do not manually edit machine-managed accepted or frozen state to imitate a
  promotion.
