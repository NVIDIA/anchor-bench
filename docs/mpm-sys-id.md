# MPM material system identification

`mpm_sys_id` fits one shared sand material against independently enabled Newton
MPM experiments. The first version supports **sand pour** and **bottle drop**.
Multiple experiments can use either simulation with different settings and data.
It requires a CUDA GPU and official Newton 1.7.0. It does not require a checkout
of the Newton repository or an OpenGL display.

```bash
uv sync --extra mpm
uv run anchor-bench mpm_sys_id \
  --config examples/mpm_sys_id/sand.yaml --output outputs/sand-fit
```

The example contains the two pour and four bottle-drop experiments, with all
simulation defaults expanded for reproducibility. It runs **continuously until
interrupted**. For a finite run, change the optimizer stopping configuration:

```yaml
optimizer:
  type: random_forest
  stopping:
    mode: bounded
    max_evaluations: 64
    # Optional: stop when every enabled experiment meets this relative error.
    target_error: 0.05
```

Continuous mode rejects evaluation limits and tolerances. The random forest
uses baseline probes, space-filling samples, and local/global surrogate proposals.
Tree disagreement is a search heuristic, not physical uncertainty. The initial
candidate uses the original baseline parameters within the configured bounds.

## Configuration

The top-level `type` is `mpm_sys_id`, with `schema_version: 1`. The task also works
through the existing `anchor-bench --config ...` entry point. The explicit
subcommand rejects configurations with a different task type.

```yaml
experiments:
  - id: pour_1m
    enabled: true
    weight: 1.0
    input_data:
      type: csv
      source: data/pour_1m.csv
    simulation:
      type: sand_pour
      drop_height_m: 1.0
      sand_volume_m3: 0.01892705892
      voxel_size: 0.02
      num_frames: 720
```

IDs must be unique and contain only letters, digits, underscores, and hyphens.
At least one experiment must be enabled. Disabled experiments are neither loaded
nor simulated and do not contribute to the objective. Their CSV files may be
absent, but their configuration must remain structurally valid.

Each experiment owns its geometry, density, time discretization, and contact
settings. Input paths resolve relative to the YAML file, including when the
command runs from a different working directory. Output paths follow the normal
Anchor Bench convention. `execution.device` defaults to `cuda:0`, `seed` to 48,
`cuda_graphs` to true, and `worker_timeout_s` to 1800 per candidate.

The checked-in example is the full field reference. Important simulation fields:

| Both simulations | Meaning |
| --- | --- |
| `drop_height_m` | Outlet-to-floor distance for pour; cap-to-sand distance for bottle |
| `density` | Bulk sand density in kg/m³ |
| `voxel_size` | Grid spacing in meters |
| `fps`, `substeps`, `num_frames` | Frame frequency, physics steps per frame, evaluation duration |
| `ground_friction` | Ground and bed-wall friction coefficient |

Pour also configures `sand_volume_m3`, `bucket_radius`, `funnel_height`,
`outlet_diameter`, and `funnel_friction`. Bottle configures `water_volume_m3`,
`bed_width`, `bed_length`, `bed_depth`, `bottle_radius`, `bottle_height`,
`cap_radius`, `cap_height`, `bottle_friction`, `pre_settle_seconds`,
`impact_settle_seconds`, `lift_speed`, `lift_distance`, and `crater_threshold`.
Lengths are meters, speeds m/s, and durations seconds. Bottle dimensions are
explicit; bottle volume does not silently select a geometry.

## CSV input format

Only CSV is supported. Use a UTF-8, comma-separated file with the exact header
below, in this order. Each subsequent row is one independent physical repeat
of the **same experiment configuration**, not a simulation frame.

Sand pour:

```csv
width_m,length_m,height_m
0.59,0.60,0.10
0.72,0.53,0.08
0.60,0.43,0.11
```

Bottle drop (illustrative values):

```csv
width_m,length_m,depth_m
0.15,0.14,0.04
0.16,0.15,0.05
```

- All measurements are **meters** and must be finite and strictly positive.
- Include at least one row. Missing cells, extra columns, nonnumeric values,
  NaN, and infinity are errors. Do not add an index column.
- Width and length use fixed X/Y measurement axes.
- Pour height is the highest pile surface above the floor; the footprint
  includes scattered particles.
- Bottle depth is positive downward, measured from the **post-removal crater
  rim to the crater floor**. Remove the bottle before measuring.
- Column means become fitting targets. Sample standard deviations are recorded
  as descriptive repeat variability; a single repeat has `null` standard deviation.
- Repeats do not increase experiment weight. Set `weight` explicitly if needed.

The bundled CSVs preserve the original six experimental datasets. Their original
filenames are retained, including `2m-gal-sand-pour.csv` (a five-US-gallon pour).

## Material and objective

There is no `material.model` selector: this version has one supported sand model.
The five optimized parameters are damping, friction, yield pressure, tensile
yield ratio, and yield stress. Bounds specify `lower`, `upper`, `scale` (`linear`
or `log`), and optionally a nonnegative log `offset` to include zero.
Damping is in seconds, pressures and stress in Pa; the two ratios are dimensionless.

Young's modulus is fixed at **9,576,000 Pa**, Poisson's ratio at **0.25**.
Plasticity defaults remain fixed: hardening 0, hardening rate 1, softening rate 1,
dilatancy 0, and viscosity 0. The schema and worker reject changes to these values
or attempts to optimize them. Experiments cannot override shared material values.

For each dimension, residual = `(prediction - target) / target`. Each experiment's
error is the RMS of its dimensional residuals. The combined error is:

```text
sqrt(sum(weight_i * error_i²) / sum(weight_i))
```

Artifacts store the error as a fraction; multiply by 100 for a percentage.
Every enabled experiment must be valid. Invalid experiments are never dropped
from the denominator, and invalid candidates cannot become the best fit.

Pour validity requires at least eight simulated seconds, retained sand fraction
at most 0.001, RMS speed below 0.03 m/s, and finite, conserved, contained states.
Bottle validity requires completed extraction, at least one second of subsequent
settling, RMS speed below 0.03 m/s, a crater inside the measurement region, and
finite, conserved states with a bottle that indented but did not pass through
the bed. Short evaluations may finish with no valid fit; this is reported by
`valid: false` and the absence of `best.json`.

## Artifacts and resume

```text
outputs/sand-fit/
  config.yaml
  inputs.json
  status.json
  best.json                 # Only when a valid candidate exists
  history.jsonl
  checkpoint.json
  runner.log
  trials/00000001/
    request.json
    response.json
    result.json
    worker.log
```

`inputs.json` stores resolved settings, measured repeats, targets, content hashes,
and dependency versions. `result.json` stores the evaluation ID, optimizer point,
full material, validity, combined error, per-experiment predictions/residuals,
and elapsed seconds. A worker response is tied to its request by SHA-256.
`best.json` has the same format as a trial result. `history.jsonl` has one result
object per completed evaluation. `checkpoint.json` records the next evaluation
and seed; seeded proposals are reconstructed from completed trial records.

Completed trial results are written atomically and are authoritative. Resume
rebuilds derived history and best-result files, reuses completed worker responses,
and retries interrupted candidates. It refuses changed data, selected experiments,
physics, parameter bounds, objective, stopping settings, or dependency versions.
A new search with different settings needs a new output directory.

```bash
uv run anchor-bench mpm_sys_id --config examples/mpm_sys_id/sand.yaml \
  --output outputs/sand-fit --resume
```

Ctrl-C and SIGTERM stop the worker, preserve progress, and leave the task
incomplete. `.completed` is written only after a bounded search returns normally.
Output and worker locks prevent concurrent evaluation of the same run. `--force`
does not erase MPM checkpoints; choose a fresh output directory to start over.
Data fingerprints are checked even when reusing an already completed run.

Existing optional `mlops` configuration logs CSV dataset lineage (with optional
upload via `upload_dataset`), candidate validity and counts, best material values,
and combined and per-experiment scores, plus final run artifacts. Local artifacts remain the
source of truth for resume. Continuous runs retain local artifacts throughout.

## Implementation and limitations

The coordinator runs the optimizer; a fresh subprocess evaluates each candidate.
Enabled experiments run sequentially in that worker to bound GPU memory use.
They share the candidate material but have independent physical states. Numerical
failures and timeouts produce invalid trials; a worker process failure stops the
search with a log path so infrastructure failures do not silently run forever.

The simulations and forest proposal algorithm were adapted from Newton commit
`a9a2d2b4` (Apache-2.0, Newton Developers). Original attribution is preserved.
The optional `mpm` extra declares scikit-learn 1.7.2 (BSD-3-Clause); file locking
uses filelock (Unlicense). Both are compatible with this Apache-2.0 project.
Warp kernel typing exceptions are localized to the compiled kernel module;
Newton's MPM step stubs incorrectly require unused controls/contacts. These
boundaries are exercised by CUDA integration tests.

Bottle water is a rigid mass at 1,000 kg/m³, neglecting bottle shell mass, slosh,
and deformation. Funnel and bottle geometry are provisional experimental
assumptions, made explicit in the example. Fitted results are specific to these
assumptions and discretization settings. This command does not provide a viewer
or import legacy Newton search histories.

To add a simulation, add a typed schema to the experiment union and an
implementation with `step`, `measure`, and `test_final`; register it in
`simulations/registry.py` and declare its CSV observables in the data loader.
Keep its validity criteria explicit and add measurement and GPU integration tests.

## Validation

```bash
uv sync --extra dev --extra mpm
uv run pytest tests/test_mpm_sys_id.py
uv run pytest -m integration tests/integration/mpm_sys_id
```

CUDA graph regression tests compare physical states with the original test's
1 mm absolute position tolerance for bottle contact (5% of its 2 cm test voxel).
Atomic impulse reductions can produce small trajectory changes between runs;
thresholded crater dimensions are not bitwise reproducible. Full-duration
comparisons should be interpreted at the configured voxel resolution.
