# End-to-End Pipeline Tests

> **Outdated:** This guide describes an earlier architecture and tests that are
> not present in the current package. See the [project README](../README.md) for
> the currently supported test commands.

Anchor Bench has two end-to-end pipeline layers:

- `tests/test_pipeline_contract.py` is the default, host-safe contract test for
  SO-101 SysID, benchmark, and analysis wiring.
- `tests/test_runtime_pipeline.py` is an opt-in full runtime smoke test that
  launches the actual Isaac Sim / Isaac Lab / Newton / GPU path.

The host-safe test should stay in the normal unit suite. The full runtime smoke
should be enabled only on machines with the full container, CUDA, and SO-101
runtime assets.

## Host-Safe Contract Test

`tests/test_pipeline_contract.py` verifies the pipeline boundary without
launching Isaac Sim or Newton.

### What It Proves

- A schema-v2 SysID `best_params.yaml` can be consumed by benchmark
  actuator-parameter loading.
- Experiment benchmark expansion passes the expected seed, output folder,
  actuator params, dataset, actuator model, motion name, and headless setting
  into the benchmark runtime boundary.
- Benchmark outputs in CSV trajectory directories can be consumed by the real
  analysis reader.
- Analysis writes `so101_metrics.csv` rows for the selected dataset, motion,
  actuator model, seed, and joints.
- Plot dependencies stay lazy: `_load_pyplot()` is not called when joint and
  end-effector plots are disabled.

### What It Does Not Prove

- CMA-ES optimizer convergence.
- Isaac Sim, Isaac Lab, Newton, Torch, or Warp runtime availability.
- Physical accuracy of the simulated trajectory.
- GUI, GPU, or container behavior.

Those checks belong in the opt-in runtime smoke test, not the normal host-safe
unit-test suite.

### Test Shape

The test creates all inputs in a temporary directory:

```text
datasets/custom_motion/
  control.csv
  state_motor.csv
  joint_list.txt
  metadata.yaml
sysid/
  seed_42/best_params.yaml
  seed_7/best_params.yaml
benchmark/
  seed_42/
    real/so101/synthetic/custom_motion/
    sim/so101/synthetic/fitted_implicit/custom_motion/
analysis/
  metrics/so101_metrics.csv
```

The experiment config uses two benchmark seeds so the experiment layer is
required to create seed-specific benchmark output roots:

```yaml
seeds: [42, 7]

actuator_models:
  fitted_implicit:
    type: implicit
    fitted_params: <tmp>/sysid/seed_{seed}/best_params.yaml
    benchmark_params: fitted

analysis:
  seeds: [42]
  plots:
    joints: false
    end_effector: false
```

Using `analysis.seeds: [42]` proves that the analysis stage can select one
seed from a multi-seed benchmark layout.

### Runtime Boundary

The test patches `anchor_bench.benchmark.newton_runner.run_benchmark_batch`.
That keeps the unit test host-safe while still checking the exact arguments that
would be handed to the Newton benchmark runtime.

After the patched benchmark call is verified, the test writes synthetic
benchmark artifacts and calls `run_experiment_analysis(..., dry_run=False)` so
the real analysis implementation reads the generated trajectory directories and
writes metrics.

## Full Runtime Smoke Test

`tests/test_runtime_pipeline.py` verifies the same SO-101 workflow through the
real public CLI entrypoints:

```text
anchor-bench sysid
anchor-bench benchmark
anchor-bench analyze --require-end-effector
```

It is skipped by default. Set `ANCHOR_BENCH_RUN_RUNTIME_E2E=1` to enable it.

### What It Proves

- The full runtime modules can be imported: `isaaclab`, `isaaclab_newton`,
  `newton`, `torch`, and `cmaes`.
- CUDA is visible inside the full runtime container.
- The SO-101 runtime USD can be resolved.
- Actual SysID optimization writes `best_params.yaml`, `optimization_log.csv`,
  and `run_summary.yaml`.
- Actual benchmark replay consumes the generated fitted params and writes real
  and simulated SAGE folders plus a benchmark `run_summary.yaml`.
- Actual analysis consumes the benchmark output, requires end-effector FK, and
  writes joint metrics, end-effector metrics, joint plots, and
  `end_effector.png`.

This is a runtime contract smoke, not a physical accuracy benchmark. It checks
that the full stack runs and produces the expected artifacts.

### Runtime Test Shape

The test creates a tiny all-joint SO-101 SAGE motion with two timestamps. By
default it uses a temporary directory. To preserve artifacts for debugging, set
`ANCHOR_BENCH_RUNTIME_OUTPUT_DIR`; the test will create a fresh
`runtime_pipeline_<uuid>` subdirectory under it.

```text
runtime_pipeline_<uuid>/
  inputs/runtime_smoke_motion/
  sysid/
    best_params.yaml
    optimization_log.csv
    run_summary.yaml
  benchmark/
    real/so101/runtime_smoke_dataset/runtime_smoke_motion/
    sim/so101/runtime_smoke_dataset/fitted_implicit/runtime_smoke_motion/
  analysis/
    metrics/so101_metrics.csv
    metrics/so101_end_effector_metrics.csv
    plots/runtime_smoke_dataset/runtime_smoke_motion/
```

If the opt-in flag is set but prerequisites are missing, the test skips with an
actionable reason naming missing runtime modules, CUDA, or the SO-101 asset. With
the full image and GPU runtime available, the test executes the complete
SysID → benchmark → analysis pipeline.

## Run It

Host-safe contract test:

```bash
./setup.sh --shell python3 -m pytest -q tests/test_pipeline_contract.py
```

Adjacent coverage:

```bash
./setup.sh --shell python3 -m pytest -q tests/test_experiments.py tests/test_runtime_workflows.py tests/test_cli_dry_run.py
```

Default container suite, with the runtime smoke discovered but skipped:

```bash
./setup.sh --shell python3 -m pytest -q
```

Opt-in full runtime smoke:

```bash
HEADLESS=1 ./setup.sh --shell env \
  ANCHOR_BENCH_RUN_RUNTIME_E2E=1 \
  ANCHOR_BENCH_RUNTIME_OUTPUT_DIR=/tmp/anchor-bench-runtime-e2e \
  python3 -m pytest -v tests/test_runtime_pipeline.py
```

## Test Directory Summary

| Test module | Coverage summary | Runtime level |
| --- | --- | --- |
| `tests/test_actuator_net_gru.py` | Host-safe import/config checks for GRU actuator config classes and input-column validation. | Host-safe unit |
| `tests/test_actuator_schema.py` | Declarative actuator, run, experiment, and Newton adapter schema behavior. | Host-safe unit |
| `tests/test_cli_dry_run.py` | CLI command construction for robots, SysID, benchmark, analysis, and train-actuator dry-runs. | Host-safe subprocess |
| `tests/test_docker_setup.py` | Docker-first setup script, persistent shell, docs, cache, UID/GID, and artifact-layout expectations. | Host-safe dry-run/docs |
| `tests/test_experiments.py` | Experiment config expansion, batching, seed layout, fitted/initial params, analysis selectors, and CLI dry-runs. | Host-safe unit/subprocess |
| `tests/test_pipeline_contract.py` | End-to-end host-safe SysID params -> benchmark boundary -> analysis artifact contract. | Host-safe contract |
| `tests/test_generic_runtime.py` | Robot-parameterized dispatch and non-SO-101 analysis behavior. | Host-safe unit |
| `tests/test_registry.py` | Robot registry discovery, configured robot bundles, and invalid robot diagnostics. | Host-safe unit |
| `tests/test_runtime_pipeline.py` | Opt-in actual SO-101 SysID optimization → benchmark replay → FK analysis smoke. | Full runtime opt-in |
| `tests/test_runtime_simulation.py` | Runtime simulation timing and scene integration contracts with fakes. | Host-safe unit |
| `tests/test_sage.py` | SAGE loader ordering, initial-state extraction, timestamp scaling, interpolation, and validation errors. | Host-safe unit |
| `tests/test_runtime_workflows.py` | runtime workflow helpers, asset preflights, optimizer behavior, benchmark row/layout helpers, and analysis behavior with fakes. | Host-safe unit |
| `tests/test_training.py` | SAGE training-source discovery, feature mapping, tiny GRU training, and TorchScript export. | Host-safe; Torch tests require torch |
| `tests/test_versions.py` | Compatibility manifest pins for Isaac Sim, Isaac Lab, and Newton. | Host-safe unit |
