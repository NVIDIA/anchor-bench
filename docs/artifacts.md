Large datasets, SO-101 benchmark outputs, neural weights, FMUs, third-party checkouts, and generated plots are local artifacts. Keep only manifests, placeholders, and small synthetic examples in git.

## Third-Party Artifacts

Use `third_party/` for local binary artifacts and optional source checkouts:

- `third_party/artifacts/h1/nvidia-motor.fmu` for the H1/teststand FMU artifact.
- `third_party/IsaacLab/` only when keeping a local Isaac Lab checkout beside this repo.

The actual FMU and checkout contents are ignored by git.

## SO-101 Runtime Assets

The repo includes `assets/robots/so101/files/so101.usd` so SO-101 smoke checks
work after cloning. Set `ANCHOR_BENCH_ASSET_DIR` only when overriding that
default with another asset directory containing `so101.usd`. SysID writes to
`outputs/sysid/...`, neural actuator training writes checkpoints and stats
under `outputs/training/<robot>/...`, benchmark playback writes SAGE-compatible
folders under `outputs/sim2real_benchmark`, and analysis writes CSV metrics
under `outputs/sim2real_analysis/metrics` plus generated joint and end-effector
plots under `outputs/sim2real_analysis/plots/<dataset>/<motion>/...`.


## Neural Actuator Artifacts

Keep `.pt`, `.pth`, `.onnx`, raw training datasets, and generated plots out
of git. Training runs should write raw checkpoints, TorchScript networks,
and stats JSON files to `outputs/training/<robot>/...`.
If a trained model needs a small human-readable descriptor, promote only
non-weight metadata to `inputs/actuator_models/<robot>/`, such as feature
columns, stats schema notes, source dataset names, or benchmark recipes.
