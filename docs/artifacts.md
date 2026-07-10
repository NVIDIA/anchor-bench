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

## Hugging Face Robot Parquets

Import every parquet below a Hugging Face dataset folder with:

The importer uses `huggingface_hub` for authenticated repository traversal, cached downloads, and retries, then verifies published LFS SHA-256 values before conversion.

```bash
python scripts/prepare_huggingface_parquets.py \
  https://huggingface.co/datasets/nvidia/Anchor-Lab/tree/main/data/so101_arm_50motion
```

The importer derives the robot and dataset names from the folder by default.
Use `--robot`, `--output`, or `--download-dir` only when those inferred
locations are unsuitable. Parquets are cached beneath
`inputs/datasets/<robot>/_parquet/<dataset>/`; converted SAGE motions are
written beneath `inputs/datasets/<robot>/<dataset>/<split>/<motion>/`.

Remote `train`, `validation`, `heldout`, and `test` filename markers are
preserved; files without a marker default to `train`. Pass only `train/` to
`train-actuator`. Held-out motions must not influence normalization,
hyperparameter selection, or checkpoint selection.

The importer detects the public Anchor-Lab SO-101, G1, and H1 field conventions
instead of carrying robot-specific joint tables. Configured robot joint names
and aliases are applied when possible; unrecognized joints retain their source
names. Every generated `metadata.yaml` records the source fields, original
joints, URL, revision, checksum, effort semantics, and whether a recorded
command was available. H1 sources without command telemetry use measured
position as the SAGE control fallback and are marked
`control_semantics: measured_position_fallback`; they should not be treated as
command-error training data.

SO-101's unsigned `tau_abs` is signed using `present_load_raw` when that
signal is available. This and all inferred units must be checked against the
source hardware calibration before physical conclusions.

## Fitted Actuator Parameters

SysID writes `best_params.yaml` as a complete schema-v2 actuator document.
It has the same `schema_version`, `backend`, and `actuators` structure as
the robot's canonical `actuators.yaml`; fit provenance and metrics live under
the optional top-level `metadata` mapping. The file can be passed directly to
`anchor-bench benchmark --actuator-params` or deliberately promoted to
`inputs/actuator_params/<robot>/`.

## Neural Actuator Artifacts

Keep `.pt`, `.pth`, `.onnx`, raw training datasets, and generated plots out
of git. Training runs should write raw checkpoints, TorchScript networks,
and stats JSON files to `outputs/training/<robot>/...`.
If a trained model needs a small human-readable descriptor, promote only
non-weight metadata to `inputs/actuator_models/<robot>/`, such as feature
columns, stats schema notes, source dataset names, or benchmark recipes.
