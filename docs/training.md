# Training actuator models

Anchor Bench's `training` task prepares mapped time-series data, trains a
recurrent actuator model, and validates the result. The maintained G1 right-arm
workflow provides two complete configurations:

- residual-torque training with identified actuator parameters;
- autoregressive full-torque training.

Their supporting physics layers, dataset preparation script, configurations,
and integration tests live under [`tests/integration/g1/`](../tests/integration/g1/).

The example pins the corrected Anchor-Lab `g1_29dof_tri_hand_closed.usd`
asset at revision `50881f268b6600251df6f2011cfaa5a392fb96eb` and verifies its checksum
before replay.

A CUDA-capable GPU is required for training and Newton replay. Run all commands
from the repository root.

## Prepare the environment and dataset

Install the development dependencies, then download and prepare the 15
revision-pinned Anchor-Lab recordings. Newton's simulation components are
included in the default installation:

```bash
uv sync --extra dev
uv run python tests/integration/g1/prepare_dataset.py
```

The preparation script writes
`tests/integration/g1/data/g1_arm_multijoint.parquet`. The dataset and
intermediate downloads are ignored by Git.

## Train the models

The two examples can run concurrently with one GPU assigned to each model.

Residual-torque model:

```bash
CUDA_VISIBLE_DEVICES=0 uv run anchor-bench \
  --force \
  --config tests/integration/g1/g1_arm_residual.yaml \
  --output outputs/g1_arm_residual
```

Autoregressive full-torque model:

```bash
CUDA_VISIBLE_DEVICES=1 uv run anchor-bench \
  --force \
  --config tests/integration/g1/g1_arm_full_torque.yaml \
  --output outputs/g1_arm_full_torque
```

Both configurations train for two epochs, write a checkpoint at the final
epoch, and run Newton replay on the three experiments selected by the
deterministic held-out split. Contact generation is disabled because these are
fixed-base arm replays without environmental contact.

The full-torque model is autoregressive and therefore takes substantially
longer than the residual model.

## Run the integration tests

To execute dataset preparation, training, and Newton replay through pytest:

```bash
uv run pytest -m integration tests/integration/g1 -v
```
