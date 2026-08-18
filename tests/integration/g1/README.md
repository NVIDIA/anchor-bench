# G1 arm training integration

This directory contains the complete reproducible G1 right-arm training example:

- the corrected G1 TriHand asset and sparse training USD layers;
- residual-torque and autoregressive full-torque training configs;
- the identified actuator parameters used to generate the sparse USD opinions;
- a revision-pinned downloader for all 15 Anchor-Lab recordings;
- end-to-end training and Newton replay tests.

Run commands from the repository root.
A CUDA-capable GPU is required for training and Newton replay.

The robot asset is
`nvidia/Anchor-Lab/robot_assets/g1_29dof_tri_hand_closed.usd` at revision
`50881f268b6600251df6f2011cfaa5a392fb96eb` with SHA-256
`9e61f39e3ace1cd60b3cf9eb7dc5422a02584ad5b23b049d2d792bbbec2b9ddc`.

## Setup

```bash
uv sync --extra simulation --extra dev
uv run python tests/integration/g1/prepare_dataset.py
```

The downloader creates
`tests/integration/g1/data/g1_arm_multijoint.parquet`. The dataset and
intermediate downloads are ignored by Git.

## Train both models

Use one GPU for each model:

```bash
CUDA_VISIBLE_DEVICES=0 uv run anchor-bench \
  --force \
  --config tests/integration/g1/g1_arm_residual.yaml \
  --output outputs/g1_arm_residual
```

```bash
CUDA_VISIBLE_DEVICES=1 uv run anchor-bench \
  --force \
  --config tests/integration/g1/g1_arm_full_torque.yaml \
  --output outputs/g1_arm_full_torque
```

The configs train for 2 epochs, write a checkpoint at the final epoch, and run
Newton replay on the three experiments selected by the deterministic held-out
split. Contact generation is disabled because this is a fixed-base arm replay
without environmental contact.

## Run through pytest

```bash
uv run pytest -m integration tests/integration/g1 -v
```

The full-torque model is autoregressive and therefore takes substantially
longer than the residual model.
