# Training actuator models

Anchor Bench's `training` task prepares mapped time-series data, trains a
recurrent actuator model, and validates the result. The maintained G1 right-arm
workflow provides two complete configurations:

- residual-torque training with identified actuator parameters;
- autoregressive full-torque training.

The configurations live under [`examples/training/`](../examples/training/).
Their reusable physics assets live under
[`assets/robots/g1/`](../assets/robots/g1/), and the integration tests consume
the same examples rather than maintaining separate copies.

The example pins the corrected Anchor-Lab `g1_29dof_tri_hand_closed.usd`
asset at revision `50881f268b6600251df6f2011cfaa5a392fb96eb` and verifies its checksum
before replay.

A CUDA-capable GPU is required for training and Newton replay. Run all commands
from the repository root.

## Prepare the environment

Install the development dependencies. Newton's simulation components and the
Hugging Face client are included in the default installation:

```bash
uv sync --extra dev
```

The configurations select all 15 original G1 Parquet files from a pinned
Anchor-Lab revision. The first run downloads them into the Hugging Face cache;
later runs reuse those bytes. Anchor Bench maps their raw joint and feature
names without rewriting the source files.

## Train the models

The two examples can run concurrently with one GPU assigned to each model.

Residual-torque model:

```bash
CUDA_VISIBLE_DEVICES=0 uv run anchor-bench \
  --force \
  --config examples/training/g1_arm_residual.yaml \
  --output outputs/g1_arm_residual
```

Autoregressive full-torque model:

```bash
CUDA_VISIBLE_DEVICES=1 uv run anchor-bench \
  --force \
  --config examples/training/g1_arm_full_torque.yaml \
  --output outputs/g1_arm_full_torque
```

Both configurations train for 50 epochs, write checkpoints every 10 epochs,
and run Newton replay on the three experiments selected by the deterministic
held-out split. Contact generation is disabled because these are fixed-base arm
replays without environmental contact.

The full-torque model is autoregressive and therefore takes substantially
longer than the residual model.

### CUDA execution optimizations

Autoregressive training captures a fixed-size optimizer step in a CUDA graph
after one eager warm-up batch. Full batches replay that graph, while a final
smaller batch continues through the eager path. Feedback column indices and
normalization tensors are also created once on the training device.

Training and validation batches are staged on the GPU when their combined size
is at most 512 MiB and at most 10 percent of currently available device memory.
Larger datasets remain streamed from the CPU, avoiding an unbounded GPU-memory
commitment.

## Run the integration tests

To execute dataset acquisition, training, and Newton replay through pytest:

```bash
uv run pytest -m integration tests/integration/g1 -v
```

The integration tests load the distributed configurations and override only
the epoch count and checkpoint interval to keep the test run bounded.
