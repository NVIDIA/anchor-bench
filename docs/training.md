# Training actuator models

Anchor Bench's `training` task prepares mapped time-series data, trains a
recurrent actuator model, and validates the result. The maintained G1 right-arm
workflow provides two complete configurations:

- residual-torque training with identified actuator parameters;
- autoregressive full-torque training.

The configurations and dataset preparation script live under
[`examples/training/`](../examples/training/). Their reusable physics assets
live under [`assets/robots/g1/`](../assets/robots/g1/), and the integration tests
consume the same examples rather than maintaining separate copies.

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
uv run python examples/training/prepare_dataset.py
```

The preparation script downloads the 15 original Parquet files under
`examples/training/data/`. The configs load them through one local glob and
map their raw joint and feature names without rewriting the source files. The
downloads are ignored by Git.

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

To execute dataset preparation, training, and Newton replay through pytest:

```bash
uv run pytest -m integration tests/integration/g1 -v
```

The integration tests load the distributed configurations and override only
the dataset location, epoch count, and checkpoint interval to keep the test run
bounded.

## Newton GRU deployment

For compatible per-joint models, training saves `model.onnx` alongside the
PyTorch training weights. The ONNX file
embeds ordered feature names, fitted input and output normalization, and the
sample timestep. Load it directly with `newton.actuators.DriveNeuralGRU` and bind
it to joint DOFs with `newton.actuators.Actuator` or `ModelBuilder.add_actuator`.
Evaluate the actuator once per training sample interval; physics substeps may
run more frequently while holding the resulting effort. Reset both actuator
state buffers at each episode boundary.

Newton replay validates the exported ONNX graph through `DriveNeuralGRU`, using
the live simulated joint state and commands. It updates hidden state once per
sample and resets it between experiments. Output normalization is applied once
by the drive. Residual models produce residual effort, so retain the base PD
controller when deploying them; full-torque models replace that controller.

Newton currently requires one scalar output per joint lane, identity or tanh
output activation, and at most one caller-supplied input. Mappings must each
contain one joint's inputs and output, with the same feature order in every
mapping. MIMO training and PyTorch weights remain supported. `metadata.yaml`
records `newton.supported` and either the ONNX artifact path or an incompatibility
reason. Unsupported models do not produce ONNX files; requesting Newton GRU
replay for them fails explicitly. The G1 examples now share one per-joint model
across four mappings. The full-torque example keeps
`previous_torque` feedback and removes `solver_pd` and `qfrc_bias` from its inputs
to fit Newton's one-custom-input contract. Applications must supply the chosen
custom input as a DOF-sized Warp array on the actuator's control object.

The Newton dependency is pinned to a source revision containing `DriveNeuralGRU`
and includes the `onnx` extra. Install the locked dependencies with `uv sync`.
