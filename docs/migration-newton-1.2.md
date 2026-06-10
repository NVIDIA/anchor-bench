
# Newton 1.2 Migration

Anchor Bench stores actuator configs in the native Newton 1.2 component shape.
Legacy `model_type` blocks are intentionally migrated rather
than copied verbatim.

Old categories map as follows:

| Legacy value | Native controller | Notes |
| --- | --- | --- |
| `implicit` | `pd` | Limits move to clamp components. |
| `dcmotor` | `pd` + `dc_motor` clamp | DC saturation is a clamp, not a model type. |
| `lstm` | `lstm` | Network file becomes controller metadata. |
| `gru` | compatibility runtime shim | Newton-native GRU is not available in this snapshot. |
| `hybrid_residual` | `pd` + GRU runtime shim | PD remains base controller. |
| `fmu` | `fmu` | FMU path is controller metadata. |

Do not add new legacy `model_type` configs under `configs/robots/`.

Hybrid GRU residuals use the same GRU network contract as full-torque GRUs,
but live under an actuator `residual:` block. The base `controller.type` stays
`pd` so PD remains active while `anchor_bench.runtime.actuator_net_gru` adds
`torque_residual` as feed-forward effort.

At runtime, `anchor_bench.actuators.instantiate_newton_actuators()` authors
`NewtonActuator` prims only for actuator families Newton can parse natively in
this snapshot. GRU controllers and GRU residuals intentionally remain on the
local `anchor_bench.runtime.actuator_net_gru` compatibility shim because the
vendored Newton actuator schema supports neural MLP/LSTM controllers, not GRU.
Exported GRU TorchScript archives expose a `.gru` module for that shim rather
than Newton neural-actuator metadata.
