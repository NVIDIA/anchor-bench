
# Newton 1.2 Migration

Anchor Bench stores actuator configs in the native Newton 1.2 component shape.
Legacy `model_type` blocks are intentionally migrated rather
than copied verbatim.

Old categories map as follows:

| Legacy value | Native controller | Notes |
| --- | --- | --- |
| `implicit` | `pd` | Limits move to clamp components. |
| `dcmotor` | `pd` + `dc_motor` clamp | DC saturation is a clamp, not a model type. |
| `lstm` / `gru` | `lstm` / `gru` | Network file becomes controller metadata. |
| `hybrid_residual` | `pd` + residual component | PD remains base controller. |
| `fmu` | `fmu` | FMU path is controller metadata. |

Do not add new legacy `model_type` configs under `configs/robots/`.

Hybrid GRU residuals use the same GRU network contract as full-torque GRUs,
but live under an actuator `residual:` block. The base `controller.type` stays
`pd` so implicit solver PD remains active while the GRU contributes
`torque_residual` as feed-forward effort.
