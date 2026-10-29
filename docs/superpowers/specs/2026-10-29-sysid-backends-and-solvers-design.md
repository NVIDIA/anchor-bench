# Pluggable sysID backends and configurable Newton solvers

Date: 2026-10-29
Status: Approved on 2026-10-29; implementation plan in review

## Intent and agreed scope

Allow actuator system identification to select a simulator backend and, within
Newton, select and configure a physics solver. Deliver the work in independently
reviewable PRs. Implement Newton MuJoCo and VBD; leave additional real simulator
implementations and MPM outside this project.

Use the Newton fork at
`https://github.com/skulkar22/newton`, branch `dev/rigid-vbd-integration`.
The inspected revision is `1e5036c48ec92d44782dfe0f448dc25d64984b58`
(`1.8.0.dev0`). Lock this exact revision; updating the branch deliberately is a
dependency change, not an implicit change between runs.

User requirements:

- Keep the simulator boundary separate from Newton's solver selection.
- Accept solver-specific settings through a general YAML dictionary rather than
  duplicating all solver constructor options in handwritten configuration models.
- Warn about unknown/skipped solver settings. In recorded settings, replace that
  setting's value with a short string explaining why it was not set.
- Include the extension points needed for another simulator to supply sysID.
- Use external implicit PD actuation with effort clamping and bias feedback.
- Ignore the velocity limit in the new external actuator path.
- Preserve existing MuJoCo configuration and native-actuation behavior.

## Current architecture

`SimulationContext` owns one lazily loaded model and state per process. Its only
abstract operation is model loading; it does not define a complete stepping or
sysID execution protocol. `SIMULATION_CONTEXTS` currently registers only Newton.

`PopulationEvaluator.evaluate(population, trajectories)` is already independent
of the simulator. The objective and optimizer consume scores through this
protocol. However, `ActuatorSysIdTask` constructs `NewtonPopulationEvaluator`
directly, the simulation schema permits only Newton, and the evaluator checks
for `NewtonSimulationContext` before manipulating Newton/Warp arrays.

The Newton context hardcodes MuJoCo construction and attribute registration.
The evaluator maps three YAML options to MuJoCo constructor keywords and fixes
its internal algorithm to `solver="newton"`.

Common result writing currently invokes a Newton-specific USD exporter and
requires an `identified.usda` result. This coupling must move behind the backend
boundary as well as evaluator construction.

## Approaches and selected direction

Two architecture choices were considered:

1. Expand `SimulationContext` into a universal physics interface for every sysID
   array operation. This centralizes execution but forces different simulators
   and device layouts into one detailed model/state contract.
2. Keep `SimulationContext` responsible for resources, and select a backend's
   population evaluator through the existing scoring protocol. Keep solver
   integration beneath Newton. This preserves efficient native simulation and
   scoring while leaving the optimization algorithm independent of the backend.

The discussion selected the second direction, including the backend selection
work in this project. Do not introduce a universal array abstraction or duplicate
Newton's existing `SolverBase` lifecycle interface.

For numerical settings, use constructor-based discovery. Arbitrary post-build
attribute assignment is unsuitable: constructors resolve settings, allocate
buffers, and create cached values that assigning an attribute may not update.

## Backend boundary

Provide an explicit sysID backend registration containing:

- A backend name and configuration model.
- A population-evaluator factory implementing the existing scoring protocol.
- A simulation context factory compatible with resource ownership.
- An optional fitted-model export hook.

Registration must connect to the existing simulation registry rather than
requiring inconsistent declarations in unrelated catalogs. Validate duplicate
registrations and missing sysID capabilities before running optimization.
Support registration through the Python API and discovery through the
`anchor_bench.sysid_backends` entry-point group, so an installed backend can run
through the CLI without editing core source. Load registration metadata before
parsing its configuration, with deterministic duplicate-name errors.

Common sysID data remains joint names, candidate physical parameters, aligned
command/measurement trajectories, timing, and one finite position-MSE score per
candidate. A backend owns candidate application, state resets, actuation,
stepping, and score calculation. The objective, CMA-ES implementation, dataset
loader, delay semantics, and buffer exclusion are unchanged.

The simulation configuration dispatches on its existing `type` field. Keep
shared timing fields available to task invariant validation; Newton owns
`usd_path` and its solver block. Registered backend configuration models retain
their own validation and complete serialization. A custom backend's parsing and
execution must not construct or require a Newton context.

Common output files remain `config.yaml`, `best_params.yaml`,
`optimization_log.csv`, `run_summary.yaml`, and `.completed`. Backend-specific
model export is optional. Newton continues to write `identified.usda`; another
backend can write a different identified model or only the common results.
Represent optional model artifacts without requiring a USD path. Preserve the
existing Newton result access and filename for current callers.

Prove the boundary with a test backend using its own configuration, model/state
objects, deterministic population evaluation, and non-USD export. It must run
through configuration parsing, optimization, and common output writing without
constructing Newton objects. This is contract validation, not a second simulator
implementation.

## Newton solver selection and settings

Newton's backend owns a solver registry/factory and the small setup hooks needed
by each supported solver. Initial public names are `warp_mujoco` and `vbd`.
Unknown solver names fail before simulation setup. Settings never import an
arbitrary class named by a configuration string.

The preferred configuration is:

```yaml
simulation:
  type: newton
  usd_path: path/to/robot.usda
  physics_freq: 1000
  control_freq: 200
  render_freq: 60
  solver:
    type: vbd
    settings:
      iterations: 20
      rigid_compliant_alm: true
      rigid_articulation_solve: block_sparse_joints
```

Inspect the selected solver's public constructor to identify keyword parameters.
Exclude resource arguments owned by the context, including `model` and
`collision_pipeline`. Do not interpret an unrestricted `**kwargs` argument as
proof that an arbitrary option is supported. Normalize supported enum-valued
options from YAML using explicit public types; runtime resources are never
constructed from arbitrary YAML objects. Constructor failures for recognized
arguments remain errors and must identify the offending configuration path.

Keep legacy MuJoCo `integrator`, `solver_iterations`, and
`line_search_iterations`, translating them to the same construction path.
Preserve the current internal `solver="newton"` default for legacy configs.
If legacy and new settings specify the same argument, reject ambiguous duplicate
specification rather than silently choosing precedence.

Register selected-solver custom attributes before USD import and on population
builders. Preserve solver-specific attributes authored in USD. VBD colors the
population builder before finalization and checks articulation layout for sparse
modes. Arbitrary YAML overrides of per-joint or per-particle attributes are not
part of this first implementation; they are distinct from constructor options.

The initialized-context identity includes solver selection and accepted settings
alongside world count and builder configuration. Reusing the process singleton
with a different solver/configuration must fail clearly. Solver settings remain
fixed during a run; candidate physical parameters update between generations.

### Warning and provenance behavior

Warn once during setup for each unknown, internally managed, or demonstrably
ignored requested setting. Keep the input configuration intact. Add the compact
runtime solver-settings mapping to `run_summary.yaml`; accepted requested values
remain values, and skipped values become explanatory strings:

```yaml
simulation:
  type: newton
  solver:
    type: vbd
    settings:
      iterations: 20
      misspelled_option: "NOT SET: unknown constructor parameter"
```

Do not create a second failure list or copy every library default into the
summary. The dependency lock identifies defaults. Preserve the fork revision in
run metadata. Use documented ignored parameters and captured, attributable
Newton warnings to distinguish accepted-but-ignored options; a deprecation
warning alone does not prove an option is ignored. Do not claim to detect every
conditionally inactive setting from constructor introspection.

## Newton lifecycle and VBD capabilities

Use the solver's existing `step`, `reset`, `notify_model_changed`, and custom
attribute registration interfaces. Small Newton-owned handlers cover resource
construction, preparation, capability checks, and reset coordination, not a
second hierarchy duplicating every solver method.

Create an explicit `CollisionPipeline` for VBD before solver construction and
use the fork's solver-owned collision scheduling. Do not also run a duplicate
external collision pass. Reset collision matching as well as solver history
between independent trajectories. Preallocate/warm up required buffers outside
capture, then restore initial simulation, collision, and actuator state before
capturing or evaluating a trajectory.

The pinned fork publishes `state.joint_q` and `state.joint_qd` for revolute,
prismatic, and D6 joints, including continuous turns. Consume these directly;
do not overwrite continuous output with generic inverse kinematics. Reset from
measured joint coordinates with matching forward-kinematic body poses and
preserve full turn counts. Candidate/world isolation, delayed commands, scoring
alignment, and stable canonical state-buffer identities remain invariant.

Validate physical capabilities using solver mode, joint types, actual authored
model values, and configured candidate bounds. In this fork:

- Drive stiffness/damping and passive viscous damping support revolute,
  prismatic, and D6 joints.
- Dry friction under local compliant ALM supports static sticking. Sparse modes
  use regularized Coulomb friction, which has different near-zero behavior.
- Revolute armature requires a declared articulation solved with
  `block_sparse_joints` or `block_sparse_levels`; the local fallback must not be
  accepted for a fitted or nonzero armature.
- Sparse reduced-elastic articulation joints have further restrictions on
  armature, passive damping, friction, and mimics. Reject unsupported combinations.
- External actuation supplies effort clamping independently of VBD's unsupported
  native effort-limit property. Velocity limiting is intentionally unused.

Do not reject armature and friction merely because released Newton 1.4.0 lacked
them. Conversely, accepting a constructor setting does not establish that the
selected model/joint/mode supports every fitted physical parameter.

## External implicit PD actuation with bias

Add an explicit controller implementation choice: existing configurations default
to `native`; new external actuation is `external_implicit`. This is a design
proposal for the configuration spelling, not an existing field. Support the
external mode with MuJoCo and VBD so solver comparisons can use the same control
law. Keep native MuJoCo behavior as the compatibility baseline.

The supported matrix is MuJoCo/native, MuJoCo/external implicit, and VBD/external
implicit. VBD/native is outside this project. Selecting VBD with the compatibility
default `native` must fail before optimization with a message requiring
`controller.implementation: external_implicit`; do not switch it silently. The
VBD example explicitly sets this field for every identified joint group.

Use Newton's `Actuator`, `DrivePD`, `ClampingMaxEffort`,
`JointSpaceResponse`, and `Actuator.set_effort_mode_implicit`. Effort saturation
is enforced inside the implicit actuator solve, not after an explicit PD step.
Use one actuator with all optimized DOFs for each replicated model so the
implicit solve retains coupling within each articulation and isolation between
worlds. All joints sharing an articulation's coupled external controller must
use compatible implementation/options; reject a mixed native/external selection
within that articulation for the first implementation.

Provide optional `implicit_options` under an external controller for public
`Actuator.ImplicitOptions` values. Use the same recognized-option filtering and
compact skipped-setting reporting policy. External implicit mode always includes
bias prediction; do not silently fall back to explicit PD, omit bias, or use a
diagonal response.

Each physics substep performs:

1. Expose the current simulated joint/body state and delayed control targets.
2. Refresh the coupled joint-space inverse-mass response at the current state,
   including the current candidate's armature.
3. Evaluate `newton.eval_inverse_dynamics_passive` at the current simulated state
   into preallocated gravity and Coriolis arrays; form their sum as `bias_force`.
4. Clear actuator output effort, evaluate the implicit actuator with the physics
   timestep and `bias_force`, and write its clamped effort to `control.joint_f`.
5. Advance the selected Newton solver. Native PD is disabled on externally
   controlled DOFs; physical armature, dry friction, and viscous damping remain
   active in the physics solver.

The bias convention is `tau = M*qdd + b`, with `b = gravity + Coriolis`.
The actuator predicts `qd_next = qd + M_inverse * (impulse - dt*b)`. Passing
`bias_force` is not an additional gravity-compensation feedforward torque.
The physics solver applies its own gravity/Coriolis forces; do not add the bias
as another output torque. This interprets the user's bias-feedback requirement
as the pinned fork's implicit actuator bias input.

Use `JointSpaceResponse.refresh(state)` for both MuJoCo and VBD initially. This
constructs the full coupled articulation mass response with current armature,
keeps the controller prediction consistent across solvers, and avoids depending
on a MuJoCo factorization from the preceding step. This is a design choice for
review. It is a prediction model, not the exact constrained solver response:
contacts, friction, damping, joint limits, and loop constraints are not folded
into its inertia. The bias input similarly covers gravity and Coriolis, not
contact/limit reactions, passive joint damping, or other applied forces. Do not
claim equivalence to either solver's complete constrained dynamics. A future
solver-faithful inertia response can use `refresh_from_solve` once current-state
preparation and its behavior are independently validated.

The fork's passive inverse-dynamics helper allocates scratch per call. The
external mode therefore requires Warp's device memory pool for CUDA capture;
warm up and validate that requirement before capturing. If the memory pool is
unavailable, log the reason and execute the same full response/bias/actuator
sequence eagerly. Do not capture only the solver or silently omit bias to make
capture succeed.

Bind each candidate's gains to the actuator-owned parameter arrays after
implicit-mode binding; assigning the model's native PD arrays alone is
insufficient. Do not rebind arrays after graph capture. Capture response refresh,
bias evaluation, actuator stepping, and solver stepping as one substep sequence.
Reset any actuator history between independent trajectories. Keep the existing
trajectory command-delay implementation and do not apply a second actuator delay.

Ignore `velocity_limit` in external mode. Retain the legacy input field for
configuration compatibility, report `"NOT SET: velocity limit ignored"` in
runtime controller provenance, and do not add a DC-motor speed envelope or hard
speed constraint. The external model export must not author a stronger finite
velocity-limit opinion that implies it was enforced during fitting. Export
metadata must preserve the selected controller implementation and solver so
replay can reproduce its actuation semantics.

Replay acceptance means reimporting the exported fitted model and evaluating
its fitted parameters through the same sysID backend/evaluator with the recorded
solver/controller configuration. It does not add external-actuator or VBD
selection to the training replay validator. That validator remains on its
existing native MuJoCo path, with regression compatibility checked in stage 1.
Do not add a separate replay CLI or change the training validator's product
contract as part of this project.

## Delivery stages

Each stage is its own PR and must be mergeable with its tests and documentation.
The default execution is one stage at a time. Do not combine all product changes
into one PR or merge/publish stages merely because this design is approved.

| Stage | Change | Acceptance boundary |
| --- | --- | --- |
| 1 | Pin the Newton fork and migrate existing API usage | Existing native MuJoCo sysID and training/replay workflows work on the exact fork revision; no new solver exposed yet. |
| 2 | Register sysID backends, dispatch configuration/evaluator, and isolate model export | Existing Newton behavior remains; an installed test backend runs the complete CLI flow without Newton context construction or USD-only output requirements. |
| 3 | Add Newton solver factory, general settings, legacy translation, and compact runtime provenance | MuJoCo runs through the new factory; known options reach construction, unknown/ignored options warn and appear as strings, and singleton configuration conflicts are rejected. |
| 4 | Add external implicit PD, coupled response, gravity/Coriolis bias, effort clamp, and reset/capture support | MuJoCo external mode has verified saturation, live candidate gains/armature, coupled multi-DOF response, bias sign, and independent trajectory/world behavior. Native mode remains compatible. |
| 5 | Enable VBD with its preparation, collision/reset integration, capability checks, and example | VBD sysID runs with external implicit PD and effort clamping, supported fitted physical parameters, repeatable replay/export, and validated eager/graph execution. |

An installed third-party simulator, a universal simulator tensor API, MPM,
explicit external PD, velocity limiting, generic YAML model-attribute overrides,
and equivalence of all solver physics are non-goals. BDF2 may be configured only
where the fork supports it; adding a separate BDF2 feature project is unnecessary.

## Validation

Use meaningful behavior tests at each boundary, not only constructor mocks:

- Existing native MuJoCo configuration, result exports, and score alignment.
- Backend discovery/registration, strict custom schema round-trip, config errors,
  evaluator selection, optional/non-USD exports, and Newton-independent execution.
- Actual constructor-option forwarding and errors, legacy compatibility, unknown
  and documented ignored settings, provenance without input-config mutation.
- PD effort saturation under large position error, implicit coupled response
  versus an analytic small articulation, and gravity/Coriolis bias sign and
  effect. Include nonzero armature and multiple replicated worlds.
- Candidate updates visible during CUDA graph replay, independent trajectory
  reset, history restoration after warm-up/capture, and eager/graph agreement.
- VBD friction/damping changes, sparse-mode armature, unsupported model/mode
  rejection, continuous joint coordinates, fresh collisions after reset, and
  absence of duplicate native PD forces.
- A short real SO-101 sysID run per supported mode and replay of exported
  parameters using the recorded solver/controller configuration. Do not require
  identical fitted parameters or trajectories across different solvers.

The repository gates apply to every code/dependency stage: `uv run ruff check`,
`uv run ruff format --check`, `uv run pyright`, and appropriate pytest coverage.
Because the dependency migration and simulation changes affect shared
infrastructure and integrations, run the complete unit and integration suite
with `uv run pytest -o "addopts="` for those stages. CUDA claims require actual
CUDA validation; document any unavailable hardware or unrelated baseline failure.
Documentation-only design work does not run these gates.

## Builder decisions and follow-up checks

The builder may choose file layout, precise registration function signatures,
optional-result compatibility helpers, and test fixture structure. The proposed
controller field names and entry-point group are design choices for review.

Confirm contact capacity warm-up and graph capture restore all solver-owned
history, not just the visible state arrays. The chosen generic response path is
part of the external controller's recorded runtime implementation; no stale
MuJoCo solve or unrecorded controller-mode fallback is allowed.

The inspected fork source and its tests support these design claims, but have
not been installed or executed during design research. Stage 1 establishes the
runtime baseline before subsequent feature work.

## Source evidence

- [Pinned fork revision](https://github.com/skulkar22/newton/commit/1e5036c48ec92d44782dfe0f448dc25d64984b58).
- [VBD capabilities, settings, collision and reset behavior](https://github.com/skulkar22/newton/blob/1e5036c48ec92d44782dfe0f448dc25d64984b58/newton/_src/solvers/vbd/solver_vbd.py).
- [External implicit actuation and bias conventions](https://github.com/skulkar22/newton/blob/1e5036c48ec92d44782dfe0f448dc25d64984b58/docs/concepts/actuators.rst).
- [Coupled joint-space response](https://github.com/skulkar22/newton/blob/1e5036c48ec92d44782dfe0f448dc25d64984b58/newton/_src/actuators/joint_space_response.py).
- [Actuator, bias, clamping, and live-update tests](https://github.com/skulkar22/newton/blob/1e5036c48ec92d44782dfe0f448dc25d64984b58/newton/tests/test_actuators.py).
- [Sparse articulation and armature tests](https://github.com/skulkar22/newton/blob/1e5036c48ec92d44782dfe0f448dc25d64984b58/newton/tests/test_vbd_sparse_articulation.py).
