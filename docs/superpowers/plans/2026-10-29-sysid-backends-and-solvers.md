# Pluggable sysID Backends and Newton Solvers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver simulator-pluggable sysID, general Newton solver settings, and VBD with external implicit PD, bias prediction, and effort clamping through five separate PRs.

**Architecture:** Keep model/state ownership in `SimulationContext` and backend-specific population evaluation behind `PopulationEvaluator`. Put Newton solver construction and preparation beneath `NewtonSimulationContext`; use Newton's actuator and solver lifecycle APIs rather than a universal physics-array interface.

**Tech Stack:** Python >=3.10, Pydantic 2, Newton fork `1.8.0.dev0`, Warp >=1.17.0, MuJoCo/MuJoCo Warp ~=3.16.0, Optuna/CMA-ES, OpenUSD, pytest, uv.

**Spec:** [Approved design](../specs/2026-10-29-sysid-backends-and-solvers-design.md).

## Global Constraints

- Pin `https://github.com/skulkar22/newton`, branch provenance `dev/rigid-vbd-integration`, exact revision `1e5036c48ec92d44782dfe0f448dc25d64984b58`. Never substitute a moving branch head.
- Preserve existing MuJoCo configuration and native-actuation behavior.
- Supported new matrix: MuJoCo/native, MuJoCo/external implicit, VBD/external implicit. VBD/native fails before optimization.
- External mode uses `JointSpaceResponse.refresh(state)` for both solvers, gravity plus Coriolis as `bias_force`, and `ClampingMaxEffort` inside the implicit solve. Ignore velocity limiting.
- Use one coupled actuator over the identified DOFs; do not split coupled coordinates into independent scalar actuator solves. Keep worlds isolated.
- Unknown, internally managed, and demonstrably ignored options warn once; replace their values with `"NOT SET: <reason>"` in runtime provenance. Preserve input configuration. Constructor errors for accepted settings remain errors.
- No MPM, real second simulator, explicit external PD, generic YAML model-attribute overrides, new replay CLI, or training-validator feature expansion.
- Preserve command-delay indexing, buffer exclusion, canonical state identity, native Newton USD outputs, and common result files.
- Use `vidur/` branches, authenticated `gh` for GitHub operations, and no attribution or co-author trailers to automated tools.
- All new/touched code headers use the 2026 repository licensing header. Run the repository gates before committing applicable code/dependency changes, including the complete unit/integration gate for shared infrastructure and dependency commits. A later stage's planned checks cannot justify an earlier unchecked commit.

## Review Focus

1. Plugin-only settings must survive strict parsing, serialization, configuration hashing, and tracking; test this in Task 3.
2. A recognized but invalid constructor argument must not be mislabeled as an ignored typo; test actual constructor failure and duplicate aliases in Task 5.
3. New solver/model configuration must not silently reuse a process singleton or stale captured graph; test this in Tasks 3 and 6.
4. Candidate gains and armature must affect captured implicit control without duplicate native PD; test this in Tasks 7 and 8.
5. Trajectory resets must discard solver/contact/actuator history, preserve full turns, and retain bias updates; test this in Tasks 8 and 9.

## Delivery and parallel execution

| PR | Branch | Tasks | Mergeable outcome |
| --- | --- | --- | --- |
| 1 | `vidur/sysid-newton-pin` | 1–2 | Existing native workflows run on the pinned fork. |
| 2 | `vidur/sysid-backends` | 3–4 | An installed test backend runs sysID without Newton model construction or USD-only output. |
| 3 | `vidur/sysid-solver-settings` | 5–6 | MuJoCo uses the solver factory and general settings with compact runtime provenance. |
| 4 | `vidur/sysid-implicit-pd` | 7–8 | External implicit MuJoCo PD includes coupled response, bias, and effort limits. |
| 5 | `vidur/sysid-vbd` | 9 | VBD sysID, example, and fitted-model replay work through the same backend. |

Execute and review one PR stage at a time. The first product branch includes the approved design/plan commits; later stages start from the accepted preceding stage. Do not merge PRs automatically or implement later stages against unreviewed API assumptions.

Parallel agents are explicitly requested. Assign disjoint file ownership within a stage and parallelize independent research, regression work, and review. The controller owns dependency files and integration. Tasks sharing an interface or source file wait for that interface to land; do not run all five stages as concurrent edits. Reuse a suitable clean checkout; additional worktrees are only needed if isolation actually requires them, using the native worktree workflow.

At each stage finish, run the applicable gates, obtain independent code review, create a concrete draft PR with `gh`, and attach its URL to the task. PR titles/descriptions describe that stage only. A stage is not complete merely because a plan checkbox is checked.

## File and interface map

| File | Responsibility |
| --- | --- |
| `core/simulation/__init__.py` | Simulator-neutral resource identity and lazy model/state ownership. |
| `core/simulation/registry.py` | Authoritative backend registrations, sysID capability, and installed-provider discovery. |
| `schema/simulation.py` (new) | Shared sysID timing configuration and Newton-specific simulation/solver configuration. |
| `tasks/actuator_sys_id/backends/newton.py` (new) | Lazy Newton context/evaluator/export/provenance adapters. |
| `tasks/actuator_sys_id/task.py` | Select backend, optimize, request optional export and runtime description. |
| `tasks/actuator_sys_id/artifacts.py`, `model.py` | Common outputs and optional backend model artifacts. |
| `core/simulation/newton_options.py` (new) | Recognize public construction options and record skipped settings. |
| `core/simulation/newton_solvers.py` (new) | Newton solver descriptors, preparation, and owned resources. |
| `core/simulation/newton.py`, `newton_stepper.py` | Newton model lifecycle, canonical stepping, capture, and reset coordination. |
| `tasks/actuator_sys_id/newton_actuation.py` (new) | Coupled external implicit actuator, bias, live parameters, and history. |
| `tasks/actuator_sys_id/newton_capabilities.py` (new) | Validate VBD mode/joint/candidate/model capability combinations. |
| `tasks/actuator_sys_id/simulation.py`, `usd.py` | Newton evaluation and fitted USD semantics. |

Paths above are relative to `anchor_bench/`. Tests and documentation are named in their owning tasks below. Keep existing imports available through re-exports where the original name is public.

## PR 1: Pinned Newton runtime

### Task 1: Pin the fork and migrate model construction

**Files:** Modify `pyproject.toml`, `uv.lock`, `THIRD_PARTY_LICENSES.md`, `anchor_bench/core/simulation/newton.py`, `tests/test_newton_simulation_context.py`, `tests/integration/usd_trajectory.py`, `docs/system-identification.md`.

**Interfaces:** Preserve `NewtonSimulationContext(usd_path: str, *, fixed_base: bool=False)` and `initialize(world_count=...)`. Internally use `builder.replicate(template, copy_count=world_count)` and an owned `CollisionPipeline(model, broad_phase="explicit")` with `pipeline.contacts()`.

- [x] **Step 1: Write migration regressions.** Extend `test_initializes_one_configurable_replicated_simulation` and `test_lazily_builds_one_newton_model_and_state`: record `copy_count`, one pipeline, contacts identity, attributes registered before import/finalize, and unchanged MuJoCo options. Have fakes reject the deprecated `world_count` replicate keyword and deprecated `model.contacts()` path.
- [x] **Step 2: Run the red tests.** `uv run pytest tests/test_newton_simulation_context.py -q`; require failures at the old construction calls. Record dependency/version availability separately rather than mistaking installation failure for a behavioral red test.
- [x] **Step 3: Pin and migrate.** Set the dependency to `newton[importers,sim] @ git+https://github.com/skulkar22/newton.git@1e5036c48ec92d44782dfe0f448dc25d64984b58`; run `uv lock` and `uv sync --locked --extra dev`. Use the public construction APIs above; remove the obsolete `newton.use_coord_layout_targets` assignment from the integration helper. Do not add external collision stepping to native MuJoCo.
- [x] **Step 4: Verify resolution and tests.** Run `uv lock --check`, the Task 1 tests, and inspect Newton's installed `direct_url.json`: `vcs_info.commit_id` must equal the full SHA and package version must be `1.8.0.dev0`. Keep importer/sim extras. Resolve root constraints for Warp >=1.17.0, MuJoCo/MJWarp ~=3.16.0, and schemas >=0.5.0; update affected third-party version/source rows from the actual lock. Do not assume the dependency's uv index settings propagate to the root.
- [x] **Step 5: Gate and commit.** Run the applicable lint/type gates plus targeted construction tests. Commit as `build: pin Newton rigid VBD integration revision` including migration and license-manifest changes. Do not expose VBD or change native actuation in this task.

### Task 2: Verify real native sysID, replay, and training on the fork

**Files:** Create `tests/integration/test_newton_native_runtime.py`; extend `tests/test_newton_replay_adapter.py` or its real-runtime integration counterparts; modify `anchor_bench/tasks/train/validators/newton_replay_adapter.py`, `anchor_bench/tasks/train/data_processors/gru.py`, `anchor_bench/core/simulation/newton_stepper.py`, and `anchor_bench/tasks/actuator_sys_id/simulation.py` only for demonstrated fork incompatibilities. Preserve existing SO-101/G1 integration assertions.

**Interfaces:** Existing candidate application, `notify_model_changed`, measured-pose reset, direct/residual training replay, and CPU bias mappings remain native MuJoCo contracts. Reuse the existing `NewtonStepExecutor` public API.

- [ ] **Step 1: Add physical tests.** `test_native_mujoco_candidate_updates_change_each_world_response` uses two scalar-joint worlds and swaps candidates, asserting finite scores and swapped world responses. `test_native_mujoco_repeated_trajectory_reset_is_deterministic` evaluates A/B/A and compares repeated A, delay alignment, and buffer exclusion. `test_native_mujoco_cuda_graph_matches_eager_with_live_candidates` covers odd/even substeps, stable `state` identity, and live gain/armature/target updates.
- [ ] **Step 2: Establish the fork baseline.** Run `uv run pytest -m integration tests/integration/test_newton_native_runtime.py -q`. Add `test_native_replay_direct_and_residual_reset` and `test_offline_bias_matches_live_replay_bias` with nonzero q/qd and MuJoCo `dof_ref`; compare the latter against `mj_forward(...).qfrc_bias`. New regression tests may already pass because this is a compatibility upgrade; do not manufacture failures or change physics to make them red.
- [ ] **Step 3: Fix only demonstrated API incompatibilities.** Preserve native target arrays, modes, reset order, and `joint_target_q_start` indexing. Keep the training replay product on native MuJoCo. Warm up real graphs outside capture and restore state before measuring.
- [ ] **Step 4: Run Stage 1 acceptance.** Run the complete gate below, including existing SO-101 two-generation/six-world export-MSE and held-out accuracy assertions plus G1 residual/full-torque training. Keep SO-101 MSE tolerance `rel=1e-4, abs=1e-7`. Diagnose changed outcomes instead of lowering thresholds. GPU-only claims require observed CUDA results.
- [ ] **Step 5: Review, commit, and prepare PR 1.** Commit compatibility coverage/fixes as `test: verify native simulation against pinned Newton`. Use independent agents on native sysID versus training/bias tests with disjoint files. Review the combined branch, then prepare its own draft PR and attach it.

## PR 2: Simulator-pluggable sysID

### Task 3: Generalize resources and register typed backend capabilities

**Files:** Modify `anchor_bench/core/simulation/__init__.py`, `newton.py`, `registry.py`, `anchor_bench/schema/actuator_sys_id.py`, `anchor_bench/tasks/actuator_sys_id/model.py`, `anchor_bench/errors.py`; create `anchor_bench/schema/simulation.py`, `tests/test_simulation_context.py`, `tests/test_sysid_backends.py`; extend `tests/test_actuator_sys_id_config.py`, `tests/test_simulation_registry.py`, and test-only context subclasses in `tests/test_newton_simulation_context.py`, `tests/test_data_processors.py`, `tests/test_data_processor_runner.py`. Make package initializers lazy only where needed to remove import cycles.

**Interfaces:**

```python
SimulationContext.__new__(cls, *, resource_key: Hashable) -> Self
SimulationContext.__init__(self, *, resource_key: Hashable) -> None
SimulationContext.resource_key -> Hashable
SimulationContext._load(self) -> tuple[ModelT, StateT]
```

Move file resolution/validation, `usd_path`, and `fixed_base` to `NewtonSimulationContext`; its existing public constructor calls the base with `(resolved_path, fixed_base)`. Preserve the existing training adapter `create_simulation_context(backend, usd_path, *, fixed_base=False)`.

Define `SysIdSimulationConfig(type: str, physics_freq, control_freq, render_freq)` with common timing validation. Move `WarpMujocoSolverConfig` and `NewtonSimulationConfig` to the neutral schema module and re-export their old names; Newton's config extends the common model with `type: Literal["newton"]`, `usd_path`, and existing solver fields. `ActuatorSysIdConfig.simulation` uses `SerializeAsAny[SysIdSimulationConfig]` and registered strict dispatch before validation. Add frozen `IdentifiedModelArtifact(path: Path, format: str)` to `model.py` now so registry annotations are defined before Task 4 consumes them.

In `core/simulation/registry.py` define frozen `SimulationBackend(name, context_type, sysid: SysIdBackend|None)` and `SysIdBackend(config_model, create_context, create_evaluator, export_model=None, describe_runtime=None)`. Factories have these signatures:

```python
create_context(config: ActuatorSysIdConfig) -> SimulationContext[Any, Any]
create_evaluator(config: ActuatorSysIdConfig, space: ParameterSpace,
                 context: SimulationContext[Any, Any]) -> PopulationEvaluator
export_model(config: ActuatorSysIdConfig, best: Candidate,
             context: SimulationContext[Any, Any], output: Path
             ) -> IdentifiedModelArtifact | None
describe_runtime(evaluator: PopulationEvaluator) -> Mapping[str, JsonValue]
register_simulation_backend(registration: SimulationBackend) -> None
get_sysid_backend(name: str) -> SysIdBackend
discover_sysid_backends() -> None
```

`JsonValue` is Pydantic's public recursive JSON-compatible alias, re-exported by the neutral schema module. Use postponed annotations and `TYPE_CHECKING` to avoid schema/registry/task cycles.

- [ ] **Step 1: Write boundary tests.** `test_path_free_context_loads_once` constructs `ToyContext(resource_key=("toy", 2))` without a file and asserts model/state cache identity. Add incompatible-class/key and unhashable-key cases. `test_plugin_simulation_roundtrips_without_losing_fields` validates a typed toy config, rejects extras/wrong types, preserves fields through JSON and config hashing, and detects changed plugin values.
- [ ] **Step 2: Run red tests.** `uv run pytest tests/test_simulation_context.py tests/test_sysid_backends.py tests/test_actuator_sys_id_config.py -q`; require missing neutral-resource/registration behavior to fail.
- [ ] **Step 3: Implement atomic registration and discovery.** Entry-point group is `anchor_bench.sysid_backends`; providers are zero-argument callables returning `SimulationBackend`. Sort names, reject duplicate metadata names before loading, require entry-point and registration names to match, reject duplicate built-in/Python names, and wrap errors in `SimulationBackendError(AnchorBenchError)`. Discover before both direct `model_validate` and YAML parsing; mark discovery successful only after all registrations validate. A context factory returning the wrong concrete registered type is an error.
- [ ] **Step 4: Verify shared-resource compatibility.** Add `test_discovery_is_idempotent`, `test_discovery_error_does_not_leave_partial_registration`, `test_registered_backend_without_sysid_fails_early`, and repeated singleton-incompatibility tests. Run all Task 3 files plus data-processor/context tests. Imports must not construct Newton objects or import the sysID task from the neutral registry.
- [ ] **Step 5: Gate and commit.** Run lint/format/type checks and Task 3 tests; commit as `refactor: register simulation resources and sysid capabilities`.

### Task 4: Dispatch evaluation, isolate export, and prove installed-plugin CLI use

**Files:** Create `anchor_bench/tasks/actuator_sys_id/backends/__init__.py`, `backends/newton.py`, `tests/integration/test_sysid_backend_cli.py`, and `tests/fixtures/sysid_backend/`; modify `task.py`, `simulation.py`, `artifacts.py`, `model.py` in that task package, `tests/test_actuator_sys_id_task.py`, and `docs/system-identification.md`.

**Interfaces:** Consume Task 3's `IdentifiedModelArtifact`. Add `identified_model_path: Path|None` and retain optional `identified_usd_path` in `SysIdResult`; explicitly declared format `"usd"` supplies the compatibility path, never suffix inference. Extend the existing `write_artifacts(..., config, model_artifact: IdentifiedModelArtifact|None=None, runtime_simulation: Mapping[str,JsonValue]|None=None)` signature. It never imports `usd.py`. Newton's export callback invokes the existing USD writer and returns `format="usd"`.

Preserve `PopulationEvaluator.evaluate` as its only required method. Newton's factory injects the selected context into `NewtonPopulationEvaluator(config, space, *, simulation_context=None)`; preserve its existing two-positional-argument use. The task selects one backend and context, constructs the evaluator, optimizes, invokes its optional exporter/runtime description, and writes common artifacts.

- [ ] **Step 1: Write orchestration/export regressions.** `test_task_uses_registered_evaluator_and_context` asserts selected factories receive the same context and optimizer behavior remains unchanged. Parameterize `test_common_artifacts_allow_missing_or_json_model_export`; assert optional result paths and no forced USD import. `test_export_failure_prevents_completion` asserts no `.completed` marker after exporter failure.
- [ ] **Step 2: Run red tests.** `uv run pytest tests/test_actuator_sys_id_task.py tests/test_sysid_backends.py -q`; require the hardcoded Newton task/export path to fail toy-backend expectations.
- [ ] **Step 3: Implement the lazy Newton adapter and generic task path.** Re-export moved Newton config names for existing callers. Keep common scores, history, MLOps upload/completion ordering, delay, and datasets unchanged. A missing sysID capability must fail before constructing the optimizer or evaluating a population.
- [ ] **Step 4: Prove real installed discovery.** The integration fixture writes an importable toy package and `.dist-info/METADATA` plus `entry_points.txt` into `tmp_path`, then launches a fresh CLI subprocess with that path. Use real Parquet and CMA-ES; guard Newton context/model construction to raise. Assert plugin-specific config values survive, scores are finite/repeatable, common outputs complete, JSON export appears, and `identified.usda` does not. Repeat with no exporter. Run `uv run pytest -m integration tests/integration/test_sysid_backend_cli.py tests/integration/so101_actuator_sysid/test_cli.py -q`.
- [ ] **Step 5: Gate, commit, review, and prepare PR 2.** Run the full gate; document the Python registration API and entry-point provider contract. Commit as `feat: run sysid through registered simulator backends` and prepare the separate PR.

## PR 3: General solver settings

### Task 5: Resolve constructor settings and preserve legacy MuJoCo YAML

**Files:** Create `anchor_bench/core/simulation/newton_options.py`, `newton_solvers.py`, `tests/test_newton_solver_options.py`, `tests/test_newton_solvers.py`; modify `anchor_bench/schema/simulation.py` and solver-config tests.

**Interfaces:** Define frozen `OptionResolution(kwargs: dict[str, object], recorded: dict[str, JsonValue])` and:

```python
resolve_constructor_options(constructor: Callable[..., object],
    requested: Mapping[str, JsonValue], *, path: str,
    managed: frozenset[str],
    enum_converters: Mapping[str, Callable[[JsonValue], object]],
    ignored: Mapping[str, str]) -> OptionResolution
get_newton_solver(name: str) -> NewtonSolverDescriptor
resolve_newton_solver_settings(config: NewtonSolverConfig) -> OptionResolution
```

`NewtonSolverConfig` accepts `type: str`, JSON-compatible `settings`, and optional legacy MuJoCo fields. Use explicit field-presence tracking for alias conflicts; do not inject legacy defaults into VBD or generic settings. Keep legacy runtime defaults `integrator="implicitfast"`, `solver="newton"`, and existing required iteration settings for legacy syntax.

`NewtonSolverDescriptor` contains name, solver class, `register_attributes(builder)`, `prepare_builder(builder)`, and `build_resources(model, options)`; Task 6 defines the resource record. Initially register only `warp_mujoco`.

- [ ] **Step 1: Write option tests.** `test_known_options_reach_constructor` asserts exact keyword forwarding. `test_unknown_managed_and_documented_ignored_options_warn_once` asserts omission plus string replacement. `test_kwargs_does_not_accept_arbitrary_names` rejects names merely hidden by `**kwargs`. `test_legacy_aliases_preserve_mujoco_arguments` asserts original option dictionary; `test_duplicate_legacy_and_settings_argument_fails` rejects even equal duplicate values.
- [ ] **Step 2: Run red tests.** `uv run pytest tests/test_newton_solver_options.py tests/test_newton_solvers.py tests/test_actuator_sys_id_config.py -q`.
- [ ] **Step 3: Implement public-option resolution.** Exclude `self`, positional-only parameters, `*args`, `**kwargs`, and context-owned resources including `model`/`collision_pipeline`. Use explicit public enum converters for supported enum settings; never load runtime objects from arbitrary strings. Unknown solver names fail. Maintain a small pinned-version map only for documented ignored options; do not mark an option ignored merely because it is deprecated.
- [ ] **Step 4: Verify error boundaries.** Add `test_recognized_invalid_value_is_constructor_error`, asserting a supported option rejected by the real constructor fails with the configuration path and original reason. Verify warnings and provenance never mutate the input dictionary, and unrelated Newton warnings remain visible.
- [ ] **Step 5: Gate and commit.** Run quality gates and Task 5 tests; commit as `feat: resolve generic Newton solver settings`.

### Task 6: Integrate solver resources, context identity, and compact provenance

**Files:** Modify `anchor_bench/core/simulation/newton.py`, `newton_stepper.py`, `newton_solvers.py`, `anchor_bench/tasks/actuator_sys_id/simulation.py`, `backends/newton.py`, `tests/test_newton_simulation_context.py`, `tests/test_actuator_sys_id_simulation.py`, `tests/test_actuator_sys_id_task.py`, and `docs/system-identification.md`.

**Interfaces:** Define frozen `NewtonSolverResources(solver: SolverBase, contacts: Contacts|None, step_contacts: Contacts|None, collision_pipeline: CollisionPipeline|None)`. Context initialization adds `solver_type: str="warp_mujoco"` and optional `solver_resolution: OptionResolution|None` while preserving `solver_options`; supplying both option forms is an error. The evaluator passes Task 5's resolution, while direct callers' raw `solver_options` are resolved inside the context. Resolve once before resource creation and retain `solver_settings: Mapping[str,JsonValue]` for reporting. The executor's step argument comes from `step_contacts`, independently of inspected `context.contacts`.

Newton's backend `describe_runtime(evaluator)` returns the compact runtime simulation configuration. The common writer nests it in `run_summary.yaml`; accepted requested values stay values and omitted values become strings. Record the exact Newton revision without duplicating every constructor default or adding separate failure lists.

- [ ] **Step 1: Write context/report tests.** `test_solver_attributes_are_registered_before_import_and_replication` checks both builders. `test_context_rejects_changed_solver_or_accepted_options` asserts singleton protection. `test_runtime_solver_settings_replace_only_skipped_values` checks output YAML and unchanged input/config hash. `test_same_accepted_options_reuse_context` covers equivalent requests with skipped keys.
- [ ] **Step 2: Run red tests.** Run modified context/evaluator/task tests plus Task 5 files; require hardcoded construction or missing provenance behavior to fail.
- [ ] **Step 3: Use descriptor resources throughout Newton setup.** Type solver ownership as `SolverBase`. Register attributes before import/replication, prepare the replicated builder before finalization, and include solver identity/normalized accepted values in singleton compatibility. Never expose VBD until Task 9 supplies its preparation and validation.
- [ ] **Step 4: Run real MuJoCo settings acceptance.** Parameterize a short native runtime over legacy/new settings forms and compare forwarded settings and scores with the original behavior. Run the native runtime integration, installed toy-backend integration, and SO-101 integration. Validate constructor errors prevent optimization/completion.
- [ ] **Step 5: Gate, commit, review, and prepare PR 3.** Run full acceptance and document generic YAML/legacy compatibility. Commit as `feat: configure Newton solvers through shared construction` and prepare the separate PR.

## PR 4: External implicit PD with bias

### Task 7: Build the coupled external actuator and bind live candidates

**Files:** Create `anchor_bench/tasks/actuator_sys_id/newton_actuation.py`, `tests/test_newton_sysid_actuation.py`; modify controller schema in `anchor_bench/schema/actuator_sys_id.py` and its tests.

**Interfaces:** Add `PdControllerConfig.implementation: Literal["native", "external_implicit"]="native"` and optional JSON `implicit_options` valid only for external mode. Reject mixed implementations/options within one coupled articulation for this release. Define:

```python
ExternalImplicitPD.build(model: newton.Model, config: ActuatorSysIdConfig, *,
    joint_names: Sequence[str], q_indices: np.ndarray,
    dof_indices: np.ndarray, target_indices: np.ndarray) -> ExternalImplicitPD
ExternalImplicitPD.apply_population(population: Sequence[Candidate]) -> None
ExternalImplicitPD.reset() -> None
ExternalImplicitPD.before_substep(state: newton.State,
    control: newton.Control, dt: float) -> None
ExternalImplicitPD.runtime_metadata() -> Mapping[str, JsonValue]
ExternalImplicitPD.supports_graph_capture() -> tuple[bool, str | None]
```

- [ ] **Step 1: Write analytic actuator tests.** `test_implicit_pd_matches_scalar_solution_with_bias` asserts the uncapped scalar effort against the equation below. `test_implicit_pd_effort_is_bounded_inside_solve` uses large error and nonzero bias to assert `abs(effort) <= effort_limit`. `test_implicit_pd_uses_off_diagonal_response` compares a two-DOF articulation with a dense coupled oracle, not independent scalar PD.

  For the unclamped oracle, use `A=M_inverse`, `K=diag(dt*kp+kd)` and solve `(I + dt*K*A)*tau = kp*(target_q-q) + kd*target_qd - K*qd + dt*K*A*bias`. Use small deterministic articulated models; compare the actuator effort, not exact constrained simulator trajectories.
- [ ] **Step 2: Run red tests.** `uv run pytest tests/test_newton_sysid_actuation.py tests/test_actuator_sys_id_config.py -q` must fail because the external mode is absent.
- [ ] **Step 3: Implement through Newton's public actuator API.** Construct full coupled `JointSpaceResponse`, `DrivePD`, `ClampingMaxEffort`, and one `Actuator` across selected DOFs/worlds. Call `set_effort_mode_implicit` before storing gain references; write candidates into the resulting `drive.kp`/`drive.kd` views. Keep separate position, DOF, target indices and preallocated full-DOF gravity/Coriolis/bias arrays. Do not add DC-motor velocity limiting or a second command delay.
- [ ] **Step 4: Verify updates and provenance.** Add `test_bound_gain_views_remain_live_after_implicit_binding`, `test_response_refresh_includes_updated_armature`, `test_bias_uses_current_state_without_mutating_body_poses`, and `test_external_velocity_limit_is_recorded_as_ignored`. Assert the implicit options use the same skip/error policy as solver options. Check no extra gravity-compensation torque is added.
- [ ] **Step 5: Gate and commit.** Run quality gates, controller/config tests, and real actuator tests; commit as `feat: add coupled implicit PD with bias and effort limits`.

### Task 8: Integrate full physics-substep capture, reset, and fitted-model replay

**Files:** Modify `anchor_bench/core/simulation/newton_stepper.py`, `newton.py`, `anchor_bench/tasks/actuator_sys_id/simulation.py`, `usd.py`, `backends/newton.py`; extend `tests/test_newton_step_executor.py`, `tests/test_actuator_sys_id_simulation.py`, `tests/test_actuator_sys_id_usd.py`; create `tests/integration/test_sysid_external_pd.py` and `tests/integration/sysid_replay.py`; update `docs/system-identification.md`.

**Interfaces:** The executor accepts optional `before_substep: Callable[[State, Control, float], None]` and calls it after clearing forces, before every solver step, both eager and captured; supplying it without a control buffer is an error. Add `restore_state: Callable[[], None]|None` to capture orchestration and `can_capture: Callable[[], tuple[bool,str|None]]|None`; native callers remain compatible. Expose context `configure_step_callbacks(*, before_substep, restore_state, can_capture, configuration_key: Hashable) -> None` to attach these once before capture. Reject replacement by different callbacks after attachment; do not retain a stale actuator owner or graph. Include controller implementation and accepted implicit options in the evaluator's builder/runtime configuration key. `ExternalImplicitPD.before_substep` executes response refresh, passive inverse dynamics, bias sum, effort clear, and implicit actuator step with physics dt. Replay helper runs in a fresh process to avoid source/export singleton collisions.

- [ ] **Step 1: Write lifecycle regressions.** `test_external_pd_runs_before_each_physics_substep` asserts full ordering and dt. `test_external_candidate_updates_do_not_restore_native_pd` asserts native ke/kd stay zero while external gains change. `test_bias_and_clamp_are_present_in_captured_substeps` asserts numerical output, not just callback invocation. `test_capture_restores_initial_state_and_histories` compares repeated independent runs. `test_context_rejects_native_external_reconfiguration_or_hook_replacement` prevents stale callback/gain ownership when the USD and world count are unchanged.
- [ ] **Step 2: Run red tests.** Run the modified stepper/evaluator tests and new external-PD integration. Existing native tests must continue to pass with callbacks omitted.
- [ ] **Step 3: Wire the external path.** Disable native gains on external DOFs (MuJoCo effort mode; VBD cannot rely on target mode). Keep physical armature/friction/damping updates in the model. Retain delayed command targets at control rate and recompute response/bias/effort at every physics substep. Warm up full intervals outside capture, restore both visible state buffers/control plus solver/contact/actuator history before capture and again afterward. Reset output efforts between substeps and histories between trajectories.
- [ ] **Step 4: Verify physical, graph, and export behavior.** Add two-world A/B/A and live-candidate CUDA replay cases; compare eager/graph results for odd/even substeps. Require `device.is_cuda` and `wp.is_mempool_enabled(device)` for capture of bias scratch allocations; when unavailable, log the reason and run the entire sequence eagerly. Persist external mode/options/ignored velocity metadata. External export must not author a stronger finite velocity opinion. Reimport the exported USD in a fresh process with recorded config and fitted parameters, tile the candidate across worlds, and compare MSE to the recorded optimum. Keep the existing native replay helper and training validator unchanged.
- [ ] **Step 5: Gate, commit, review, and prepare PR 4.** Run full acceptance including native and external MuJoCo plus installed backend integration. Commit as `feat: run sysid with external implicit PD actuation` and prepare its separate PR.

## PR 5: VBD sysID

### Task 9: Add VBD resources, capability validation, and real sysID acceptance

**Files:** Modify `anchor_bench/core/simulation/newton_solvers.py`, `newton.py`, `anchor_bench/tasks/actuator_sys_id/simulation.py`, `backends/newton.py`; create `newton_capabilities.py` in that task package, `tests/test_newton_sysid_capabilities.py`, `tests/integration/test_sysid_vbd.py`, and `examples/system_identification/so101_vbd.yaml`; extend solver/context/reset tests and `docs/system-identification.md`.

**Interfaces:** Add `validate_newton_sysid_capabilities(config: ActuatorSysIdConfig, model: newton.Model, *, joint_names: Sequence[str], dof_indices: np.ndarray) -> None`. Add descriptor `vbd`: register its attributes before USD import, color the replicated builder before finalize, create the collision pipeline before solver construction, and pass the pipeline to VBD. Resources expose `contacts=solver.contacts` for inspection and `step_contacts=None` for solver-owned detection.

- [ ] **Step 1: Write capability and ordering tests.** `test_vbd_requires_external_implicit`, `test_vbd_pipeline_owns_contacts_and_detection`, and `test_vbd_is_colored_after_replication` assert the supported configuration and resource order. `test_vbd_armature_requires_revolute_in_actual_sparse_articulation` checks upper bounds/nonzero fixed values and imported unselected joints, not only initial values or a sparse-setting string. Reject sparse local fallback, unsupported joint types, sparse reduced-elastic combinations, unsupported tendon resets, collision settings that leave stale contacts after episode reset, and `integrate_with_external_rigid_solver=True` without a rigid-body coupler.
- [ ] **Step 2: Run red tests.** Run `uv run pytest tests/test_newton_sysid_capabilities.py tests/test_newton_solvers.py tests/test_newton_simulation_context.py -q`; missing VBD resources/capability checks must fail.
- [ ] **Step 3: Enable VBD with the shared implicit controller.** Use `block_sparse_joints` or `block_sparse_levels` where armature requires them. Keep local compliant ALM available only for compatible zero-armature models; document its static friction versus sparse regularized friction. Send appropriate model-change notifications. Reset VBD with `flags=0`, write measured/full-turn coordinates and velocities into both buffers, run FK, reset pipeline contact matching and actuator state, then let VBD's schedule generate fresh contacts. Read fork-published joint outputs directly without `eval_ik` or overwriting body poses. Normalize public collision-slot/frequency enums and Warp deterministic options; classify the fork's documented ignored VBD options explicitly, including the three `rigid_contact_stick_*_eps` fields, `rigid_tendon_relaxation`, and `rigid_enable_dahl_friction`.
- [ ] **Step 4: Run real VBD acceptance and document the example.** `test_vbd_sysid_changes_friction_damping_and_sparse_armature` proves each supported property affects motion. `test_vbd_reset_preserves_continuous_turns_and_cold_contacts` checks A/B/A and contact resets. `test_vbd_external_pd_cuda_matches_eager` covers full bias/actuator/solver graph execution and live candidates. A short SO-101 sparse VBD run uses explicit `controller.implementation: external_implicit`, effort limits, ignored velocity, finite scores, common outputs, and fresh-process fitted-model replay matching its logged MSE. Do not require identical identified parameters across MuJoCo and VBD. Test BDF2 settings only under the fork's documented restrictions, with early errors for incompatible models.
- [ ] **Step 5: Gate, commit, review, and prepare PR 5.** Run full acceptance for native/external MuJoCo, VBD, and toy backend, then commit as `feat: support VBD actuator system identification` and prepare its separate PR.

## Stage acceptance and finishing

For every product stage run from the repository root:

```bash
uv run ruff check
uv run ruff format --check
uv run pyright
uv run pytest <all-targeted-test-files-for-this-stage>
uv run pytest -o "addopts="
```

The final command includes integration tests because all stages affect shared simulation/task infrastructure or its dependencies. Run tests once after the final relevant edits; repeat only for new failures or changes. Report actual hardware skips, download/environment failures, and unrelated baseline failures without weakening assertions or claiming unexecuted checks passed.

The existing relevant integration gates are:

```bash
uv run pytest -m integration tests/integration/so101_actuator_sysid/test_cli.py
uv run pytest -m integration tests/integration/g1/test_training.py
```

New integration modules join the targeted gate when their stage lands. Independent review checks the design and tests against the exact diff before PR creation. Use a temporary body file with `gh pr create --draft --body-file ...`, describe only the completed stage and observed validation, then attach its URL with the app artifact tool. Preserve separate stage branches/PRs and wait for stage acceptance before beginning dependent stages.

## Planning evidence and self-review

- Dependency migration research verified `copy_count`, explicit collision pipelines, intrinsic coordinate targets, retained native APIs, and the actual fork dependency floors.
- Backend research verified the USD assumption in the base context, eager-import cycles, polymorphic Pydantic serialization, and a real installed-metadata fixture rather than mocked discovery.
- Solver/actuator research verified owned `contacts=None`, packed gain views after implicit binding, per-call bias scratch allocation, full coupled response, and fresh-process replay requirements.
- Coverage: Tasks 1–2 cover the pinned runtime; Tasks 3–4 cover backend/config/context/export/discovery; Tasks 5–6 cover settings/identity/provenance; Tasks 7–8 cover implicit PD/bias/clamp/reset/capture/export; Task 9 covers VBD capabilities, runtime, examples, and replay. No design section is assigned to an unimplemented future stage outside these five PRs.
- Self-review checked interface definitions before consumers, export/provenance types, legacy/raw option entry points, context identity including controller mode, and the ownership of captured callbacks. File arrangement and helper bodies remain implementation choices; user-visible configuration names, supported modes, physics sequence, error behavior, and acceptance boundaries are fixed by this plan and the approved design.
