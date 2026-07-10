"""Host-safe resolution and immutable execution for actuator training.

Resolution freezes profiles, mappings, SysID state, and every semantic
fingerprint before Torch or Newton is imported. The default backend crosses
that runtime boundary only for a fresh, non-dry run.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from anchor_bench.core.artifacts import RunManifest, run_slot
from anchor_bench.core.hashing import (
    canonical_object,
    canonical_sha256,
    contained_path,
    repo_relative_posix,
    require_relative_posix_path,
    sha256_file,
    sha256_tree,
    tree_fingerprint,
    validate_sha256,
)
from anchor_bench.core.manifest_schema import ProfileSnapshot
from anchor_bench.core.paths import phase_run_dir
from anchor_bench.core.plan import CommandPlan, command_identity_metadata
from anchor_bench.core.profiles import load_profile, load_yaml_mapping, require_logical_name
from anchor_bench.robots.fingerprints import fingerprint_contains_preview
from anchor_bench.sysid.state import (
    AcceptedSysIdState,
    CompleteSysIdState,
    SysIdCandidateState,
    canonicalize_parameter_maps,
    load_external_sysid_candidate,
    require_parameter_cells,
)
from anchor_bench.training.schema import (
    ResolvedTrainingMapping,
    TrainingFeature,
    TrainingFit,
    TrainingMapping,
    TrainingNetwork,
    TrainingProfile,
    TrainingTarget,
)

if TYPE_CHECKING:
    from anchor_bench.training.trainer import TrainingArtifacts


DatasetFingerprintResolver = Callable[[Path], Mapping[str, Any]]
RunDirectoryResolver = Callable[..., Path]
TrainingBackend = Callable[["ResolvedTrainingRun", Path], "TrainingArtifacts"]
JointSelector = Sequence[str] | Literal["all"]


@dataclass(frozen=True)
class TrainingInvocation:
    """Typed command inputs; supplied fields replace profile fields."""

    robot: str
    profile: str = "default"
    sysid_state: str | Path | None = None
    replicate: str | None = None
    dry_run: bool = True
    data: str | None = None
    sample_dt_s: float | None = None
    network: TrainingNetwork | None = None
    target: TrainingTarget | None = None
    mappings: Sequence[TrainingMapping] | None = None
    input_joints: JointSelector | None = None
    output_joints: JointSelector | None = None
    input_features: Sequence[TrainingFeature] | None = None
    fit: TrainingFit | None = None
    seed: int | None = None
    skip_optuna: bool | None = None
    trials: int | None = None
    trial_epochs: int | None = None
    final_epochs: int | None = None
    noise_std: float | None = None


@dataclass(frozen=True)
class TrainingResolverInputs:
    """Repository paths and semantic fingerprints supplied by integration."""

    repository_root: Path
    profiles_directory: Path
    accepted_sysid_state: Path
    canonical_joint_names: Sequence[str]
    joint_aliases: Mapping[str, str]
    robot_identity_fingerprint: Mapping[str, Any]
    asset_fingerprint: Mapping[str, Any]
    source_fingerprint: Mapping[str, Any]
    dependency_fingerprints: Mapping[str, Any]
    dataset_fingerprint: Mapping[str, Any] | None = None
    dataset_fingerprint_resolver: DatasetFingerprintResolver | None = None
    run_directory_resolver: RunDirectoryResolver = phase_run_dir


@dataclass(frozen=True)
class ResolvedTrainingRun:
    """Complete semantic training input and deterministic output location."""

    repository_root: Path
    robot: str
    profile_name: str
    profile_snapshot: ProfileSnapshot
    resolved_config: TrainingProfile
    cli_overrides: dict[str, Any]
    replicate: str | None
    sysid_state: CompleteSysIdState
    sysid_state_sha256: str
    mappings: tuple[ResolvedTrainingMapping, ...]
    fingerprints: dict[str, Any]
    upstream: dict[str, Any]
    run_input: dict[str, Any]
    run_input_sha256: str
    run_dir: Path
    dry_run: bool
    plan: CommandPlan


@dataclass(frozen=True)
class TrainingRunOutcome:
    """Result shared by dry runs, completed runs, and validated no-ops."""

    plan: CommandPlan
    run_dir: Path
    run_input_sha256: str
    dry_run: bool
    no_op: bool
    manifest: RunManifest | None = None


def _as_document(value: Any) -> Any:
    """Convert one typed override to its YAML-compatible representation."""

    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(key): _as_document(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_as_document(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _selection_document(value: JointSelector) -> str | list[str]:
    """Serialize a typed joint selector without changing its semantics."""

    if isinstance(value, str):
        if value != "all":
            raise ValueError("joint selectors supplied as strings must be 'all'")
        return value
    return [str(item) for item in value]


def _override_document(invocation: TrainingInvocation) -> dict[str, Any]:
    """Extract explicit typed replacements and validate shorthand conflicts."""

    if invocation.mappings is not None and (
        invocation.input_joints is not None or invocation.output_joints is not None
    ):
        raise ValueError("mappings cannot be combined with input_joints or output_joints")
    if invocation.output_joints is not None and invocation.input_joints is None:
        raise ValueError("input_joints is required when output_joints is supplied")

    result: dict[str, Any] = {}
    for name in ("data", "sample_dt_s", "network", "target", "mappings", "input_features", "fit"):
        value = getattr(invocation, name)
        if value is not None:
            result[name] = _as_document(value)
    if invocation.input_joints is not None:
        mapping: dict[str, Any] = {
            "input_joints": _selection_document(invocation.input_joints),
        }
        if invocation.output_joints is not None:
            mapping["output_joints"] = _selection_document(invocation.output_joints)
        result["mappings"] = [mapping]
    if invocation.sysid_state is not None:
        result["sysid_state"] = str(invocation.sysid_state)

    fit_fields = {
        name: getattr(invocation, name)
        for name in ("seed", "skip_optuna", "trials", "trial_epochs", "final_epochs", "noise_std")
        if getattr(invocation, name) is not None
    }
    if fit_fields:
        fit = dict(result.get("fit") or {})
        fit.update(fit_fields)
        result["fit"] = fit
    return result


def _canonical_joint_name(
    name: str,
    *,
    canonical: tuple[str, ...],
    aliases: Mapping[str, str],
) -> str:
    """Resolve one alias to a known canonical robot joint."""

    resolved = aliases.get(name, name)
    if resolved not in set(canonical):
        raise ValueError(f"unknown training joint: {name}")
    return resolved


def _canonical_joint_selection(
    selection: JointSelector,
    *,
    canonical: tuple[str, ...],
    aliases: Mapping[str, str],
    label: str,
) -> tuple[str, ...]:
    """Expand ``all``, resolve aliases, and reject physical duplicates."""

    if selection == "all":
        return canonical
    if isinstance(selection, str):
        raise ValueError(f"{label} must be an ordered list or 'all'")
    resolved = tuple(
        _canonical_joint_name(str(name), canonical=canonical, aliases=aliases)
        for name in selection
    )
    duplicates = sorted({name for index, name in enumerate(resolved) if name in resolved[:index]})
    if duplicates:
        raise ValueError(f"duplicate {label} after alias resolution: {', '.join(duplicates)}")
    return resolved


def _canonical_mappings(
    mappings: Sequence[TrainingMapping],
    *,
    canonical_joint_names: Sequence[str],
    aliases: Mapping[str, str],
) -> tuple[ResolvedTrainingMapping, ...]:
    """Resolve every physical mapping into canonical robot joint order."""

    canonical = tuple(canonical_joint_names)
    if not canonical or len(canonical) != len(set(canonical)):
        raise ValueError("canonical_joint_names must be non-empty and unique")
    unknown_targets = sorted(set(aliases.values()) - set(canonical))
    if unknown_targets:
        raise ValueError("joint aliases target unknown canonical joints: " + ", ".join(unknown_targets))

    resolved: list[ResolvedTrainingMapping] = []
    for mapping in mappings:
        inputs = _canonical_joint_selection(
            mapping.input_joints,
            canonical=canonical,
            aliases=aliases,
            label="input_joints",
        )
        outputs = _canonical_joint_selection(
            mapping.output_selection,
            canonical=canonical,
            aliases=aliases,
            label="output_joints",
        )
        resolved.append(ResolvedTrainingMapping(input_joints=inputs, output_joints=outputs))

    identities = [(mapping.input_joints, mapping.output_joints) for mapping in resolved]
    if len(identities) != len(set(identities)):
        raise ValueError("mappings contain duplicates after alias resolution")
    expected = (len(resolved[0].input_joints), len(resolved[0].output_joints))
    for index, mapping in enumerate(resolved[1:], start=1):
        width = (len(mapping.input_joints), len(mapping.output_joints))
        if width != expected:
            raise ValueError(
                f"mapping {index} has shape {width[0]}->{width[1]}; "
                f"expected {expected[0]}->{expected[1]}"
            )
    return tuple(resolved)


def _resolved_profile_document(
    loaded: TrainingProfile,
    overrides: Mapping[str, Any],
    mappings: Sequence[ResolvedTrainingMapping],
) -> dict[str, Any]:
    """Apply whole-field overrides and freeze canonical explicit mappings."""

    document = loaded.to_dict()
    document.update(dict(overrides))
    document["mappings"] = [mapping.to_dict() for mapping in mappings]
    return document


def _resolve_dataset_fingerprint(
    data: str,
    inputs: TrainingResolverInputs,
) -> tuple[str, dict[str, Any]]:
    """Normalize a dataset path and capture its exact tree fingerprint."""

    relative = require_relative_posix_path(data)
    dataset_path = contained_path(inputs.repository_root, relative, must_exist=False)
    if inputs.dataset_fingerprint is not None:
        fingerprint = dict(inputs.dataset_fingerprint)
    elif inputs.dataset_fingerprint_resolver is not None:
        fingerprint = dict(inputs.dataset_fingerprint_resolver(dataset_path))
    else:
        fingerprint = tree_fingerprint(dataset_path)
    canonical_object(fingerprint)
    digest = fingerprint.get("tree_sha256")
    if not isinstance(digest, str):
        raise ValueError("dataset fingerprint must contain tree_sha256")
    validate_sha256(digest, label="dataset tree_sha256")
    return relative, {"path": relative, **fingerprint}


def _canonical_sysid_state(
    state: CompleteSysIdState,
    canonical_joint_names: Sequence[str],
) -> CompleteSysIdState:
    """Reject alias cells and impose canonical robot order on SysID maps."""

    document = state.to_dict()
    document["parameters"] = canonicalize_parameter_maps(
        state.parameters,
        canonical_joint_names,
        require_all_joints=False,
    )
    if isinstance(state, AcceptedSysIdState):
        return AcceptedSysIdState.from_dict(document, "accepted SysID state")
    return SysIdCandidateState.from_dict(document, "candidate SysID state")


def _load_sysid_state(
    selection: str,
    invocation: TrainingInvocation,
    inputs: TrainingResolverInputs,
) -> tuple[CompleteSysIdState, dict[str, Any]]:
    """Load accepted SysID state or authenticate one completed candidate."""

    if selection == "default":
        state = AcceptedSysIdState.from_dict(
            load_yaml_mapping(inputs.accepted_sysid_state),
            inputs.accepted_sysid_state,
        )
        canonical = _canonical_sysid_state(state, inputs.canonical_joint_names)
        return canonical, {
            "sysid_state": {
                "kind": "accepted",
                "path": repo_relative_posix(
                    inputs.accepted_sysid_state,
                    repository_root=inputs.repository_root,
                ),
                "state_sha256": canonical_sha256(canonical.to_dict()),
            }
        }

    selected_path = Path(selection)
    if not selected_path.is_absolute():
        selected_path = inputs.repository_root / selected_path
    selected_path = contained_path(inputs.repository_root, selected_path, must_exist=True)
    if selected_path.name != "sysid_state.yaml" or not selected_path.is_file():
        raise ValueError(
            "sysid_state must be 'default' or sysid_state.yaml from a completed SysID run"
        )
    loaded = load_external_sysid_candidate(selected_path.parent, expected_robot=invocation.robot)
    canonical = _canonical_sysid_state(loaded.state, inputs.canonical_joint_names)
    return canonical, {
        "sysid_state": {
            "kind": "candidate",
            "run": repo_relative_posix(
                loaded.run_dir,
                repository_root=inputs.repository_root,
            ),
            "run_input_sha256": loaded.manifest.run_input_sha256,
            "run_manifest_sha256": loaded.manifest.run_manifest_sha256,
            "state_sha256": canonical_sha256(canonical.to_dict()),
        }
    }


def _required_sysid_cells(
    config: TrainingProfile,
    mappings: Sequence[ResolvedTrainingMapping],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return exact baseline parameter axes consumed by training."""

    needs_pd = config.target == "torque_residual" or any(
        feature.name == "solver_pd" for feature in config.input_features
    )
    requirements: list[tuple[str, tuple[str, ...]]] = []
    if needs_pd:
        input_joints = tuple(
            dict.fromkeys(joint for mapping in mappings for joint in mapping.input_joints)
        )
        requirements.extend((name, input_joints) for name in ("stiffness", "damping"))
    if config.target == "torque_residual":
        output_joints = tuple(
            dict.fromkeys(joint for mapping in mappings for joint in mapping.output_joints)
        )
        requirements.extend(
            (name, output_joints) for name in ("dry_friction", "viscous_friction")
        )
    return tuple(requirements)


def _validate_sysid_requirements(
    state: CompleteSysIdState,
    config: TrainingProfile,
    mappings: Sequence[ResolvedTrainingMapping],
) -> None:
    """Require every baseline cell needed by the resolved target/features."""

    for parameter, joints in _required_sysid_cells(config, mappings):
        require_parameter_cells(state, (parameter,), joints)


def _fingerprints(
    dataset: Mapping[str, Any],
    inputs: TrainingResolverInputs,
) -> dict[str, Any]:
    """Validate and freeze the semantic fingerprint envelope."""

    result = {
        "robot": dict(inputs.robot_identity_fingerprint),
        "dataset": dict(dataset),
        "asset": dict(inputs.asset_fingerprint),
        "source": dict(inputs.source_fingerprint),
        "dependencies": dict(inputs.dependency_fingerprints),
    }
    canonical_object(result)
    if not fingerprint_contains_preview(result["asset"]):
        _fingerprint_digest(result["asset"], label="asset")
    return result


def _command_plan(
    invocation: TrainingInvocation,
    config: TrainingProfile,
    run_dir: Path,
    run_input_sha256: str,
    *,
    preview_identity: bool,
) -> CommandPlan:
    """Build the host-safe summary carried by a resolved training run."""

    command = [
        "anchor-bench",
        "train-actuator",
        "--robot",
        invocation.robot,
        "--profile",
        invocation.profile,
    ]
    if invocation.sysid_state is not None:
        command.extend(["--sysid-state", str(invocation.sysid_state)])
    if invocation.replicate is not None:
        command.extend(["--replicate", invocation.replicate])
    if invocation.dry_run:
        command.append("--dry-run")
    return CommandPlan(
        workflow="train-actuator",
        robot=invocation.robot,
        command=command,
        dry_run=invocation.dry_run,
        notes=[
            "profile is a complete recipe; explicitly supplied fields replace it",
            "episode splitting precedes mapping expansion",
            "execution writes one immutable schema-v3 network candidate",
        ],
        metadata={
            "profile": invocation.profile,
            "seed": config.fit.seed,
            "replicate": invocation.replicate,
            **command_identity_metadata(
                run_dir,
                run_input_sha256,
                preview=preview_identity,
            ),
            "resolved_config": config.to_dict(),
        },
    )


def resolve_training_run(
    invocation: TrainingInvocation,
    inputs: TrainingResolverInputs,
) -> ResolvedTrainingRun:
    """Resolve and hash training without importing Torch, Newton, or robot assets."""

    robot = require_logical_name(invocation.robot, label="robot name")
    profile_name = require_logical_name(invocation.profile, label="profile name")
    if invocation.replicate is not None and (
        not invocation.replicate
        or len(invocation.replicate) > 128
        or any(ord(character) < 32 for character in invocation.replicate)
    ):
        raise ValueError(
            "replicate must be a non-empty printable label of at most 128 characters"
        )

    loaded = load_profile(inputs.profiles_directory, profile_name, TrainingProfile)
    cli_overrides = _override_document(invocation)
    preliminary_document = loaded.model.to_dict()
    preliminary_document.update(cli_overrides)
    preliminary = TrainingProfile.from_dict(
        preliminary_document,
        "resolved training config",
    )
    mappings = _canonical_mappings(
        preliminary.mappings,
        canonical_joint_names=inputs.canonical_joint_names,
        aliases=inputs.joint_aliases,
    )
    resolved_document = _resolved_profile_document(
        loaded.model,
        cli_overrides,
        mappings,
    )
    data, dataset = _resolve_dataset_fingerprint(str(resolved_document["data"]), inputs)
    resolved_document["data"] = data
    resolved_config = TrainingProfile.from_dict(
        resolved_document,
        "resolved training config",
    )

    sysid_state, upstream = _load_sysid_state(
        resolved_config.sysid_state,
        invocation,
        inputs,
    )
    normalized_sysid_selection = "default"
    if upstream["sysid_state"]["kind"] == "candidate":
        normalized_sysid_selection = (
            f'{upstream["sysid_state"]["run"]}/sysid_state.yaml'
        )
    if resolved_config.sysid_state != normalized_sysid_selection:
        normalized_document = resolved_config.to_dict()
        normalized_document["sysid_state"] = normalized_sysid_selection
        resolved_config = TrainingProfile.from_dict(
            normalized_document,
            "resolved training config",
        )
    _validate_sysid_requirements(sysid_state, resolved_config, mappings)
    sysid_state_sha256 = canonical_sha256(sysid_state.to_dict())
    fingerprints = _fingerprints(dataset, inputs)
    profile_snapshot = ProfileSnapshot.from_dict(
        loaded.manifest_data(repository_root=inputs.repository_root),
        "profile snapshot",
    )
    run_input = {
        "schema_version": 1,
        "phase": "train",
        "robot": robot,
        "profile": {
            "name": profile_name,
            "content": loaded.content,
            "content_sha256": loaded.content_sha256,
        },
        "resolved_config": resolved_config.to_dict(),
        "seed": resolved_config.fit.seed,
        "replicate": invocation.replicate,
        "sysid_state": {
            "content": sysid_state.to_dict(),
            "content_sha256": sysid_state_sha256,
        },
        "fingerprints": fingerprints,
    }
    run_input_sha256 = canonical_sha256(run_input)
    run_dir = inputs.run_directory_resolver(
        "train",
        robot=robot,
        profile=profile_name,
        seed=resolved_config.fit.seed,
        run_input_sha256=run_input_sha256,
    )
    plan = _command_plan(
        invocation,
        resolved_config,
        run_dir,
        run_input_sha256,
        preview_identity=fingerprint_contains_preview(fingerprints),
    )
    return ResolvedTrainingRun(
        repository_root=Path(inputs.repository_root),
        robot=robot,
        profile_name=profile_name,
        profile_snapshot=profile_snapshot,
        resolved_config=resolved_config,
        cli_overrides=cli_overrides,
        replicate=invocation.replicate,
        sysid_state=sysid_state,
        sysid_state_sha256=sysid_state_sha256,
        mappings=mappings,
        fingerprints=fingerprints,
        upstream=upstream,
        run_input=run_input,
        run_input_sha256=run_input_sha256,
        run_dir=run_dir,
        dry_run=invocation.dry_run,
        plan=plan,
    )


def _fingerprint_digest(fingerprint: Mapping[str, Any], *, label: str) -> str:
    """Read one conventional raw or tree hash from a fingerprint."""

    for key in ("sha256", f"{label}_sha256", "tree_sha256"):
        value = fingerprint.get(key)
        if isinstance(value, str):
            return validate_sha256(value, label=f"{label} fingerprint {key}")
    raise ValueError(
        f"{label} fingerprint must contain sha256, {label}_sha256, or tree_sha256"
    )


def _stat_value(values: Sequence[float]) -> float | tuple[float, ...]:
    """Encode a width-one vector as a scalar for runtime metadata."""

    result = tuple(float(value) for value in values)
    return result[0] if len(result) == 1 else result


def _parameter_values(
    state: CompleteSysIdState,
    parameter: str,
    joints: Sequence[str],
) -> float | tuple[float, ...]:
    """Read ordered mapping-specific values from the frozen SysID state."""

    require_parameter_cells(state, (parameter,), joints)
    return _stat_value([state.parameters[parameter][joint] for joint in joints])


def _optional_parameter_values(
    state: CompleteSysIdState,
    parameter: str,
    joints: Sequence[str],
) -> float | tuple[float, ...] | None:
    """Return an optional complete vector and reject a partial vector."""

    parameter_map = state.parameters.get(parameter)
    if parameter_map is None:
        return None
    present = [joint in parameter_map for joint in joints]
    if any(present) and not all(present):
        missing = [
            joint
            for joint, exists in zip(joints, present, strict=True)
            if not exists
        ]
        cells = ", ".join(f"{parameter}/{joint}" for joint in missing)
        raise ValueError(f"missing required parameter cells: {cells}")
    if not any(present):
        return None
    return _stat_value([parameter_map[joint] for joint in joints])


def _runtime_mappings(
    resolved: ResolvedTrainingRun,
    effort_limits: Callable[[Sequence[str]], Sequence[float]],
) -> tuple[Any, ...]:
    """Build mapping-specific physics, limits, and clamping metadata."""

    from anchor_bench.training.metadata import (
        ArmatureValues,
        FrictionBaseline,
        PDBaseline,
        RuntimeMapping,
    )

    config = resolved.resolved_config
    needs_pd = config.target == "torque_residual" or any(
        feature.name == "solver_pd" for feature in config.input_features
    )
    result: list[RuntimeMapping] = []
    for mapping in resolved.mappings:
        pd = None
        if needs_pd:
            pd = PDBaseline(
                joint_names=mapping.input_joints,
                kp=_parameter_values(
                    resolved.sysid_state,
                    "stiffness",
                    mapping.input_joints,
                ),
                kd=_parameter_values(
                    resolved.sysid_state,
                    "damping",
                    mapping.input_joints,
                ),
            )
        friction = None
        if config.target == "torque_residual":
            friction = FrictionBaseline(
                joint_names=mapping.output_joints,
                dry=_parameter_values(
                    resolved.sysid_state,
                    "dry_friction",
                    mapping.output_joints,
                ),
                viscous=_parameter_values(
                    resolved.sysid_state,
                    "viscous_friction",
                    mapping.output_joints,
                ),
            )
        armature_values = _optional_parameter_values(
            resolved.sysid_state,
            "armature",
            mapping.output_joints,
        )
        armature = (
            None
            if armature_values is None
            else ArmatureValues(
                joint_names=mapping.output_joints,
                values=armature_values,
            )
        )
        result.append(
            RuntimeMapping(
                input_joints=mapping.input_joints,
                output_joints=mapping.output_joints,
                pd_baseline=pd,
                friction_baseline=friction,
                armature=armature,
                effort_limits=_stat_value(effort_limits(mapping.output_joints)),
                clamping=(
                    "residual_output"
                    if config.target == "torque_residual"
                    else "total_output"
                ),
            )
        )
    return tuple(result)


def _assemble_episodes(
    resolved: ResolvedTrainingRun,
    assembler: Any,
) -> tuple[list[Any], list[Any]]:
    """Split recorded episodes first, then expand each split by mapping."""

    from anchor_bench.training.data import (
        discover_training_sources,
        split_training_sources,
    )

    dataset_path = contained_path(
        resolved.repository_root,
        resolved.resolved_config.data,
        must_exist=True,
    )
    expected_dataset_sha256 = _fingerprint_digest(
        resolved.fingerprints["dataset"],
        label="dataset",
    )
    actual_dataset_sha256 = sha256_tree(dataset_path)
    if actual_dataset_sha256 != expected_dataset_sha256:
        raise RuntimeError(
            "training dataset changed after resolution: "
            f"expected {expected_dataset_sha256}, got {actual_dataset_sha256}"
        )
    sources = discover_training_sources(dataset_path)
    train_sources, validation_sources = split_training_sources(
        sources,
        seed=resolved.resolved_config.fit.seed,
    )

    def expand(split_sources: Sequence[Any]) -> list[Any]:
        """Expand one already-split source list into independent mappings."""

        episodes: list[Any] = []
        for source in split_sources:
            prepared = assembler.load(
                source.path,
                resolved.resolved_config.sample_dt_s,
            )
            cache_key = repo_relative_posix(
                source.path,
                repository_root=resolved.repository_root,
            )
            for mapping in resolved.mappings:
                episodes.append(
                    assembler.assemble(
                        prepared,
                        mapping=mapping,
                        features=resolved.resolved_config.input_features,
                        target=resolved.resolved_config.target,
                        dynamic_bias_cache_key=cache_key,
                    )
                )
        return episodes

    return expand(train_sources), expand(validation_sources)


def _dynamic_bias_provenance(episodes: Sequence[Any]) -> Mapping[str, Any] | None:
    """Pool distinct per-source/mapping provider records deterministically."""

    records: dict[str, dict[str, Any]] = {}
    for episode in episodes:
        provenance = episode.dynamic_bias_provenance
        if provenance is None:
            continue
        document = dict(provenance)
        records.setdefault(canonical_sha256(document), document)
    if not records:
        return None
    return {"evaluations": [records[key] for key in sorted(records)]}


def train_resolved_run(
    resolved: ResolvedTrainingRun,
    staging_dir: Path,
) -> "TrainingArtifacts":
    """Run the real lazy Newton/Torch backend for one fresh staging directory."""

    from anchor_bench.robots.fingerprints import (
        verify_robot_identity_fingerprint,
        verify_runtime_asset_fingerprint,
    )
    from anchor_bench.robots.registry import get_robot_from_root
    from anchor_bench.training.dynamic_bias import NewtonMuJoCoDynamicBiasProvider
    from anchor_bench.training.episode import EpisodeAssembler
    from anchor_bench.training.metadata import RuntimeHashes
    from anchor_bench.training.trainer import TrainingJob, train_stateful_gru

    bundle = get_robot_from_root(resolved.robot, resolved.repository_root)
    expected_robot = resolved.fingerprints.get("robot")
    if not isinstance(expected_robot, Mapping):
        raise ValueError("resolved training robot fingerprint must be a mapping")
    verify_robot_identity_fingerprint(
        expected_robot,
        bundle,
        repository_root=resolved.repository_root,
    )
    provider = NewtonMuJoCoDynamicBiasProvider(bundle)
    snapshot = provider.asset_snapshot
    expected_asset = resolved.fingerprints.get("asset")
    if not isinstance(expected_asset, Mapping):
        raise ValueError("resolved training asset fingerprint must be a mapping")
    verify_runtime_asset_fingerprint(
        expected_asset,
        snapshot.path,
        runtime_usd=bundle.spec.runtime_usd,
    )
    expected_asset_sha256 = _fingerprint_digest(
        resolved.fingerprints["asset"],
        label="asset",
    )
    if snapshot.sha256 != expected_asset_sha256:
        raise RuntimeError(
            "training asset changed after resolution: "
            f"expected {expected_asset_sha256}, got {snapshot.sha256}"
        )
    needs_bias = any(
        feature.name == "dynamic_bias"
        for feature in resolved.resolved_config.input_features
    )
    assembler = EpisodeAssembler(provider if needs_bias else None)
    train_episodes, validation_episodes = _assemble_episodes(resolved, assembler)
    runtime_mappings = _runtime_mappings(resolved, snapshot.effort_limits)
    provenance = _dynamic_bias_provenance(
        [*train_episodes, *validation_episodes]
    )
    hashes = RuntimeHashes(
        profile_sha256=resolved.profile_snapshot.content_sha256,
        resolved_config_sha256=canonical_sha256(
            resolved.resolved_config.to_dict()
        ),
        sysid_state_sha256=sha256_file(staging_dir / "sysid_state.yaml"),
        asset_sha256=snapshot.sha256,
        source_sha256=_fingerprint_digest(
            resolved.fingerprints["dataset"],
            label="dataset",
        ),
    )
    return train_stateful_gru(
        TrainingJob(
            train_episodes=train_episodes,
            val_episodes=validation_episodes,
            output_dir=staging_dir,
            target=resolved.resolved_config.target,
            mappings=resolved.mappings,
            mapping_runtime=runtime_mappings,
            input_features=resolved.resolved_config.input_features,
            network=resolved.resolved_config.network,
            fit=resolved.resolved_config.fit,
            sample_dt_s=resolved.resolved_config.sample_dt_s,
            hashes=hashes,
            dynamic_bias_provenance=provenance,
            stats_metadata={
                "run_input_sha256": resolved.run_input_sha256,
                "replicate": resolved.replicate,
            },
        )
    )


def _validate_backend_artifacts(staging_dir: Path, artifacts: Any) -> None:
    """Require the backend result to name every fixed training artifact."""

    expected = {
        "checkpoint_path": staging_dir / "checkpoints" / "best.pt",
        "stats_path": staging_dir / "stats.json",
        "network_path": staging_dir / "network.pt",
    }
    for attribute, path in expected.items():
        actual = getattr(artifacts, attribute, None)
        if actual is None or Path(actual).resolve() != path.resolve():
            raise ValueError(f"training backend {attribute} must be {path}")
        if path.is_symlink() or not path.is_file():
            raise ValueError(
                f"training backend did not produce regular file {path}"
            )


def execute_training_run(
    resolved: ResolvedTrainingRun,
    backend: TrainingBackend | None = None,
) -> TrainingRunOutcome:
    """Execute once through the immutable protocol, or return a no-op."""

    if resolved.dry_run:
        return TrainingRunOutcome(
            resolved.plan,
            resolved.run_dir,
            resolved.run_input_sha256,
            True,
            False,
        )

    selected_backend = backend or train_resolved_run
    completed_manifest: RunManifest | None = None
    no_op = False
    with run_slot(
        resolved.run_dir,
        resolved.run_input_sha256,
        containment_root=resolved.repository_root,
    ) as slot:
        no_op = slot.noop
        if slot.noop:
            completed_manifest = slot.manifest
        else:
            if slot.tmp_dir is None:  # pragma: no cover
                raise RuntimeError("fresh training run has no staging directory")
            from anchor_bench.training.artifacts import (
                write_training_inputs,
                write_training_manifest,
            )

            write_training_inputs(slot.tmp_dir, resolved)
            artifacts = selected_backend(resolved, slot.tmp_dir)
            _validate_backend_artifacts(slot.tmp_dir, artifacts)
            completed_manifest = write_training_manifest(slot.tmp_dir, resolved)
    return TrainingRunOutcome(
        resolved.plan,
        resolved.run_dir,
        resolved.run_input_sha256,
        False,
        no_op,
        completed_manifest,
    )


__all__ = [
    "ResolvedTrainingRun",
    "TrainingBackend",
    "TrainingInvocation",
    "TrainingResolverInputs",
    "TrainingRunOutcome",
    "execute_training_run",
    "resolve_training_run",
    "train_resolved_run",
]
