"""Host-safe resolution and immutable execution envelope for SysID.

Resolution performs all semantic work before a simulator is imported. A
separately injected backend performs optimization only for a non-dry,
previously unseen run.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anchor_bench.core.artifacts import RunManifest, run_slot
from anchor_bench.core.hashing import (
    canonical_object,
    canonical_sha256,
    repo_relative_posix,
    require_relative_posix_path,
    tree_fingerprint,
    validate_sha256,
)
from anchor_bench.core.manifest_schema import ProfileSnapshot
from anchor_bench.core.paths import phase_run_dir
from anchor_bench.core.plan import CommandPlan, command_identity_metadata
from anchor_bench.core.profiles import load_profile, load_yaml_mapping, require_logical_name
from anchor_bench.robots.fingerprints import fingerprint_contains_preview
from anchor_bench.sysid.schema import (
    SUPPORTED_SYSID_PARAMETERS,
    SysIdLossConfig,
    SysIdOptimizerConfig,
    SysIdParameterSpec,
    SysIdProfile,
    SysIdSimulationConfig,
)
from anchor_bench.sysid.state import (
    AcceptedSysIdState,
    SysIdParentState,
    canonicalize_parameter_maps,
    load_external_sysid_candidate,
    require_parameter_cells,
)

DatasetFingerprintResolver = Callable[[Path], Mapping[str, Any]]
RunDirectoryResolver = Callable[..., Path]
ExecutionBackend = Callable[["ResolvedSysIdRun"], "SysIdExecutionResult"]


@dataclass(frozen=True)
class SysIdInvocation:
    """Typed command inputs; non-None profile fields replace whole fields."""

    robot: str
    profile: str = "default"
    initial_state: Path | None = None
    replicate: str | None = None
    dry_run: bool = True
    data: str | None = None
    joints: Sequence[str] | None = None
    simulation: SysIdSimulationConfig | None = None
    num_envs: int | None = None
    buffer_time: float | None = None
    seed: int | None = None
    optimizer: SysIdOptimizerConfig | None = None
    parameters: Mapping[str, SysIdParameterSpec] | None = None
    loss: SysIdLossConfig | None = None


@dataclass(frozen=True)
class SysIdResolverInputs:
    """Repository and fingerprint inputs supplied by the integration layer."""

    repository_root: Path
    profiles_directory: Path
    accepted_state: Path
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
class SysIdExecutionResult:
    """Serializable result returned by an injected optimization backend."""

    fitted_parameters: Mapping[str, Mapping[str, float]]
    metrics: Mapping[str, Any]
    optimizer_artifacts: Mapping[str, str | bytes]

    def validated_metrics(self) -> dict[str, Any]:
        """Return canonical metrics or fail before installation."""

        if not isinstance(self.metrics, Mapping):
            raise TypeError("SysID metrics must be a mapping")
        result = dict(self.metrics)
        canonical_object(result)
        return result


@dataclass(frozen=True)
class ResolvedSysIdRun:
    """Complete semantic SysID input and deterministic output location."""

    repository_root: Path
    robot: str
    profile_name: str
    profile_snapshot: ProfileSnapshot
    resolved_config: SysIdProfile
    cli_overrides: dict[str, Any]
    replicate: str | None
    parent_state: SysIdParentState
    parent_state_sha256: str
    canonical_joint_names: tuple[str, ...]
    joint_aliases: dict[str, str]
    fingerprints: dict[str, Any]
    upstream: dict[str, Any]
    run_input: dict[str, Any]
    run_input_sha256: str
    run_dir: Path
    dry_run: bool
    plan: CommandPlan

    def validate_fitted_parameters(
        self, fitted: Mapping[str, Mapping[str, float]]
    ) -> dict[str, dict[str, float]]:
        """Validate and canonically order the backend's exact sparse patch."""

        if not isinstance(fitted, Mapping):
            raise TypeError("fitted_parameters must be a mapping")
        configured = set(self.resolved_config.parameters)
        unexpected = sorted(set(fitted) - configured)
        missing_parameters = sorted(configured - set(fitted))
        if unexpected:
            raise ValueError(
                "fitted_parameters contains unconfigured parameters: "
                + ", ".join(unexpected)
            )
        if missing_parameters:
            raise ValueError(
                "fitted_parameters is missing configured parameters: "
                + ", ".join(missing_parameters)
            )

        ordered: dict[str, dict[str, float]] = {}
        expected_joints = tuple(self.resolved_config.joints)
        for parameter in SUPPORTED_SYSID_PARAMETERS:
            spec = self.resolved_config.parameters.get(parameter)
            if spec is None:
                continue
            raw_joint_map = fitted[parameter]
            if not isinstance(raw_joint_map, Mapping):
                raise TypeError(f"fitted parameter {parameter!r} must be a joint mapping")
            missing = [joint for joint in expected_joints if joint not in raw_joint_map]
            extras = sorted(set(raw_joint_map) - set(expected_joints))
            if missing or extras:
                details: list[str] = []
                if missing:
                    details.append("missing " + ", ".join(missing))
                if extras:
                    details.append("unexpected " + ", ".join(extras))
                raise ValueError(
                    f"fitted parameter {parameter!r} has wrong joint cells: "
                    + "; ".join(details)
                )
            values: list[float] = []
            ordered[parameter] = {}
            for joint in expected_joints:
                try:
                    value = float(raw_joint_map[joint])
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"fitted cell {parameter}/{joint} must be numeric"
                    ) from error
                if not math.isfinite(value):
                    raise ValueError(f"fitted cell {parameter}/{joint} must be finite")
                if value < spec.bounds.lower or value > spec.bounds.upper:
                    raise ValueError(
                        f"fitted cell {parameter}/{joint} ({value}) is outside bounds "
                        f"[{spec.bounds.lower}, {spec.bounds.upper}]"
                    )
                ordered[parameter][joint] = value
                values.append(value)
            if spec.mode == "shared" and any(
                value != values[0] for value in values[1:]
            ):
                raise ValueError(
                    f"shared fitted parameter {parameter!r} must have one value"
                )
        return ordered


@dataclass(frozen=True)
class SysIdRunOutcome:
    """Result shared by dry runs, completed runs, and validated no-ops."""

    plan: CommandPlan
    run_dir: Path
    run_input_sha256: str
    dry_run: bool
    no_op: bool
    manifest: RunManifest | None = None


def _override_document(invocation: SysIdInvocation) -> dict[str, Any]:
    """Extract only explicitly supplied typed replacements."""

    result: dict[str, Any] = {}
    for name in (
        "data", "joints", "simulation", "num_envs", "buffer_time",
        "seed", "optimizer", "parameters", "loss",
    ):
        value = getattr(invocation, name)
        if value is None:
            continue
        if hasattr(value, "to_dict"):
            result[name] = value.to_dict()
        elif isinstance(value, Mapping):
            result[name] = {
                key: item.to_dict() if hasattr(item, "to_dict") else item
                for key, item in value.items()
            }
        elif isinstance(value, Sequence) and not isinstance(value, str):
            result[name] = list(value)
        else:
            result[name] = value
    return result


def _canonical_joint_selection(
    joints: Sequence[str],
    canonical_joint_names: Sequence[str],
    aliases: Mapping[str, str],
) -> list[str]:
    """Resolve aliases once and reject duplicate physical joints."""

    canonical = tuple(canonical_joint_names)
    if not canonical:
        raise ValueError("canonical_joint_names must contain at least one joint")
    if len(canonical) != len(set(canonical)):
        raise ValueError("canonical_joint_names contains duplicates")
    canonical_set = set(canonical)
    bad_targets = sorted(
        {target for target in aliases.values() if target not in canonical_set}
    )
    if bad_targets:
        raise ValueError(
            "joint aliases target unknown canonical joints: " + ", ".join(bad_targets)
        )
    resolved: list[str] = []
    for name in joints:
        joint = aliases.get(name, name)
        if joint not in canonical_set:
            raise ValueError(f"unknown SysID joint: {name}")
        resolved.append(joint)
    duplicates = sorted({joint for joint in resolved if resolved.count(joint) > 1})
    if duplicates:
        raise ValueError(
            "duplicate SysID joints after alias resolution: " + ", ".join(duplicates)
        )
    return resolved


def _resolve_dataset_fingerprint(
    data: str, inputs: SysIdResolverInputs
) -> tuple[str, dict[str, Any]]:
    """Normalize the dataset path and obtain its exact tree fingerprint."""

    relative = require_relative_posix_path(data)
    dataset_path = inputs.repository_root / relative
    if inputs.dataset_fingerprint is not None:
        fingerprint = dict(inputs.dataset_fingerprint)
    elif inputs.dataset_fingerprint_resolver is not None:
        fingerprint = dict(inputs.dataset_fingerprint_resolver(dataset_path))
    else:
        fingerprint = tree_fingerprint(dataset_path)
    canonical_object(fingerprint)
    if "tree_sha256" not in fingerprint:
        raise ValueError("dataset fingerprint must contain tree_sha256")
    validate_sha256(fingerprint["tree_sha256"], label="dataset tree_sha256")
    return relative, {"path": relative, **fingerprint}


def _canonical_parent(
    state: SysIdParentState, canonical_joint_names: Sequence[str]
) -> SysIdParentState:
    """Reject aliases in state files and impose canonical robot ordering."""

    document = state.to_dict()
    document["parameters"] = canonicalize_parameter_maps(
        state.parameters, canonical_joint_names, require_all_joints=False
    )
    if isinstance(state, AcceptedSysIdState):
        return AcceptedSysIdState.from_dict(document, "accepted SysID state")
    return type(state).from_dict(document, "candidate SysID state")


def _load_parent_state(
    invocation: SysIdInvocation, inputs: SysIdResolverInputs
) -> tuple[SysIdParentState, dict[str, Any]]:
    """Load the accepted default or authenticate one prior candidate."""

    if invocation.initial_state is None:
        state = AcceptedSysIdState.from_dict(
            load_yaml_mapping(inputs.accepted_state), inputs.accepted_state
        )
        canonical = _canonical_parent(state, inputs.canonical_joint_names)
        return canonical, {
            "initial_state": {
                "kind": "accepted",
                "path": repo_relative_posix(
                    inputs.accepted_state, repository_root=inputs.repository_root
                ),
                "state_sha256": canonical_sha256(canonical.to_dict()),
            }
        }

    state_path = Path(invocation.initial_state)
    if state_path.name != "sysid_state.yaml" or not state_path.is_file():
        raise ValueError(
            "initial_state must select sysid_state.yaml from a completed SysID run"
        )
    loaded = load_external_sysid_candidate(
        state_path.parent, expected_robot=invocation.robot
    )
    canonical = _canonical_parent(loaded.state, inputs.canonical_joint_names)
    return canonical, {
        "initial_state": {
            "kind": "candidate",
            "run": repo_relative_posix(
                loaded.run_dir, repository_root=inputs.repository_root
            ),
            "run_input_sha256": loaded.manifest.run_input_sha256,
            "run_manifest_sha256": loaded.manifest.run_manifest_sha256,
            "state_sha256": canonical_sha256(canonical.to_dict()),
        }
    }


def _command_plan(
    invocation: SysIdInvocation,
    config: SysIdProfile,
    run_dir: Path,
    run_input_sha256: str,
    *,
    preview_identity: bool,
) -> CommandPlan:
    """Build the lightweight summary carried by a resolved run."""

    command = [
        "anchor-bench", "sysid", "--robot", invocation.robot,
        "--profile", invocation.profile,
    ]
    if invocation.initial_state is not None:
        command.extend(["--initial-state", str(invocation.initial_state)])
    if invocation.replicate is not None:
        command.extend(["--replicate", invocation.replicate])
    if invocation.dry_run:
        command.append("--dry-run")
    return CommandPlan(
        workflow="sysid",
        robot=invocation.robot,
        command=command,
        dry_run=invocation.dry_run,
        notes=[
            "profile is a complete recipe; explicitly supplied fields replace it",
            "execution writes an immutable manifest-authenticated candidate state",
        ],
        metadata={
            "profile": invocation.profile,
            "seed": config.seed,
            "replicate": invocation.replicate,
            **command_identity_metadata(
                run_dir,
                run_input_sha256,
                preview=preview_identity,
            ),
            "resolved_config": config.to_dict(),
        },
    )


def resolve_sysid_run(
    invocation: SysIdInvocation, inputs: SysIdResolverInputs
) -> ResolvedSysIdRun:
    """Resolve and hash a request without importing or running a simulator."""

    robot = require_logical_name(invocation.robot, label="robot name")
    if invocation.replicate is not None and (
        not invocation.replicate
        or len(invocation.replicate) > 128
        or any(ord(character) < 32 for character in invocation.replicate)
    ):
        raise ValueError(
            "replicate must be a non-empty printable label of at most 128 characters"
        )

    loaded_profile = load_profile(
        inputs.profiles_directory, invocation.profile, SysIdProfile
    )
    cli_overrides = _override_document(invocation)
    resolved_document = loaded_profile.model.to_dict()
    resolved_document.update(cli_overrides)
    resolved_document["joints"] = _canonical_joint_selection(
        resolved_document["joints"],
        inputs.canonical_joint_names,
        inputs.joint_aliases,
    )
    data, dataset = _resolve_dataset_fingerprint(
        resolved_document["data"], inputs
    )
    resolved_document["data"] = data
    resolved_config = SysIdProfile.from_dict(
        resolved_document, "resolved SysID config"
    )

    parent_state, upstream = _load_parent_state(invocation, inputs)
    require_parameter_cells(
        parent_state,
        tuple(resolved_config.parameters),
        resolved_config.joints,
    )
    parent_state_sha256 = canonical_sha256(parent_state.to_dict())
    fingerprints = {
        "robot": dict(inputs.robot_identity_fingerprint),
        "dataset": dataset,
        "asset": dict(inputs.asset_fingerprint),
        "source": dict(inputs.source_fingerprint),
        "dependencies": dict(inputs.dependency_fingerprints),
    }
    canonical_object(fingerprints)
    profile_snapshot = ProfileSnapshot.from_dict(
        loaded_profile.manifest_data(repository_root=inputs.repository_root),
        "profile snapshot",
    )
    run_input = {
        "schema_version": 1,
        "phase": "sysid",
        "robot": robot,
        "profile": {
            "name": invocation.profile,
            "content": loaded_profile.content,
            "content_sha256": loaded_profile.content_sha256,
        },
        "resolved_config": resolved_config.to_dict(),
        "seed": resolved_config.seed,
        "replicate": invocation.replicate,
        "parent_state": {
            "content": parent_state.to_dict(),
            "content_sha256": parent_state_sha256,
        },
        "fingerprints": fingerprints,
    }
    run_input_sha256 = canonical_sha256(run_input)
    run_dir = inputs.run_directory_resolver(
        "sysid",
        robot=robot,
        profile=invocation.profile,
        seed=resolved_config.seed,
        run_input_sha256=run_input_sha256,
    )
    plan = _command_plan(
        invocation,
        resolved_config,
        run_dir,
        run_input_sha256,
        preview_identity=fingerprint_contains_preview(fingerprints),
    )
    return ResolvedSysIdRun(
        repository_root=Path(inputs.repository_root),
        robot=robot,
        profile_name=invocation.profile,
        profile_snapshot=profile_snapshot,
        resolved_config=resolved_config,
        cli_overrides=cli_overrides,
        replicate=invocation.replicate,
        parent_state=parent_state,
        parent_state_sha256=parent_state_sha256,
        canonical_joint_names=tuple(inputs.canonical_joint_names),
        joint_aliases=dict(inputs.joint_aliases),
        fingerprints=fingerprints,
        upstream=upstream,
        run_input=run_input,
        run_input_sha256=run_input_sha256,
        run_dir=run_dir,
        dry_run=invocation.dry_run,
        plan=plan,
    )


def execute_sysid_run(
    resolved: ResolvedSysIdRun,
    backend: ExecutionBackend | None = None,
) -> SysIdRunOutcome:
    """Execute once through the immutable protocol, or return a no-op."""

    if resolved.dry_run:
        return SysIdRunOutcome(
            resolved.plan,
            resolved.run_dir,
            resolved.run_input_sha256,
            True,
            False,
        )
    if backend is None:
        raise ValueError("a SysID execution backend is required for a non-dry run")

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
                raise RuntimeError("fresh SysID run has no staging directory")
            result = backend(resolved)
            if not isinstance(result, SysIdExecutionResult):
                raise TypeError("SysID backend must return SysIdExecutionResult")
            from anchor_bench.sysid.artifacts import write_sysid_run

            completed_manifest = write_sysid_run(
                slot.tmp_dir, resolved, result
            )
    return SysIdRunOutcome(
        resolved.plan,
        resolved.run_dir,
        resolved.run_input_sha256,
        False,
        no_op,
        completed_manifest,
    )


__all__ = [
    "ExecutionBackend",
    "ResolvedSysIdRun",
    "SysIdExecutionResult",
    "SysIdInvocation",
    "SysIdResolverInputs",
    "SysIdRunOutcome",
    "execute_sysid_run",
    "resolve_sysid_run",
]
