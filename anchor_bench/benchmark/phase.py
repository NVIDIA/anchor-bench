"""Host-safe resolution and immutable execution for candidate benchmarks.

The resolver authenticates a completed SysID or training run, freezes held-out
motion identity, and validates the runtime contract before a replay backend is
allowed to import Newton or Torch.  The executor invokes that backend only for
a fresh, non-dry run.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias

from anchor_bench.benchmark.schema import (
    BenchmarkCandidateReference,
    BenchmarkProfile,
    BenchmarkSignal,
    BenchmarkSimulationConfig,
)
from anchor_bench.core.artifacts import RunManifest, run_slot, verify_run_directory
from anchor_bench.core.hashing import (
    canonical_object,
    canonical_sha256,
    contained_path,
    repo_relative_posix,
    require_relative_posix_path,
    tree_fingerprint,
)
from anchor_bench.core.manifest_schema import GeneratedFile, ProfileSnapshot
from anchor_bench.core.paths import benchmark_run_dir
from anchor_bench.core.plan import CommandPlan, command_identity_metadata
from anchor_bench.core.profiles import load_profile, require_logical_name
from anchor_bench.robots.fingerprints import fingerprint_contains_preview
from anchor_bench.sysid.state import LoadedSysIdCandidate, load_verified_sysid_candidate
from anchor_bench.training.archive import (
    EmbeddedRuntimeMetadata,
    read_runtime_metadata_archive,
)
from anchor_bench.training.metadata import RuntimeGRUMetadata


_SAGE_FILES = ("control.csv", "state_motor.csv", "joint_list.txt", "metadata.yaml")
_SOURCE_ARTIFACTS = {
    "sysid": {
        "parent_state": "parent_state.yaml",
        "fitted_patch": "fitted_patch.yaml",
        "sysid_state": "sysid_state.yaml",
    },
}


@dataclass(frozen=True)
class BenchmarkInvocation:
    """Typed benchmark command inputs and whole-field profile replacements."""

    robot: str
    run: Path
    profile: str = "default"
    dry_run: bool = True
    headless: bool = False
    data: str | None = None
    motions: Sequence[str] | None = None
    simulation: BenchmarkSimulationConfig | None = None
    num_envs: int | None = None
    buffer_time: float | None = None
    sync_initial_state: bool | None = None
    signals: Sequence[BenchmarkSignal] | None = None
    output_root: str | None = None


LoadedRuntimeMetadata = EmbeddedRuntimeMetadata


@dataclass(frozen=True)
class BenchmarkMotionSource:
    """One authenticated held-out SAGE motion known before runtime startup."""

    name: str
    path: Path
    repository_path: str
    joint_names: tuple[str, ...]
    canonical_joint_names: tuple[str, ...]
    fingerprint: dict[str, Any]


@dataclass(frozen=True)
class SysIdBenchmarkCandidate:
    """A reconstructed SysID candidate and its explicit replay contract."""

    loaded: LoadedSysIdCandidate
    selected_joints: tuple[str, ...]
    control_freq: int
    artifact_hashes: dict[str, str]


@dataclass(frozen=True)
class TrainingBenchmarkCandidate:
    """A self-contained neural archive and authenticated schema-v3 contract."""

    network_path: Path
    metadata: RuntimeGRUMetadata
    metadata_sha256: str
    input_joints: tuple[str, ...]
    output_joints: tuple[str, ...]
    artifact_hashes: dict[str, str]


BenchmarkCandidate: TypeAlias = SysIdBenchmarkCandidate | TrainingBenchmarkCandidate
NeuralMetadataLoader = Callable[[Path], LoadedRuntimeMetadata]
RunDirectoryResolver = Callable[..., Path]
ReplayBackend = Callable[["ResolvedBenchmarkRun"], "BenchmarkExecutionResult"]


@dataclass(frozen=True)
class BenchmarkResolverInputs:
    """Repository, robot, fingerprint, and injectable loader dependencies."""

    repository_root: Path
    profiles_directory: Path
    canonical_joint_names: Sequence[str]
    joint_aliases: Mapping[str, str]
    robot_identity_fingerprint: Mapping[str, Any]
    asset_fingerprint: Mapping[str, Any]
    source_fingerprint: Mapping[str, Any]
    dependency_fingerprints: Mapping[str, Any]
    metadata_loader: NeuralMetadataLoader | None = None
    run_directory_resolver: RunDirectoryResolver = benchmark_run_dir


@dataclass(frozen=True)
class BenchmarkTrajectoryResult:
    """Real and simulated CSV bytes returned for one selected motion."""

    real_csv: str | bytes
    simulated_csv: str | bytes


@dataclass(frozen=True)
class BenchmarkExecutionResult:
    """Serializable replay results returned by an injected runtime backend."""

    metrics: Mapping[str, Any]
    trajectories: Mapping[str, BenchmarkTrajectoryResult]

    def validated_metrics(self) -> dict[str, Any]:
        """Return finite canonical metrics or reject the backend result."""

        if not isinstance(self.metrics, Mapping):
            raise TypeError("benchmark metrics must be a mapping")
        result = dict(self.metrics)
        if not result:
            raise ValueError("benchmark metrics must not be empty")
        canonical_object(result)
        return result


@dataclass(frozen=True)
class ResolvedBenchmarkRun:
    """Complete semantic benchmark input and deterministic output location."""

    repository_root: Path
    robot: str
    profile_name: str
    profile_snapshot: ProfileSnapshot
    resolved_profile: BenchmarkProfile
    resolved_config: dict[str, Any]
    cli_overrides: dict[str, Any]
    source_reference: BenchmarkCandidateReference
    source_manifest: RunManifest
    candidate: BenchmarkCandidate
    dataset_path: Path
    motions: tuple[BenchmarkMotionSource, ...]
    fingerprints: dict[str, Any]
    run_input: dict[str, Any]
    run_input_sha256: str
    run_dir: Path
    dry_run: bool
    headless: bool
    plan: CommandPlan


@dataclass(frozen=True)
class BenchmarkRunOutcome:
    """Result shared by dry runs, completed runs, and validated no-ops."""

    plan: CommandPlan
    run_dir: Path
    run_input_sha256: str
    dry_run: bool
    no_op: bool
    manifest: RunManifest | None = None


def load_embedded_runtime_metadata(network_path: Path) -> LoadedRuntimeMetadata:
    """Read schema-v3 metadata through the shared host-safe archive reader."""

    return read_runtime_metadata_archive(network_path)


def _override_document(invocation: BenchmarkInvocation) -> dict[str, Any]:
    """Extract only semantic typed replacements explicitly supplied by CLI."""

    result: dict[str, Any] = {}
    for name in (
        "data",
        "motions",
        "simulation",
        "num_envs",
        "buffer_time",
        "sync_initial_state",
        "signals",
    ):
        value = getattr(invocation, name)
        if value is None:
            continue
        if hasattr(value, "to_dict"):
            result[name] = value.to_dict()
        elif isinstance(value, Sequence) and not isinstance(value, str):
            result[name] = list(value)
        else:
            result[name] = value
    return result


def _manifest_entry(manifest: RunManifest, role: str, expected_path: str) -> GeneratedFile:
    """Return one exact generated-file role/path from a verified manifest."""

    entries = [entry for entry in manifest.generated_files if entry.role == role]
    if len(entries) != 1:
        raise ValueError(f"source run must contain exactly one generated-file role {role!r}")
    entry = entries[0]
    if entry.path != expected_path:
        raise ValueError(
            f"source generated-file role {role!r} must use path {expected_path!r}"
        )
    return entry


def _canonicalize_joint_names(
    names: Sequence[str],
    *,
    canonical_joint_names: Sequence[str],
    aliases: Mapping[str, str],
    context: str,
    permit_unmapped: bool,
) -> tuple[str, ...]:
    """Resolve robot aliases and reject duplicate physical joint columns."""

    canonical = tuple(canonical_joint_names)
    if not canonical or len(canonical) != len(set(canonical)):
        raise ValueError("canonical_joint_names must be non-empty and unique")
    canonical_set = set(canonical)
    bad_targets = sorted({target for target in aliases.values() if target not in canonical_set})
    if bad_targets:
        raise ValueError("joint aliases target unknown canonical joints: " + ", ".join(bad_targets))
    result: list[str] = []
    mapped: list[str] = []
    for name in names:
        if not isinstance(name, str) or not name or name != name.strip():
            raise ValueError(f"{context} contains an invalid joint name")
        resolved = aliases.get(name, name)
        if resolved not in canonical_set:
            if permit_unmapped:
                result.append(name)
                continue
            raise ValueError(f"{context} contains unknown robot joint {name!r}")
        if resolved in mapped:
            raise ValueError(f"{context} maps more than one column to robot joint {resolved!r}")
        mapped.append(resolved)
        result.append(resolved)
    return tuple(result)


def _is_sage_motion(path: Path) -> bool:
    """Return whether a real directory contains the four required SAGE files."""

    return path.is_dir() and not path.is_symlink() and all((path / name).is_file() for name in _SAGE_FILES)


def _discover_motion_paths(dataset: Path, selection: Literal["*"] | tuple[str, ...]) -> tuple[Path, ...]:
    """Resolve explicit motion children or discover every SAGE motion."""

    if selection == "*":
        paths = tuple(sorted((path for path in dataset.rglob("*") if _is_sage_motion(path))))
    else:
        paths = tuple(
            contained_path(dataset, require_relative_posix_path(name), must_exist=True)
            for name in selection
        )
        invalid = [path for path in paths if not _is_sage_motion(path)]
        if invalid:
            raise ValueError("selected paths are not SAGE motions: " + ", ".join(map(str, invalid)))
    if not paths:
        raise ValueError(f"benchmark dataset contains no selected SAGE motions: {dataset}")
    return paths


def _dataset_relative_selection(
    selection: Literal["*"] | tuple[str, ...], dataset_relative: str
) -> Literal["*"] | tuple[str, ...]:
    """Accept either dataset-child names or repository paths inside the dataset."""

    if selection == "*":
        return selection
    prefix = f"{dataset_relative}/"
    normalized: list[str] = []
    for raw_name in selection:
        name = require_relative_posix_path(raw_name)
        normalized.append(name.removeprefix(prefix) if name.startswith(prefix) else name)
    return tuple(normalized)


def _csv_columns(path: Path) -> set[str]:
    """Read one CSV header without loading trajectory arrays."""

    try:
        with path.open(newline="", encoding="utf-8") as stream:
            return set(csv.DictReader(stream).fieldnames or [])
    except OSError as error:
        raise ValueError(f"cannot read benchmark state header {path}: {error}") from error


def _load_motion_sources(
    *,
    repository_root: Path,
    data: str,
    selection: Literal["*"] | tuple[str, ...],
    signals: Sequence[BenchmarkSignal],
    sync_initial_state: bool,
    canonical_joint_names: Sequence[str],
    aliases: Mapping[str, str],
) -> tuple[tuple[BenchmarkMotionSource, ...], dict[str, Any]]:
    """Fingerprint selected SAGE motions and inspect joints before runtime."""

    dataset_relative = require_relative_posix_path(data)
    dataset = contained_path(repository_root, dataset_relative, must_exist=True)
    dataset_fingerprint = tree_fingerprint(dataset)
    paths = _discover_motion_paths(
        dataset, _dataset_relative_selection(selection, dataset_relative)
    )
    motions: list[BenchmarkMotionSource] = []
    for path in paths:
        name = path.relative_to(dataset).as_posix()
        raw_names = tuple(
            line.strip()
            for line in (path / "joint_list.txt").read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if not raw_names:
            raise ValueError(f"{path}/joint_list.txt must contain at least one joint")
        if len(raw_names) != len(set(raw_names)):
            raise ValueError(f"{path}/joint_list.txt contains duplicate joint names")
        canonical_names = _canonicalize_joint_names(
            raw_names,
            canonical_joint_names=canonical_joint_names,
            aliases=aliases,
            context=f"{path}/joint_list.txt",
            permit_unmapped=True,
        )
        control_columns = _csv_columns(path / "control.csv")
        missing_control = sorted({"timestamp", "positions"} - control_columns)
        if missing_control:
            raise ValueError(
                f"{path}/control.csv missing required columns: {', '.join(missing_control)}"
            )
        state_columns = _csv_columns(path / "state_motor.csv")
        required_columns = {"timestamp", "positions"}
        if "velocity" in signals or sync_initial_state:
            required_columns.add("velocities")
        missing_columns = sorted(required_columns - state_columns)
        if missing_columns:
            raise ValueError(
                f"{path}/state_motor.csv missing required columns: {', '.join(missing_columns)}"
            )
        motions.append(
            BenchmarkMotionSource(
                name=name,
                path=path,
                repository_path=repo_relative_posix(path, repository_root=repository_root),
                joint_names=raw_names,
                canonical_joint_names=canonical_names,
                fingerprint=tree_fingerprint(path),
            )
        )
    fingerprint = {
        "dataset": {"path": dataset_relative, **dataset_fingerprint},
        "motions": [
            {
                "name": motion.name,
                "path": motion.repository_path,
                "joint_names": list(motion.joint_names),
                "canonical_joint_names": list(motion.canonical_joint_names),
                **motion.fingerprint,
            }
            for motion in motions
        ],
    }
    canonical_object(fingerprint)
    return tuple(motions), fingerprint


def _source_reference(
    run_dir: Path,
    manifest: RunManifest,
    *,
    repository_root: Path,
) -> BenchmarkCandidateReference:
    """Bind a verified source manifest to its canonical repository path."""

    relative = repo_relative_posix(run_dir, repository_root=repository_root)
    parts = Path(relative).parts
    if len(parts) != 6:
        raise ValueError("source run does not use the canonical immutable-run layout")
    if parts[3] != manifest.profile.name:
        raise ValueError(
            "source run profile directory does not match manifest profile name"
        )
    if manifest.seed is None or parts[4] != f"seed-{manifest.seed}":
        raise ValueError("source run seed directory does not match manifest seed")
    return BenchmarkCandidateReference(
        phase=manifest.phase,
        robot=manifest.robot,
        run=relative,
        run_input_sha256=manifest.run_input_sha256,
        run_manifest_sha256=manifest.run_manifest_sha256,
    )


def _fingerprint_digest(
    fingerprint: Mapping[str, Any], keys: Sequence[str], *, label: str
) -> str:
    """Read one required digest from a source manifest fingerprint."""

    matches = [fingerprint[key] for key in keys if key in fingerprint]
    if len(matches) != 1 or not isinstance(matches[0], str):
        raise ValueError(f"training source manifest must provide one {label} digest")
    return matches[0]


def _load_sysid_candidate(
    run_dir: Path,
    manifest: RunManifest,
    *,
    expected_robot: str,
    canonical_joint_names: Sequence[str],
    aliases: Mapping[str, str],
) -> SysIdBenchmarkCandidate:
    """Reconstruct and validate a SysID replay candidate from verified files."""

    artifacts = {
        role: _manifest_entry(manifest, role, path).sha256
        for role, path in _SOURCE_ARTIFACTS["sysid"].items()
    }
    loaded = load_verified_sysid_candidate(run_dir, manifest, expected_robot=expected_robot)
    config = manifest.resolved_config
    raw_joints = config.get("joints")
    simulation = config.get("simulation")
    if not isinstance(raw_joints, Sequence) or isinstance(raw_joints, str):
        raise ValueError("SysID source resolved_config.joints must be an ordered list")
    if not isinstance(simulation, Mapping):
        raise ValueError("SysID source resolved_config.simulation must be a mapping")
    control_freq = simulation.get("control_freq")
    if isinstance(control_freq, bool) or not isinstance(control_freq, int) or control_freq <= 0:
        raise ValueError("SysID source simulation.control_freq must be a positive integer")
    selected = _canonicalize_joint_names(
        raw_joints,
        canonical_joint_names=canonical_joint_names,
        aliases=aliases,
        context="SysID source joints",
        permit_unmapped=False,
    )
    return SysIdBenchmarkCandidate(
        loaded=loaded,
        selected_joints=selected,
        control_freq=control_freq,
        artifact_hashes=artifacts,
    )


def _validate_metadata_hashes(metadata: RuntimeGRUMetadata, manifest: RunManifest) -> None:
    """Bind embedded metadata hashes to the authenticated training manifest."""

    if metadata.hashes.profile_sha256 != manifest.profile.content_sha256:
        raise ValueError("embedded profile_sha256 does not match the training manifest")
    if metadata.hashes.resolved_config_sha256 != manifest.resolved_config_sha256:
        raise ValueError("embedded resolved_config_sha256 does not match the training manifest")
    sysid_state = _manifest_entry(manifest, "sysid_state", "sysid_state.yaml")
    if metadata.hashes.sysid_state_sha256 != sysid_state.sha256:
        raise ValueError("embedded sysid_state_sha256 does not match the training manifest")
    asset = manifest.fingerprints.get("asset")
    dataset = manifest.fingerprints.get("dataset")
    if not isinstance(asset, Mapping) or not isinstance(dataset, Mapping):
        raise ValueError("training source manifest is missing asset/dataset fingerprints")
    asset_sha = _fingerprint_digest(
        asset, ("asset_sha256", "tree_sha256", "sha256"), label="asset"
    )
    source_sha = _fingerprint_digest(
        dataset, ("dataset_sha256", "tree_sha256", "sha256"), label="dataset"
    )
    if metadata.hashes.asset_sha256 != asset_sha:
        raise ValueError("embedded asset_sha256 does not match the training manifest")
    if metadata.hashes.source_sha256 != source_sha:
        raise ValueError("embedded source_sha256 does not match the training dataset fingerprint")


def _ordered_union(rows: Sequence[Sequence[str]]) -> tuple[str, ...]:
    """Return first-seen joint names across ordered axes."""

    result: list[str] = []
    for row in rows:
        for value in row:
            if value not in result:
                result.append(value)
    return tuple(result)


def _load_training_candidate(
    run_dir: Path,
    manifest: RunManifest,
    *,
    expected_robot: str,
    canonical_joint_names: Sequence[str],
    aliases: Mapping[str, str],
    loader: NeuralMetadataLoader,
) -> TrainingBenchmarkCandidate:
    """Authenticate one network and validate its self-contained runtime contract."""

    if manifest.robot != expected_robot:
        raise ValueError(
            f"training source robot mismatch: expected {expected_robot!r}, got {manifest.robot!r}"
        )
    network_entry = _manifest_entry(manifest, "network", "network.pt")
    network_path = run_dir / network_entry.path
    loaded = loader(network_path)
    if not isinstance(loaded, LoadedRuntimeMetadata):
        raise TypeError("metadata_loader must return LoadedRuntimeMetadata")
    metadata = loaded.metadata
    if not isinstance(metadata, RuntimeGRUMetadata):
        raise TypeError("metadata_loader returned an invalid runtime metadata model")
    _validate_metadata_hashes(metadata, manifest)
    inputs: list[tuple[str, ...]] = []
    outputs: list[tuple[str, ...]] = []
    for index, mapping in enumerate(metadata.mappings):
        inputs.append(
            _canonicalize_joint_names(
                mapping.input_joints,
                canonical_joint_names=canonical_joint_names,
                aliases=aliases,
                context=f"neural mapping {index} input_joints",
                permit_unmapped=False,
            )
        )
        outputs.append(
            _canonicalize_joint_names(
                mapping.output_joints,
                canonical_joint_names=canonical_joint_names,
                aliases=aliases,
                context=f"neural mapping {index} output_joints",
                permit_unmapped=False,
            )
        )
    return TrainingBenchmarkCandidate(
        network_path=network_path,
        metadata=metadata,
        metadata_sha256=loaded.sha256,
        input_joints=_ordered_union(inputs),
        output_joints=_ordered_union(outputs),
        artifact_hashes={
            "network": network_entry.sha256,
            "metadata": loaded.sha256,
        },
    )


def _validate_timing(candidate: BenchmarkCandidate, control_freq: int) -> None:
    """Fail resolution when candidate and benchmark control timesteps differ."""

    expected_dt = 1.0 / control_freq
    candidate_dt = (
        1.0 / candidate.control_freq
        if isinstance(candidate, SysIdBenchmarkCandidate)
        else candidate.metadata.sample_dt_s
    )
    if not math.isclose(candidate_dt, expected_dt, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(
            f"candidate timestep {candidate_dt} does not match benchmark timestep {expected_dt}"
        )


def _required_axes(candidate: BenchmarkCandidate) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return candidate input and metric-output joints in stable order."""

    if isinstance(candidate, SysIdBenchmarkCandidate):
        return candidate.selected_joints, candidate.selected_joints
    return candidate.input_joints, candidate.output_joints


def _validate_motion_joints(
    motions: Sequence[BenchmarkMotionSource], candidate: BenchmarkCandidate
) -> None:
    """Require every candidate input/output axis in every selected motion."""

    inputs, outputs = _required_axes(candidate)
    required = _ordered_union((inputs, outputs))
    for motion in motions:
        available = set(motion.canonical_joint_names)
        missing = [joint for joint in required if joint not in available]
        if missing:
            raise ValueError(
                f"benchmark motion {motion.name!r} is missing candidate joints: {', '.join(missing)}"
            )


def _semantic_profile_document(profile: BenchmarkProfile) -> dict[str, Any]:
    """Remove storage-only output_root from benchmark semantic identity."""

    document = profile.to_dict()
    document.pop("output_root")
    return document


def _candidate_identity(
    reference: BenchmarkCandidateReference, candidate: BenchmarkCandidate
) -> dict[str, Any]:
    """Return source and artifact identities included in benchmark hashing."""

    result: dict[str, Any] = {
        "source": reference.to_dict(),
        "artifacts": dict(candidate.artifact_hashes),
    }
    if isinstance(candidate, SysIdBenchmarkCandidate):
        result["contract"] = {
            "joints": list(candidate.selected_joints),
            "control_freq": candidate.control_freq,
        }
    else:
        result["contract"] = {
            "metadata_sha256": candidate.metadata_sha256,
            "input_joints": list(candidate.input_joints),
            "output_joints": list(candidate.output_joints),
            "sample_dt_s": candidate.metadata.sample_dt_s,
        }
    return result


def _command_plan(
    invocation: BenchmarkInvocation,
    run_dir: Path,
    run_input_sha256: str,
    source: BenchmarkCandidateReference,
    motions: Sequence[BenchmarkMotionSource],
    *,
    preview_identity: bool,
) -> CommandPlan:
    """Build the lightweight summary carried by a resolved benchmark."""

    command = [
        "anchor-bench",
        "benchmark",
        "--robot",
        invocation.robot,
        "--run",
        source.run,
        "--profile",
        invocation.profile,
    ]
    for motion in invocation.motions or ():
        command.extend(["--motion", motion])
    if invocation.output_root is not None:
        command.extend(["--output-root", invocation.output_root])
    if invocation.headless:
        command.append("--headless")
    if invocation.dry_run:
        command.append("--dry-run")
    return CommandPlan(
        workflow="benchmark",
        robot=invocation.robot,
        command=command,
        dry_run=invocation.dry_run,
        notes=[
            "evaluates one exact immutable SysID or training candidate",
            "benchmark output is evidence only and never promotes accepted state",
        ],
        metadata={
            "profile": invocation.profile,
            "source": source.to_dict(),
            "motions": [motion.name for motion in motions],
            **command_identity_metadata(
                run_dir,
                run_input_sha256,
                preview=preview_identity,
            ),
        },
    )


def resolve_benchmark_run(
    invocation: BenchmarkInvocation, inputs: BenchmarkResolverInputs
) -> ResolvedBenchmarkRun:
    """Resolve and authenticate a candidate benchmark without runtime imports."""

    robot = require_logical_name(invocation.robot, label="robot name")
    repository_root = Path(inputs.repository_root).resolve(strict=True)
    source_dir = contained_path(repository_root, invocation.run, must_exist=True)
    source_manifest = verify_run_directory(source_dir)
    if source_manifest.phase not in ("sysid", "train"):
        raise ValueError("benchmark source must be a completed SysID or training run")
    if source_manifest.robot != robot:
        raise ValueError(
            f"benchmark robot mismatch: requested {robot!r}, source is {source_manifest.robot!r}"
        )
    source_reference = _source_reference(
        source_dir, source_manifest, repository_root=repository_root
    )

    loaded_profile = load_profile(
        inputs.profiles_directory, invocation.profile, BenchmarkProfile
    )
    cli_overrides = _override_document(invocation)
    resolved_document = loaded_profile.model.to_dict()
    resolved_document.update(cli_overrides)
    if invocation.output_root is not None:
        resolved_document["output_root"] = invocation.output_root
    resolved_profile = BenchmarkProfile.from_dict(
        resolved_document, "resolved benchmark config"
    )

    loader = inputs.metadata_loader or load_embedded_runtime_metadata
    if source_manifest.phase == "sysid":
        candidate: BenchmarkCandidate = _load_sysid_candidate(
            source_dir,
            source_manifest,
            expected_robot=robot,
            canonical_joint_names=inputs.canonical_joint_names,
            aliases=inputs.joint_aliases,
        )
    else:
        candidate = _load_training_candidate(
            source_dir,
            source_manifest,
            expected_robot=robot,
            canonical_joint_names=inputs.canonical_joint_names,
            aliases=inputs.joint_aliases,
            loader=loader,
        )
    _validate_timing(candidate, resolved_profile.simulation.control_freq)

    motions, motion_fingerprints = _load_motion_sources(
        repository_root=repository_root,
        data=resolved_profile.data,
        selection=resolved_profile.motions,
        signals=resolved_profile.signals,
        sync_initial_state=resolved_profile.sync_initial_state,
        canonical_joint_names=inputs.canonical_joint_names,
        aliases=inputs.joint_aliases,
    )
    _validate_motion_joints(motions, candidate)

    fingerprints = {
        **motion_fingerprints,
        "robot": dict(inputs.robot_identity_fingerprint),
        "asset": dict(inputs.asset_fingerprint),
        "source": dict(inputs.source_fingerprint),
        "dependencies": dict(inputs.dependency_fingerprints),
    }
    canonical_object(fingerprints)
    semantic_profile = _semantic_profile_document(resolved_profile)
    semantic_profile_source = dict(loaded_profile.content)
    semantic_profile_source.pop("output_root")
    profile_document = loaded_profile.manifest_data(
        repository_root=repository_root
    )
    profile_document["content"] = semantic_profile_source
    profile_document["content_sha256"] = canonical_sha256(
        semantic_profile_source
    )
    profile_snapshot = ProfileSnapshot.from_dict(
        profile_document,
        "benchmark profile snapshot",
    )
    run_input = {
        "schema_version": 1,
        "phase": "benchmark",
        "robot": robot,
        "profile": {
            "name": invocation.profile,
            "content": semantic_profile_source,
            "content_sha256": canonical_sha256(semantic_profile_source),
        },
        "resolved_config": semantic_profile,
        "seed": None,
        "replicate": None,
        "candidate": _candidate_identity(source_reference, candidate),
        "fingerprints": fingerprints,
    }
    run_input_sha256 = canonical_sha256(run_input)
    output_directory = contained_path(
        repository_root,
        require_relative_posix_path(resolved_profile.output_root),
        must_exist=False,
    )
    run_dir = inputs.run_directory_resolver(
        output_directory=output_directory,
        robot=robot,
        source_phase=source_reference.phase,
        profile=invocation.profile,
        source_manifest_sha256=source_reference.run_manifest_sha256,
        benchmark_input_sha256=run_input_sha256,
    )
    plan = _command_plan(
        invocation,
        run_dir,
        run_input_sha256,
        source_reference,
        motions,
        preview_identity=fingerprint_contains_preview(fingerprints),
    )
    return ResolvedBenchmarkRun(
        repository_root=repository_root,
        robot=robot,
        profile_name=invocation.profile,
        profile_snapshot=profile_snapshot,
        resolved_profile=resolved_profile,
        resolved_config=semantic_profile,
        cli_overrides=cli_overrides,
        source_reference=source_reference,
        source_manifest=source_manifest,
        candidate=candidate,
        dataset_path=contained_path(
            repository_root,
            resolved_profile.data,
            must_exist=True,
        ),
        motions=motions,
        fingerprints=fingerprints,
        run_input=run_input,
        run_input_sha256=run_input_sha256,
        run_dir=run_dir,
        dry_run=invocation.dry_run,
        headless=invocation.headless,
        plan=plan,
    )


def execute_benchmark_run(
    resolved: ResolvedBenchmarkRun,
    backend: ReplayBackend | None = None,
) -> BenchmarkRunOutcome:
    """Execute once through the immutable protocol, or return a no-op."""

    if resolved.dry_run:
        return BenchmarkRunOutcome(
            resolved.plan,
            resolved.run_dir,
            resolved.run_input_sha256,
            True,
            False,
        )
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
            if backend is None:
                raise ValueError("a benchmark replay backend is required for a fresh non-dry run")
            if slot.tmp_dir is None:  # pragma: no cover - foundation invariant
                raise RuntimeError("fresh benchmark run has no staging directory")
            result = backend(resolved)
            if not isinstance(result, BenchmarkExecutionResult):
                raise TypeError("benchmark backend must return BenchmarkExecutionResult")
            from anchor_bench.benchmark.artifacts import write_benchmark_run

            completed_manifest = write_benchmark_run(slot.tmp_dir, resolved, result)
    return BenchmarkRunOutcome(
        resolved.plan,
        resolved.run_dir,
        resolved.run_input_sha256,
        False,
        no_op,
        completed_manifest,
    )


__all__ = [
    "BenchmarkCandidate",
    "BenchmarkExecutionResult",
    "BenchmarkInvocation",
    "BenchmarkMotionSource",
    "BenchmarkResolverInputs",
    "BenchmarkRunOutcome",
    "BenchmarkTrajectoryResult",
    "LoadedRuntimeMetadata",
    "ResolvedBenchmarkRun",
    "SysIdBenchmarkCandidate",
    "TrainingBenchmarkCandidate",
    "execute_benchmark_run",
    "load_embedded_runtime_metadata",
    "resolve_benchmark_run",
]
