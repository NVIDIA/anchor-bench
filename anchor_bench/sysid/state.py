"""Typed SysID states, sparse patches, and candidate reconstruction checks.

Manifest and hashing mechanics live in :mod:`anchor_bench.core`. This module
only binds already verified manifest entries to SysID state documents.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import AfterValidator, Field, FiniteFloat, field_validator, model_validator

from anchor_bench.core.config_model import ConfigModel
from anchor_bench.core.hashing import validate_sha256
from anchor_bench.core.profiles import load_yaml_mapping
from anchor_bench.sysid.schema import SUPPORTED_SYSID_PARAMETERS


Sha256 = Annotated[str, AfterValidator(validate_sha256)]
ParameterMaps = dict[str, dict[str, float]]


def _require_text(value: str, *, name: str) -> str:
    if not value or value != value.strip():
        raise ValueError(f"{name} must be non-empty and have no surrounding whitespace")
    return value


def _validate_joint_name(joint: str) -> None:
    if not joint or joint != joint.strip():
        raise ValueError("joint names must be non-empty and have no surrounding whitespace")


def _validate_parameter_input(value: Any, *, sparse: bool) -> Any:
    if not isinstance(value, Mapping):
        return value
    unsupported = sorted(set(value) - set(SUPPORTED_SYSID_PARAMETERS))
    if unsupported:
        raise ValueError(f"unsupported SysID parameter(s): {', '.join(unsupported)}")
    if not value:
        suffix = "parameter cell" if sparse else "parameter map"
        raise ValueError(f"must contain at least one {suffix}")
    for parameter, joint_map in value.items():
        if isinstance(joint_map, Mapping) and not joint_map:
            raise ValueError(f"parameter {parameter!r} must contain at least one joint cell")
    return value


def _validate_parameter_maps(parameters: ParameterMaps) -> ParameterMaps:
    for joint_map in parameters.values():
        for joint in joint_map:
            _validate_joint_name(joint)
    return parameters


class SysIdParameterCell(ConfigModel):
    parameter: Literal[
        "stiffness",
        "damping",
        "armature",
        "dry_friction",
        "viscous_friction",
    ]
    joint: str

    @field_validator("joint")
    @classmethod
    def _valid_joint(cls, value: str) -> str:
        _validate_joint_name(value)
        return value


class SysIdBenchmarkEvidence(ConfigModel):
    run: str
    run_manifest_sha256: Sha256

    @field_validator("run")
    @classmethod
    def _valid_run(cls, value: str) -> str:
        return _require_text(value, name="benchmark run")


class SysIdPromotionRecord(ConfigModel):
    """One accepted-state revision and the evidence that produced it."""

    revision: int = Field(ge=1)
    source_run: str
    run_input_sha256: Sha256
    run_manifest_sha256: Sha256
    parent_state_sha256: Sha256
    fitted_patch_sha256: Sha256
    candidate_state_sha256: Sha256
    promotion_parent_state_sha256: Sha256
    updated_cells: list[SysIdParameterCell] = Field(default_factory=list)
    benchmarks: list[SysIdBenchmarkEvidence] = Field(default_factory=list)

    @field_validator("source_run")
    @classmethod
    def _valid_source_run(cls, value: str) -> str:
        return _require_text(value, name="source run")

    @model_validator(mode="after")
    def _unique_evidence(self) -> "SysIdPromotionRecord":
        cells = [(cell.parameter, cell.joint) for cell in self.updated_cells]
        if len(cells) != len(set(cells)):
            raise ValueError("updated_cells contains duplicate parameter/joint cells")
        benchmarks = [(item.run, item.run_manifest_sha256) for item in self.benchmarks]
        if len(benchmarks) != len(set(benchmarks)):
            raise ValueError("benchmarks contains duplicate evidence")
        return self


class SysIdProvenance(ConfigModel):
    history: list[SysIdPromotionRecord] = Field(default_factory=list)


class AcceptedSysIdState(ConfigModel):
    """Machine-managed accepted state stored in ``sysid/default.yaml``."""

    schema_version: Literal[1]
    revision: int = Field(ge=0)
    parameters: dict[str, dict[str, FiniteFloat]]
    provenance: SysIdProvenance

    @model_validator(mode="before")
    @classmethod
    def _raw_history_length(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        provenance = data.get("provenance")
        history = provenance.get("history") if isinstance(provenance, Mapping) else None
        revision = data.get("revision")
        if isinstance(revision, int) and isinstance(history, list) and revision != len(history):
            raise ValueError("revision must equal provenance history length")
        return data

    @field_validator("parameters", mode="before")
    @classmethod
    def _parameters_input(cls, value: Any) -> Any:
        return _validate_parameter_input(value, sparse=False)

    @field_validator("parameters")
    @classmethod
    def _parameters(cls, value: ParameterMaps) -> ParameterMaps:
        return _validate_parameter_maps(value)

    @model_validator(mode="after")
    def _history(self) -> "AcceptedSysIdState":
        history = self.provenance.history
        if self.revision != len(history):
            raise ValueError("revision must equal provenance history length")
        if [record.revision for record in history] != list(range(1, self.revision + 1)):
            raise ValueError("provenance history revisions must be consecutive and ordered")
        return self


class SysIdFittedPatch(ConfigModel):
    """Sparse parameter cells optimized by one SysID run."""

    schema_version: Literal[1]
    artifact_type: Literal["sysid_fitted_patch"]
    robot: str
    parameters: dict[str, dict[str, FiniteFloat]]

    @field_validator("robot")
    @classmethod
    def _valid_robot(cls, value: str) -> str:
        return _require_text(value, name="robot")

    @field_validator("parameters", mode="before")
    @classmethod
    def _parameters_input(cls, value: Any) -> Any:
        return _validate_parameter_input(value, sparse=True)

    @field_validator("parameters")
    @classmethod
    def _parameters(cls, value: ParameterMaps) -> ParameterMaps:
        return _validate_parameter_maps(value)


class SysIdCandidateSource(ConfigModel):
    parent_state_sha256: Sha256
    fitted_patch_sha256: Sha256
    run_input_sha256: Sha256


class SysIdCandidateState(ConfigModel):
    """Complete immutable candidate state emitted by one SysID run."""

    schema_version: Literal[1]
    artifact_type: Literal["sysid_state"]
    robot: str
    parameters: dict[str, dict[str, FiniteFloat]]
    source: SysIdCandidateSource

    @field_validator("robot")
    @classmethod
    def _valid_robot(cls, value: str) -> str:
        return _require_text(value, name="robot")

    @field_validator("parameters", mode="before")
    @classmethod
    def _parameters_input(cls, value: Any) -> Any:
        return _validate_parameter_input(value, sparse=False)

    @field_validator("parameters")
    @classmethod
    def _parameters(cls, value: ParameterMaps) -> ParameterMaps:
        return _validate_parameter_maps(value)


# parent_state.yaml is an exact copy of either the accepted default or a prior
# candidate. A third wrapper schema would make that copy lossy.
SysIdParentState: TypeAlias = AcceptedSysIdState | SysIdCandidateState
CompleteSysIdState: TypeAlias = SysIdParentState


def parse_parent_state(data: Any, context: str | Path = "parent_state.yaml") -> SysIdParentState:
    """Parse either complete-state shape allowed as a run parent."""

    if not isinstance(data, Mapping):
        raise ValueError(f"{context} must be a mapping")
    artifact_type = data.get("artifact_type")
    if artifact_type is None:
        return AcceptedSysIdState.from_dict(data, context)
    if artifact_type == "sysid_state":
        return SysIdCandidateState.from_dict(data, context)
    raise ValueError(f"{context}.artifact_type must be 'sysid_state' when present")


def apply_fitted_patch(parent: SysIdParentState, patch: SysIdFittedPatch) -> ParameterMaps:
    """Apply a fitted patch without mutating the complete parent state."""

    if isinstance(parent, SysIdCandidateState) and parent.robot != patch.robot:
        raise ValueError(
            f"SysID robot mismatch: parent is {parent.robot!r}, patch is {patch.robot!r}"
        )
    result = {parameter: dict(joints) for parameter, joints in parent.parameters.items()}
    for parameter, joint_map in patch.parameters.items():
        if parameter not in result:
            raise ValueError(f"unknown parameter cell {parameter}/{next(iter(joint_map))}")
        for joint, value in joint_map.items():
            if joint not in result[parameter]:
                raise ValueError(f"unknown parameter cell {parameter}/{joint}")
            result[parameter][joint] = float(value)
    return result


def _parameter_differences(expected: ParameterMaps, actual: ParameterMaps) -> list[str]:
    differences: list[str] = []
    parameters: list[str] = list(SUPPORTED_SYSID_PARAMETERS)
    parameters.extend(sorted((set(expected) | set(actual)) - set(parameters)))
    for parameter in parameters:
        if parameter not in expected or parameter not in actual:
            differences.append(parameter)
            continue
        for joint in sorted(set(expected[parameter]) | set(actual[parameter])):
            if joint not in expected[parameter] or joint not in actual[parameter]:
                differences.append(f"{parameter}/{joint}")
            elif expected[parameter][joint] != actual[parameter][joint]:
                differences.append(f"{parameter}/{joint}")
    return differences


def verify_candidate_state(
    parent: SysIdParentState,
    patch: SysIdFittedPatch,
    candidate: SysIdCandidateState,
) -> SysIdCandidateState:
    """Verify the exact equation ``candidate = parent + patch``."""

    if patch.robot != candidate.robot:
        raise ValueError(
            f"SysID robot mismatch: patch is {patch.robot!r}, candidate is {candidate.robot!r}"
        )
    if isinstance(parent, SysIdCandidateState) and parent.robot != candidate.robot:
        raise ValueError(
            f"SysID robot mismatch: parent is {parent.robot!r}, candidate is {candidate.robot!r}"
        )
    expected = apply_fitted_patch(parent, patch)
    differences = _parameter_differences(expected, candidate.parameters)
    if differences:
        raise ValueError(
            "candidate state does not equal parent state plus fitted patch; "
            f"mismatched cells: {', '.join(differences)}"
        )
    return candidate


def build_candidate_state(
    parent: SysIdParentState,
    patch: SysIdFittedPatch,
    source: SysIdCandidateSource | Mapping[str, str],
) -> SysIdCandidateState:
    """Construct and self-check a complete candidate from its run artifacts."""

    source_model = (
        source
        if isinstance(source, SysIdCandidateSource)
        else SysIdCandidateSource.model_validate(dict(source))
    )
    candidate = SysIdCandidateState(
        schema_version=1,
        artifact_type="sysid_state",
        robot=patch.robot,
        parameters=apply_fitted_patch(parent, patch),
        source=source_model,
    )
    return verify_candidate_state(parent, patch, candidate)


def canonicalize_parameter_maps(
    parameters: Mapping[str, Mapping[str, float]],
    canonical_joint_names: Sequence[str],
    *,
    require_all_joints: bool = True,
) -> ParameterMaps:
    """Validate canonical keys and order maps by robot joint order."""

    canonical = tuple(canonical_joint_names)
    if not canonical:
        raise ValueError("canonical_joint_names must contain at least one joint")
    for joint in canonical:
        _validate_joint_name(joint)
    duplicates = sorted({joint for joint in canonical if canonical.count(joint) > 1})
    if duplicates:
        raise ValueError(f"duplicate canonical joint names: {', '.join(duplicates)}")
    unsupported = sorted(set(parameters) - set(SUPPORTED_SYSID_PARAMETERS))
    if unsupported:
        raise ValueError(f"unsupported SysID parameter(s): {', '.join(unsupported)}")

    canonical_set = set(canonical)
    ordered: ParameterMaps = {}
    for parameter in SUPPORTED_SYSID_PARAMETERS:
        joint_map = parameters.get(parameter)
        if joint_map is None:
            continue
        unknown = sorted(set(joint_map) - canonical_set)
        if unknown:
            raise ValueError(
                f"parameter {parameter!r} contains non-canonical or unknown joints: "
                f"{', '.join(unknown)}"
            )
        missing = [joint for joint in canonical if joint not in joint_map]
        if require_all_joints and missing:
            raise ValueError(
                f"parameter {parameter!r} is missing canonical joints: {', '.join(missing)}"
            )
        ordered[parameter] = {}
        for joint in canonical:
            if joint not in joint_map:
                continue
            try:
                value = float(joint_map[joint])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"parameter cell {parameter}/{joint} must be numeric") from exc
            if not math.isfinite(value):
                raise ValueError(f"parameter cell {parameter}/{joint} must be finite")
            ordered[parameter][joint] = value
    return ordered


def require_parameter_cells(
    state: CompleteSysIdState,
    parameter_names: Sequence[str],
    joint_names: Sequence[str],
) -> None:
    """Require every parameter/joint cell needed by a consuming phase."""

    missing = [
        f"{parameter}/{joint}"
        for parameter in parameter_names
        for joint in joint_names
        if parameter not in state.parameters or joint not in state.parameters[parameter]
    ]
    if missing:
        raise ValueError(f"missing required parameter cells: {', '.join(missing)}")


@dataclass(frozen=True)
class LoadedSysIdCandidate:
    """Schema-valid documents tied to a previously verified run manifest."""

    run_dir: Path
    manifest: Any
    parent_state: SysIdParentState
    fitted_patch: SysIdFittedPatch
    state: SysIdCandidateState


def _object_field(value: Any, name: str) -> Any:
    return value.get(name) if isinstance(value, Mapping) else getattr(value, name, None)


def _required_manifest_entry(manifest: Any, role: str, expected_path: str) -> Any:
    entries = [
        entry
        for entry in (_object_field(manifest, "generated_files") or [])
        if _object_field(entry, "role") == role
    ]
    if not entries:
        raise ValueError(f"missing required generated-file role: {role}")
    if len(entries) != 1:
        raise ValueError(f"generated-file role {role!r} must appear exactly once")
    entry = entries[0]
    if _object_field(entry, "path") != expected_path:
        raise ValueError(f"generated-file role {role!r} must use path {expected_path!r}")
    return entry


def load_verified_sysid_candidate(
    run_dir: Path,
    manifest: Any,
    *,
    expected_robot: str | None = None,
) -> LoadedSysIdCandidate:
    """Load documents using a manifest returned by the shared verifier."""

    run_dir = Path(run_dir)
    if _object_field(manifest, "phase") != "sysid":
        raise ValueError("expected a sysid run manifest")
    manifest_robot = _object_field(manifest, "robot")
    if expected_robot is not None and manifest_robot != expected_robot:
        raise ValueError(
            f"SysID robot mismatch: expected {expected_robot!r}, manifest is {manifest_robot!r}"
        )
    parent_entry = _required_manifest_entry(manifest, "parent_state", "parent_state.yaml")
    patch_entry = _required_manifest_entry(manifest, "fitted_patch", "fitted_patch.yaml")
    _required_manifest_entry(manifest, "sysid_state", "sysid_state.yaml")

    parent = parse_parent_state(
        load_yaml_mapping(run_dir / "parent_state.yaml"),
        run_dir / "parent_state.yaml",
    )
    patch = SysIdFittedPatch.from_dict(
        load_yaml_mapping(run_dir / "fitted_patch.yaml"),
        run_dir / "fitted_patch.yaml",
    )
    state = SysIdCandidateState.from_dict(
        load_yaml_mapping(run_dir / "sysid_state.yaml"),
        run_dir / "sysid_state.yaml",
    )
    if patch.robot != manifest_robot or state.robot != manifest_robot:
        raise ValueError("SysID robot mismatch between manifest, fitted patch, and candidate state")
    verify_candidate_state(parent, patch, state)

    source_checks = {
        "parent_state_sha256": _object_field(parent_entry, "sha256"),
        "fitted_patch_sha256": _object_field(patch_entry, "sha256"),
        "run_input_sha256": _object_field(manifest, "run_input_sha256"),
    }
    for field_name, expected in source_checks.items():
        actual = getattr(state.source, field_name)
        if actual != expected:
            raise ValueError(
                f"candidate source {field_name} does not match verified manifest: "
                f"{actual} != {expected}"
            )
    return LoadedSysIdCandidate(run_dir, manifest, parent, patch, state)


def load_external_sysid_candidate(
    run_dir: Path,
    *,
    expected_robot: str | None = None,
) -> LoadedSysIdCandidate:
    """Verify a completed run through the shared foundation, then load it."""

    from anchor_bench.core.artifacts import verify_run_directory

    manifest = verify_run_directory(Path(run_dir))
    return load_verified_sysid_candidate(Path(run_dir), manifest, expected_robot=expected_robot)


SysIdAcceptedState = AcceptedSysIdState
SysIdStateDoc = SysIdCandidateState


__all__ = [
    "AcceptedSysIdState",
    "CompleteSysIdState",
    "LoadedSysIdCandidate",
    "ParameterMaps",
    "Sha256",
    "SysIdAcceptedState",
    "SysIdBenchmarkEvidence",
    "SysIdCandidateSource",
    "SysIdCandidateState",
    "SysIdFittedPatch",
    "SysIdParameterCell",
    "SysIdParentState",
    "SysIdPromotionRecord",
    "SysIdProvenance",
    "SysIdStateDoc",
    "apply_fitted_patch",
    "build_candidate_state",
    "canonicalize_parameter_maps",
    "load_external_sysid_candidate",
    "load_verified_sysid_candidate",
    "parse_parent_state",
    "require_parameter_cells",
    "verify_candidate_state",
]
