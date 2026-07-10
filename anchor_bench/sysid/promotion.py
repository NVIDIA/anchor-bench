"""Explicit sparse promotion of an immutable SysID candidate."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from anchor_bench.core.artifacts import atomic_write_yaml, sibling_lock, verify_run_directory
from anchor_bench.core.hashing import canonical_sha256, repo_relative_posix
from anchor_bench.core.profiles import load_yaml_mapping
from anchor_bench.sysid.schema import SUPPORTED_SYSID_PARAMETERS, SysIdProfile
from anchor_bench.sysid.state import (
    AcceptedSysIdState,
    SysIdBenchmarkEvidence,
    SysIdParameterCell,
    SysIdPromotionRecord,
    SysIdProvenance,
    apply_fitted_patch,
    load_verified_sysid_candidate,
)


_REQUIRED_ROLES = {
    "resolved_config": "resolved_config.yaml",
    "parent_state": "parent_state.yaml",
    "fitted_patch": "fitted_patch.yaml",
    "sysid_state": "sysid_state.yaml",
    "metrics": "metrics.yaml",
}


@dataclass(frozen=True)
class SysIdPromotionOutcome:
    """Result of one locked accepted-state update."""

    accepted_state: Path
    state: AcceptedSysIdState
    no_op: bool
    source_manifest_sha256: str


def _field(value: Any, name: str) -> Any:
    """Read one field from a strict model or a mapping."""

    return value.get(name) if isinstance(value, Mapping) else getattr(value, name, None)


def _entry(manifest: Any, role: str, path: str) -> Any:
    """Require one generated artifact role at its canonical path."""

    matches = [
        item
        for item in manifest.generated_files
        if item.role == role
    ]
    if len(matches) != 1 or matches[0].path != path:
        raise ValueError(f"SysID manifest must contain exactly one {role!r} at {path!r}")
    return matches[0]


def _validate_source_manifest(manifest: Any) -> dict[str, Any]:
    """Require the complete SysID artifact surface needed by promotion."""

    entries = {
        role: _entry(manifest, role, path)
        for role, path in _REQUIRED_ROLES.items()
    }
    optimizer_entries = [
        item
        for item in manifest.generated_files
        if item.role == "optimizer_artifact" and item.path.startswith("optimizer/")
    ]
    if not optimizer_entries:
        raise ValueError("SysID manifest must contain optimizer evidence below optimizer/")
    return entries


def _validate_patch_contract(profile: SysIdProfile, patch: Any) -> None:
    """Tie fitted patch cells and values back to the resolved source profile."""

    configured = set(profile.parameters)
    if set(patch.parameters) != configured:
        raise ValueError("fitted patch parameters do not match the resolved SysID profile")
    expected_joints = tuple(profile.joints)
    for parameter in SUPPORTED_SYSID_PARAMETERS:
        spec = profile.parameters.get(parameter)
        if spec is None:
            continue
        joint_map = patch.parameters[parameter]
        if set(joint_map) != set(expected_joints):
            raise ValueError(
                f"fitted patch cells for {parameter!r} do not match resolved joints"
            )
        values = [float(joint_map[joint]) for joint in expected_joints]
        if any(not math.isfinite(value) for value in values):
            raise ValueError(f"fitted patch values for {parameter!r} must be finite")
        if any(value < spec.bounds.lower or value > spec.bounds.upper for value in values):
            raise ValueError(f"fitted patch values for {parameter!r} are outside bounds")
        if spec.mode == "shared" and any(value != values[0] for value in values[1:]):
            raise ValueError(f"shared fitted parameter {parameter!r} must have one value")


def _benchmark_evidence(
    benchmark_runs: Sequence[Path],
    *,
    repository_root: Path,
    source_run: str,
    source_manifest: Any,
) -> list[SysIdBenchmarkEvidence]:
    """Authenticate optional benchmarks against the exact promoted candidate."""

    evidence: list[SysIdBenchmarkEvidence] = []
    for run_dir in benchmark_runs:
        manifest = verify_run_directory(run_dir)
        if manifest.phase != "benchmark" or manifest.robot != source_manifest.robot:
            raise ValueError("benchmark evidence has the wrong phase or robot")
        source = manifest.upstream.get("source")
        expected = {
            "phase": "sysid",
            "robot": source_manifest.robot,
            "run": source_run,
            "run_input_sha256": source_manifest.run_input_sha256,
            "run_manifest_sha256": source_manifest.run_manifest_sha256,
        }
        if not isinstance(source, Mapping) or any(source.get(key) != value for key, value in expected.items()):
            raise ValueError("benchmark evidence does not reference the exact SysID source run")
        evidence.append(
            SysIdBenchmarkEvidence(
                run=repo_relative_posix(run_dir, repository_root=repository_root),
                run_manifest_sha256=manifest.run_manifest_sha256,
            )
        )
    keys = [(item.run, item.run_manifest_sha256) for item in evidence]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate benchmark evidence")
    return evidence


def promote_sysid_run(
    source_run: Path,
    accepted_state: Path,
    *,
    repository_root: Path,
    benchmark_runs: Sequence[Path] = (),
) -> SysIdPromotionOutcome:
    """Validate a candidate and atomically apply its sparse fitted patch."""

    source_manifest = verify_run_directory(source_run)
    if source_manifest.phase != "sysid":
        raise ValueError("promotion source must be a completed SysID run")
    accepted_relative = repo_relative_posix(
        accepted_state,
        repository_root=repository_root,
    )
    accepted_parts = PurePosixPath(accepted_relative).parts
    if (
        len(accepted_parts) != 5
        or accepted_parts[:2] != ("configs", "robots")
        or accepted_parts[3:] != ("sysid", "default.yaml")
    ):
        raise ValueError(
            "accepted_state must be configs/robots/<robot>/sysid/default.yaml"
        )
    accepted_robot = accepted_parts[2]
    if source_manifest.robot != accepted_robot:
        raise ValueError(
            f"SysID promotion robot mismatch: source is {source_manifest.robot!r}, "
            f"accepted state is for {accepted_robot!r}"
        )
    entries = _validate_source_manifest(source_manifest)
    loaded = load_verified_sysid_candidate(
        Path(source_run),
        source_manifest,
        expected_robot=source_manifest.robot,
    )
    profile = SysIdProfile.from_dict(
        source_manifest.resolved_config,
        "SysID source resolved_config",
    )
    _validate_patch_contract(profile, loaded.fitted_patch)
    source_relative = repo_relative_posix(
        source_run,
        repository_root=repository_root,
    )
    evidence = _benchmark_evidence(
        benchmark_runs,
        repository_root=repository_root,
        source_run=source_relative,
        source_manifest=source_manifest,
    )

    with sibling_lock(accepted_state):
        current = AcceptedSysIdState.from_dict(
            load_yaml_mapping(accepted_state),
            accepted_state,
        )
        promotion_parent_hash = canonical_sha256(current.to_dict())
        next_parameters = apply_fitted_patch(current, loaded.fitted_patch)
        changed = [
            SysIdParameterCell(parameter=parameter, joint=joint)
            for parameter in SUPPORTED_SYSID_PARAMETERS
            for joint in profile.joints
            if parameter in loaded.fitted_patch.parameters
            and current.parameters[parameter][joint] != next_parameters[parameter][joint]
        ]

        matching_history = [
            record
            for record in current.provenance.history
            if record.run_input_sha256 == source_manifest.run_input_sha256
            and record.run_manifest_sha256 == source_manifest.run_manifest_sha256
            and record.fitted_patch_sha256 == entries["fitted_patch"].sha256
        ]
        existing_evidence = {
            (item.run, item.run_manifest_sha256)
            for record in matching_history
            for item in record.benchmarks
        }
        new_evidence = [
            item
            for item in evidence
            if (item.run, item.run_manifest_sha256) not in existing_evidence
        ]
        if matching_history and not changed and not new_evidence:
            return SysIdPromotionOutcome(
                Path(accepted_state),
                current,
                True,
                source_manifest.run_manifest_sha256,
            )

        revision = current.revision + 1
        record = SysIdPromotionRecord(
            revision=revision,
            source_run=source_relative,
            run_input_sha256=source_manifest.run_input_sha256,
            run_manifest_sha256=source_manifest.run_manifest_sha256,
            parent_state_sha256=entries["parent_state"].sha256,
            fitted_patch_sha256=entries["fitted_patch"].sha256,
            candidate_state_sha256=entries["sysid_state"].sha256,
            promotion_parent_state_sha256=promotion_parent_hash,
            updated_cells=changed,
            benchmarks=new_evidence,
        )
        next_state = AcceptedSysIdState(
            schema_version=1,
            revision=revision,
            parameters=next_parameters,
            provenance=SysIdProvenance(
                history=[*current.provenance.history, record]
            ),
        )
        atomic_write_yaml(accepted_state, next_state.to_dict())
        return SysIdPromotionOutcome(
            Path(accepted_state),
            next_state,
            False,
            source_manifest.run_manifest_sha256,
        )


__all__ = ["SysIdPromotionOutcome", "promote_sysid_run"]
