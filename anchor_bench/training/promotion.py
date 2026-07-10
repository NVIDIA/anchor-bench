"""Explicit promotion of one immutable Training run into the model catalog."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anchor_bench.core.artifacts import atomic_write_yaml, sibling_lock, verify_run_directory
from anchor_bench.core.hashing import (
    canonical_sha256,
    contained_path,
    repo_relative_posix,
    validate_sha256,
)
from anchor_bench.core.profiles import load_yaml_mapping
from anchor_bench.core.paths import require_canonical_phase_run_dir
from anchor_bench.training.archive import (
    EmbeddedRuntimeMetadata,
    read_runtime_metadata_archive,
)
from anchor_bench.training.schema import (
    AcceptedTrainingContract,
    AcceptedTrainingModel,
    ResolvedTrainingMapping,
    TrainingAcceptedCatalog,
    TrainingArtifactReference,
    TrainingBenchmarkEvidence,
    TrainingProfile,
    TrainingSourceReference,
)


_REQUIRED_ROLES = {
    "resolved_config": "resolved_config.yaml",
    "sysid_state": "sysid_state.yaml",
    "network": "network.pt",
    "stats": "stats.json",
    "checkpoint": "checkpoints/best.pt",
}


@dataclass(frozen=True)
class TrainingPromotionOutcome:
    """Result of one validated accepted-catalog update."""

    accepted_catalog: Path
    catalog: TrainingAcceptedCatalog
    model_key: str
    no_op: bool
    dry_run: bool
    source_manifest_sha256: str


def _entry(manifest: Any, role: str, path: str) -> Any:
    """Require one generated artifact role at its canonical path."""

    matches = [item for item in manifest.generated_files if item.role == role]
    if len(matches) != 1 or matches[0].path != path:
        raise ValueError(
            f"Training manifest must contain exactly one {role!r} at {path!r}"
        )
    return matches[0]


def _required_entries(manifest: Any) -> dict[str, Any]:
    """Require the complete fixed Training artifact surface."""

    return {
        role: _entry(manifest, role, path)
        for role, path in _REQUIRED_ROLES.items()
    }


def _fingerprint_digest(fingerprint: Mapping[str, Any], *, label: str) -> str:
    """Read one conventional raw or tree digest from a fingerprint."""

    for key in ("sha256", f"{label}_sha256", "tree_sha256"):
        value = fingerprint.get(key)
        if isinstance(value, str):
            return validate_sha256(value, label=f"{label} fingerprint {key}")
    raise ValueError(f"Training manifest {label} fingerprint is missing its digest")


def _resolved_mappings(profile: TrainingProfile) -> tuple[ResolvedTrainingMapping, ...]:
    """Require resolved_config to contain only explicit canonical mappings."""

    result: list[ResolvedTrainingMapping] = []
    for mapping in profile.mappings:
        if mapping.input_joints == "all" or mapping.output_selection == "all":
            raise ValueError("Training resolved_config mappings must be explicit")
        result.append(
            ResolvedTrainingMapping(
                input_joints=mapping.input_joints,
                output_joints=mapping.output_selection,
            )
        )
    return tuple(result)


def _validate_archive_contract(
    source_run: Path,
    manifest: Any,
    entries: Mapping[str, Any],
    profile: TrainingProfile,
) -> EmbeddedRuntimeMetadata:
    """Bind the embedded runtime contract to exact manifest inputs."""

    embedded = read_runtime_metadata_archive(source_run / "network.pt")
    metadata = embedded.metadata
    mappings = _resolved_mappings(profile)
    expected_axes = [
        (mapping.input_joints, mapping.output_joints)
        for mapping in mappings
    ]
    actual_axes = [
        (mapping.input_joints, mapping.output_joints)
        for mapping in metadata.mappings
    ]
    if actual_axes != expected_axes:
        raise ValueError("embedded runtime mappings do not match resolved_config")
    if metadata.target_columns != (profile.target,):
        raise ValueError("embedded target semantics do not match resolved_config")
    if metadata.sample_dt_s != profile.sample_dt_s:
        raise ValueError("embedded sample_dt_s does not match resolved_config")

    fingerprints = manifest.fingerprints
    expected_hashes = {
        "profile_sha256": manifest.profile.content_sha256,
        "resolved_config_sha256": manifest.resolved_config_sha256,
        "sysid_state_sha256": entries["sysid_state"].sha256,
        "asset_sha256": _fingerprint_digest(fingerprints["asset"], label="asset"),
        "source_sha256": _fingerprint_digest(
            fingerprints["dataset"],
            label="dataset",
        ),
    }
    if metadata.hashes.to_dict() != expected_hashes:
        raise ValueError("embedded runtime hashes do not match the Training manifest")
    return embedded


def _benchmark_evidence(
    benchmark_runs: Sequence[Path],
    *,
    repository_root: Path,
    source_run: str,
    source_manifest: Any,
) -> tuple[TrainingBenchmarkEvidence, ...]:
    """Authenticate optional benchmarks against the exact Training candidate."""

    evidence: list[TrainingBenchmarkEvidence] = []
    expected_source = {
        "phase": "train",
        "robot": source_manifest.robot,
        "run": source_run,
        "run_input_sha256": source_manifest.run_input_sha256,
        "run_manifest_sha256": source_manifest.run_manifest_sha256,
    }
    for run_dir in benchmark_runs:
        manifest = verify_run_directory(run_dir)
        if manifest.phase != "benchmark" or manifest.robot != source_manifest.robot:
            raise ValueError("benchmark evidence has the wrong phase or robot")
        source = manifest.upstream.get("source")
        if not isinstance(source, Mapping) or dict(source) != expected_source:
            raise ValueError(
                "benchmark evidence does not reference the exact Training source run"
            )
        evidence.append(
            TrainingBenchmarkEvidence(
                run=repo_relative_posix(
                    run_dir,
                    repository_root=repository_root,
                ),
                run_manifest_sha256=manifest.run_manifest_sha256,
            )
        )
    identities = [(item.run, item.run_manifest_sha256) for item in evidence]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate benchmark evidence")
    return tuple(
        sorted(evidence, key=lambda item: (item.run, item.run_manifest_sha256))
    )


def _source_without_benchmarks(source: TrainingSourceReference) -> dict[str, Any]:
    """Return source identity fields that are independent of attached evidence."""

    return source.model_dump(mode="json", exclude={"benchmarks"})


def _merge_benchmarks(
    existing: Sequence[TrainingBenchmarkEvidence],
    added: Sequence[TrainingBenchmarkEvidence],
) -> tuple[TrainingBenchmarkEvidence, ...]:
    """Union exact benchmark evidence without silently removing old evidence."""

    by_identity = {
        (item.run, item.run_manifest_sha256): item
        for item in [*existing, *added]
    }
    return tuple(by_identity[key] for key in sorted(by_identity))


def _accepted_catalog_path(
    accepted_catalog: Path,
    *,
    repository_root: Path,
    robot: str,
) -> Path:
    """Bind promotion to this source robot's one accepted Training catalog."""

    root = Path(repository_root).resolve(strict=True)
    requested = Path(accepted_catalog)
    if requested.is_symlink():
        raise ValueError("accepted Training catalog must not be a symlink")
    actual = contained_path(root, requested, must_exist=True)
    expected = (
        root / "configs" / "robots" / robot / "train" / "default.yaml"
    ).resolve(strict=False)
    if actual != expected:
        raise ValueError(
            "accepted Training catalog must be "
            f"configs/robots/{robot}/train/default.yaml"
        )
    return actual


def _load_catalog(path: Path) -> TrainingAcceptedCatalog:
    """Load one strict accepted Training catalog."""

    return TrainingAcceptedCatalog.from_dict(load_yaml_mapping(path), path)


def _propose_catalog(
    current: TrainingAcceptedCatalog,
    *,
    model_key: str,
    artifact: TrainingArtifactReference,
    contract: AcceptedTrainingContract,
    source: TrainingSourceReference,
) -> tuple[TrainingAcceptedCatalog, bool]:
    """Return the next complete catalog, preserving same-source evidence."""

    existing = current.models.get(model_key)
    merged_source = source
    if existing is not None and (
        existing.artifact == artifact
        and existing.contract == contract
        and _source_without_benchmarks(existing.source)
        == _source_without_benchmarks(source)
    ):
        merged_source = source.model_copy(
            update={
                "benchmarks": _merge_benchmarks(
                    existing.source.benchmarks,
                    source.benchmarks,
                )
            }
        )
        if merged_source == existing.source:
            return current, True

    model_revision = 1 if existing is None else existing.revision + 1
    next_model = AcceptedTrainingModel(
        revision=model_revision,
        artifact=artifact,
        contract=contract,
        source=merged_source,
    )
    return TrainingAcceptedCatalog(
        schema_version=1,
        revision=current.revision + 1,
        models={**current.models, model_key: next_model},
    ), False


def promote_training_run(
    source_run: Path,
    accepted_catalog: Path,
    *,
    repository_root: Path,
    benchmark_runs: Sequence[Path] = (),
    dry_run: bool = False,
) -> TrainingPromotionOutcome:
    """Validate one candidate and atomically replace its complete model entry."""

    source_manifest = verify_run_directory(source_run)
    if source_manifest.phase != "train":
        raise ValueError("promotion source must be a completed Training run")
    source_path = require_canonical_phase_run_dir(
        source_run,
        repository_root=repository_root,
        phase="train",
        robot=source_manifest.robot,
        profile=source_manifest.profile.name,
        seed=source_manifest.seed,
        run_input_sha256=source_manifest.run_input_sha256,
    )
    entries = _required_entries(source_manifest)
    profile = TrainingProfile.from_dict(
        source_manifest.resolved_config,
        "Training source resolved_config",
    )
    embedded = _validate_archive_contract(
        source_path,
        source_manifest,
        entries,
        profile,
    )
    source_relative = repo_relative_posix(
        source_path,
        repository_root=repository_root,
    )
    evidence = _benchmark_evidence(
        benchmark_runs,
        repository_root=repository_root,
        source_run=source_relative,
        source_manifest=source_manifest,
    )
    model_key = source_manifest.profile.name
    artifact_path = repo_relative_posix(
        source_path / "network.pt",
        repository_root=repository_root,
    )
    contract = AcceptedTrainingContract(
        target=profile.target,
        sample_dt_s=profile.sample_dt_s,
        mappings=_resolved_mappings(profile),
    )
    base_source = TrainingSourceReference(
        run=source_relative,
        run_input_sha256=source_manifest.run_input_sha256,
        run_manifest_sha256=source_manifest.run_manifest_sha256,
        sysid_state_sha256=entries["sysid_state"].sha256,
        benchmarks=evidence,
    )
    artifact = TrainingArtifactReference(
        path=artifact_path,
        sha256=entries["network"].sha256,
        metadata_sha256=embedded.sha256,
    )
    catalog_path = _accepted_catalog_path(
        Path(accepted_catalog),
        repository_root=repository_root,
        robot=source_manifest.robot,
    )

    if dry_run:
        next_catalog, no_op = _propose_catalog(
            _load_catalog(catalog_path),
            model_key=model_key,
            artifact=artifact,
            contract=contract,
            source=base_source,
        )
        return TrainingPromotionOutcome(
            catalog_path,
            next_catalog,
            model_key,
            no_op,
            True,
            source_manifest.run_manifest_sha256,
        )

    with sibling_lock(catalog_path):
        next_catalog, no_op = _propose_catalog(
            _load_catalog(catalog_path),
            model_key=model_key,
            artifact=artifact,
            contract=contract,
            source=base_source,
        )
        if not no_op:
            atomic_write_yaml(catalog_path, next_catalog.to_dict())
        return TrainingPromotionOutcome(
            catalog_path,
            next_catalog,
            model_key,
            no_op,
            False,
            source_manifest.run_manifest_sha256,
        )


__all__ = ["TrainingPromotionOutcome", "promote_training_run"]
