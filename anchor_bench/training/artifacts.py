"""Fixed artifact surface and manifest-last completion for Training runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import yaml

from anchor_bench.core.artifacts import generated_files, write_manifest_last
from anchor_bench.core.hashing import canonical_sha256, sha256_file, validate_sha256
from anchor_bench.core.manifest_schema import RunManifest
from anchor_bench.training.archive import read_runtime_metadata_archive

if TYPE_CHECKING:
    from anchor_bench.training.runner import ResolvedTrainingRun


_FIXED_ROLES = {
    "resolved_config.yaml": "resolved_config",
    "sysid_state.yaml": "sysid_state",
    "network.pt": "network",
    "stats.json": "stats",
    "checkpoints/best.pt": "checkpoint",
}


def _write_yaml(path: Path, document: Mapping[str, object]) -> None:
    """Write one deterministic YAML mapping into a staging directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(dict(document), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def write_training_inputs(run_dir: Path, resolved: "ResolvedTrainingRun") -> None:
    """Freeze resolved configuration and selected SysID state before fitting."""

    _write_yaml(run_dir / "resolved_config.yaml", resolved.resolved_config.to_dict())
    _write_yaml(run_dir / "sysid_state.yaml", resolved.sysid_state.to_dict())


def _fingerprint_digest(fingerprint: Mapping[str, Any], *, label: str) -> str:
    """Read one conventional raw or tree digest from a fingerprint."""

    for key in ("sha256", f"{label}_sha256", "tree_sha256"):
        value = fingerprint.get(key)
        if isinstance(value, str):
            return validate_sha256(value, label=f"{label} fingerprint {key}")
    raise ValueError(f"{label} fingerprint is missing its content digest")


def _validate_stats(path: Path) -> None:
    """Require stats.json to contain one finite JSON object."""

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid Training stats {path}: {error}") from error
    if not isinstance(document, dict):
        raise ValueError("Training stats.json must contain a JSON object")


def _validate_network_contract(run_dir: Path, resolved: "ResolvedTrainingRun") -> None:
    """Bind embedded runtime metadata to this exact resolved run."""

    embedded = read_runtime_metadata_archive(run_dir / "network.pt")
    metadata = embedded.metadata
    expected_hashes = {
        "profile_sha256": resolved.profile_snapshot.content_sha256,
        "resolved_config_sha256": canonical_sha256(resolved.resolved_config.to_dict()),
        "sysid_state_sha256": sha256_file(run_dir / "sysid_state.yaml"),
        "asset_sha256": _fingerprint_digest(resolved.fingerprints["asset"], label="asset"),
        "source_sha256": _fingerprint_digest(resolved.fingerprints["dataset"], label="dataset"),
    }
    if metadata.hashes.to_dict() != expected_hashes:
        raise ValueError("embedded runtime hashes do not match the resolved Training run")
    expected_mappings = [
        {
            "input_joints": mapping.input_joints,
            "output_joints": mapping.output_joints,
        }
        for mapping in resolved.mappings
    ]
    actual_mappings = [
        {
            "input_joints": mapping.input_joints,
            "output_joints": mapping.output_joints,
        }
        for mapping in metadata.mappings
    ]
    if actual_mappings != expected_mappings:
        raise ValueError("embedded runtime mappings do not match resolved_config.yaml")
    if metadata.target_columns != (resolved.resolved_config.target,):
        raise ValueError("embedded target semantics do not match resolved_config.yaml")
    if metadata.sample_dt_s != resolved.resolved_config.sample_dt_s:
        raise ValueError("embedded sample_dt_s does not match resolved_config.yaml")


def write_training_manifest(
    run_dir: Path,
    resolved: "ResolvedTrainingRun",
) -> RunManifest:
    """Validate fixed outputs, authenticate them, and write manifest last."""

    for relative in _FIXED_ROLES:
        path = run_dir / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Training run is missing regular artifact {relative}")
    _validate_stats(run_dir / "stats.json")
    _validate_network_contract(run_dir, resolved)
    manifest = RunManifest.build(
        phase="train",
        robot=resolved.robot,
        profile=resolved.profile_snapshot,
        resolved_config=resolved.resolved_config.to_dict(),
        cli_overrides=resolved.cli_overrides,
        seed=resolved.resolved_config.fit.seed,
        replicate=resolved.replicate,
        run_input=resolved.run_input,
        run_input_sha256=resolved.run_input_sha256,
        fingerprints=resolved.fingerprints,
        upstream=resolved.upstream,
        generated_files=generated_files(run_dir, _FIXED_ROLES),
    )
    write_manifest_last(run_dir, manifest)
    return manifest


__all__ = ["write_training_inputs", "write_training_manifest"]
