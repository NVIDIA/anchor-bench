"""Writers for one immutable SysID run.

The execution backend returns ordinary Python data.  This module owns the
on-disk representation, candidate reconstruction, generated-file inventory,
and manifest-last completion protocol.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from anchor_bench.core.artifacts import generated_files, write_manifest_last
from anchor_bench.core.hashing import require_relative_posix_path, sha256_file
from anchor_bench.core.manifest_schema import RunManifest
from anchor_bench.sysid.schema import SUPPORTED_SYSID_PARAMETERS
from anchor_bench.sysid.state import (
    SysIdCandidateSource,
    SysIdFittedPatch,
    build_candidate_state,
)

if TYPE_CHECKING:
    from anchor_bench.sysid.runner import ResolvedSysIdRun, SysIdExecutionResult


_FIXED_ROLES = {
    "resolved_config.yaml": "resolved_config",
    "parent_state.yaml": "parent_state",
    "fitted_patch.yaml": "fitted_patch",
    "sysid_state.yaml": "sysid_state",
    "metrics.yaml": "metrics",
}


def _write_yaml(path: Path, document: Mapping[str, object]) -> None:
    """Write one deterministic UTF-8 YAML mapping into a staging directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(dict(document), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _write_optimizer_artifacts(
    run_dir: Path,
    artifacts: Mapping[str, str | bytes],
) -> dict[str, str]:
    """Write authenticated backend evidence below ``optimizer/``."""

    if not artifacts:
        raise ValueError("SysID execution must produce at least one optimizer artifact")
    roles: dict[str, str] = {}
    for raw_name, payload in sorted(artifacts.items()):
        name = require_relative_posix_path(raw_name)
        relative = f"optimizer/{name}"
        target = run_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(payload, str):
            target.write_text(payload, encoding="utf-8")
        elif isinstance(payload, bytes):
            target.write_bytes(payload)
        else:
            raise TypeError(f"optimizer artifact {name!r} must be str or bytes")
        roles[relative] = "optimizer_artifact"
    return roles


def write_sysid_run(
    run_dir: Path,
    resolved: "ResolvedSysIdRun",
    result: "SysIdExecutionResult",
) -> RunManifest:
    """Write, authenticate, and finalize one staged SysID result."""

    patch = SysIdFittedPatch(
        schema_version=1,
        artifact_type="sysid_fitted_patch",
        robot=resolved.robot,
        parameters=resolved.validate_fitted_parameters(result.fitted_parameters),
    )

    _write_yaml(run_dir / "resolved_config.yaml", resolved.resolved_config.to_dict())
    _write_yaml(run_dir / "parent_state.yaml", resolved.parent_state.to_dict())
    _write_yaml(run_dir / "fitted_patch.yaml", patch.to_dict())
    _write_yaml(run_dir / "metrics.yaml", result.validated_metrics())
    optimizer_roles = _write_optimizer_artifacts(run_dir, result.optimizer_artifacts)

    candidate = build_candidate_state(
        resolved.parent_state,
        patch,
        SysIdCandidateSource(
            parent_state_sha256=sha256_file(run_dir / "parent_state.yaml"),
            fitted_patch_sha256=sha256_file(run_dir / "fitted_patch.yaml"),
            run_input_sha256=resolved.run_input_sha256,
        ),
    )
    _write_yaml(run_dir / "sysid_state.yaml", candidate.to_dict())

    roles = {**_FIXED_ROLES, **optimizer_roles}
    manifest = RunManifest.build(
        phase="sysid",
        robot=resolved.robot,
        profile=resolved.profile_snapshot,
        resolved_config=resolved.resolved_config.to_dict(),
        cli_overrides=resolved.cli_overrides,
        seed=resolved.resolved_config.seed,
        replicate=resolved.replicate,
        run_input=resolved.run_input,
        run_input_sha256=resolved.run_input_sha256,
        fingerprints=resolved.fingerprints,
        upstream=resolved.upstream,
        generated_files=generated_files(run_dir, roles),
    )
    write_manifest_last(run_dir, manifest)
    return manifest


def expected_fitted_cells(resolved: "ResolvedSysIdRun") -> tuple[tuple[str, str], ...]:
    """Return the exact sparse cells that one resolved run must fit."""

    cells: list[tuple[str, str]] = []
    for parameter in SUPPORTED_SYSID_PARAMETERS:
        if parameter not in resolved.resolved_config.parameters:
            continue
        cells.extend((parameter, joint) for joint in resolved.resolved_config.joints)
    return tuple(cells)


__all__ = ["expected_fitted_cells", "write_sysid_run"]
