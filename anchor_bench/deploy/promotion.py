"""Atomic promotion of one resolved frozen deployment state."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from anchor_bench.core.artifacts import atomic_write_yaml, ordered_locks
from anchor_bench.core.profiles import load_yaml_mapping
from anchor_bench.deploy.runner import (
    DeployResolverInputs,
    MetadataLoader,
    load_embedded_gru_metadata,
    load_resolved_deploy_profile,
    materialize_deploy_state,
)
from anchor_bench.deploy.schema import FrozenDeployState


@dataclass(frozen=True)
class DeployPromotionOutcome:
    """Result of one validated deployment promotion attempt."""

    accepted_state: Path
    state: FrozenDeployState
    no_op: bool
    dry_run: bool


def _load_existing_state(path: Path) -> FrozenDeployState | None:
    """Load an existing regular frozen state or return ``None`` when absent."""

    if path.is_symlink():
        raise ValueError(f"deployed state must not be a symlink: {path}")
    if not path.exists():
        return None
    if not path.is_file():
        raise ValueError(f"deployed state must be a regular file: {path}")
    return FrozenDeployState.from_dict(load_yaml_mapping(path), path)


def promote_deploy(
    inputs: DeployResolverInputs,
    profile_name: str = "default",
    *,
    metadata_loader: MetadataLoader = load_embedded_gru_metadata,
    dry_run: bool = False,
) -> DeployPromotionOutcome:
    """Freeze accepted sources and atomically replace ``deploy/default.yaml``."""

    profile = load_resolved_deploy_profile(inputs, profile_name)
    if dry_run:
        desired = materialize_deploy_state(
            inputs,
            profile,
            metadata_loader=metadata_loader,
        )
        return DeployPromotionOutcome(
            accepted_state=inputs.deployed_state,
            state=desired,
            no_op=_load_existing_state(inputs.deployed_state) == desired,
            dry_run=True,
        )
    with ordered_locks(
        inputs.accepted_training_state,
        inputs.accepted_sysid_state,
        inputs.deployed_state,
    ):
        desired = materialize_deploy_state(
            inputs,
            profile,
            metadata_loader=metadata_loader,
        )
        current = _load_existing_state(inputs.deployed_state)
        no_op = current == desired
        if not no_op:
            atomic_write_yaml(inputs.deployed_state, desired.to_dict())
        return DeployPromotionOutcome(
            accepted_state=inputs.deployed_state,
            state=desired,
            no_op=no_op,
            dry_run=False,
        )


__all__ = ["DeployPromotionOutcome", "promote_deploy"]
