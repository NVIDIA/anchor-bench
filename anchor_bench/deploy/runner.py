"""Host-safe deployment profile resolution and frozen-state materialization."""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, TypeAlias

from anchor_bench.benchmark.schema import BenchmarkCandidateReference
from anchor_bench.core.artifacts import ordered_locks, verify_run_directory
from anchor_bench.core.hashing import (
    canonical_sha256,
    contained_path,
    require_relative_posix_path,
    sha256_file,
    validate_sha256,
)
from anchor_bench.core.profiles import LoadedProfile, load_profile, load_yaml_mapping, require_logical_name
from anchor_bench.deploy.schema import (
    DeployProfile,
    DeploySourceProvenance,
    FrozenDeployComposition,
    FrozenDeployState,
    FrozenNeuralActuator,
    FrozenNeuralArtifact,
    FrozenNeuralFrictionBaseline,
    FrozenNeuralMapping,
    FrozenNeuralPDBaseline,
    FrozenPDActuator,
)
from anchor_bench.sysid.state import AcceptedSysIdState
from anchor_bench.training.archive import read_runtime_metadata_archive
from anchor_bench.training.metadata import RuntimeGRUMetadata, RuntimeMapping, StatValue
from anchor_bench.training.schema import AcceptedTrainingModel, TrainingAcceptedCatalog


SUPPORTED_DEPLOY_FEATURES = frozenset(
    {
        "position",
        "position_error",
        "velocity",
        "solver_pd",
        "dynamic_bias",
        "previous_torque",
    }
)


@dataclass(frozen=True)
class DeployResolverInputs:
    """Repository and accepted-state locations needed for one deployment."""

    repository_root: Path
    robot: str
    profiles_directory: Path
    accepted_training_state: Path
    accepted_sysid_state: Path
    deployed_state: Path
    canonical_joint_names: tuple[str, ...]
    joint_aliases: Mapping[str, str]

    def __post_init__(self) -> None:
        """Normalize paths and reject an invalid robot joint namespace."""

        root = Path(self.repository_root).resolve(strict=True)
        if root.is_symlink() or not root.is_dir():
            raise ValueError(f"repository_root must be a real directory: {root}")
        object.__setattr__(self, "repository_root", root)
        robot = require_logical_name(self.robot, label="robot name")
        config_root = root / "configs" / "robots" / robot
        expected_paths = {
            "profiles_directory": config_root / "deploy" / "profiles",
            "accepted_training_state": config_root / "train" / "default.yaml",
            "accepted_sysid_state": config_root / "sysid" / "default.yaml",
            "deployed_state": config_root / "deploy" / "default.yaml",
        }
        for field_name in (
            "profiles_directory",
            "accepted_training_state",
            "accepted_sysid_state",
            "deployed_state",
        ):
            candidate = Path(getattr(self, field_name))
            if not candidate.is_absolute():
                candidate = root / candidate
            candidate = Path(os.path.abspath(candidate))
            contained_path(
                root,
                candidate,
                must_exist=field_name != "deployed_state",
            )
            if candidate != expected_paths[field_name]:
                raise ValueError(
                    f"{field_name} must be {expected_paths[field_name]} for robot {robot!r}"
                )
            object.__setattr__(self, field_name, candidate)
        canonical, aliases = _joint_namespace(self.canonical_joint_names, self.joint_aliases)
        object.__setattr__(self, "canonical_joint_names", canonical)
        object.__setattr__(self, "joint_aliases", aliases)


@dataclass(frozen=True)
class ResolvedDeployProfile:
    """Selected profile bytes paired with canonical joint axes."""

    selected: LoadedProfile[DeployProfile]
    resolved: DeployProfile


@dataclass(frozen=True)
class LoadedNeuralMetadata:
    """Validated embedded metadata and the digest of its exact JSON bytes."""

    metadata: RuntimeGRUMetadata
    metadata_sha256: str

    def __post_init__(self) -> None:
        """Validate the metadata digest supplied by a loader."""

        validate_sha256(self.metadata_sha256, label="metadata_sha256")


MetadataLoader: TypeAlias = Callable[[Path], LoadedNeuralMetadata]


@dataclass(frozen=True)
class _VerifiedNeuralModel:
    """One authenticated accepted model reused by selected actuators."""

    accepted: AcceptedTrainingModel
    metadata: RuntimeGRUMetadata
    artifact_path: Path


def load_embedded_gru_metadata(artifact_path: Path) -> LoadedNeuralMetadata:
    """Read validated embedded metadata without importing Torch."""

    embedded = read_runtime_metadata_archive(artifact_path)
    return LoadedNeuralMetadata(
        metadata=embedded.metadata,
        metadata_sha256=embedded.sha256,
    )


def _joint_namespace(
    canonical_joint_names: Sequence[str],
    joint_aliases: Mapping[str, str],
) -> tuple[tuple[str, ...], dict[str, str]]:
    """Validate canonical joints and return direct alias-to-canonical mappings."""

    if isinstance(canonical_joint_names, (str, bytes)):
        raise ValueError("canonical_joint_names must be an ordered sequence")
    canonical = tuple(canonical_joint_names)
    if not canonical or any(
        not isinstance(name, str) or not name or name != name.strip()
        for name in canonical
    ):
        raise ValueError("canonical_joint_names must contain non-empty names")
    duplicates = sorted({name for index, name in enumerate(canonical) if name in canonical[:index]})
    if duplicates:
        raise ValueError("duplicate canonical joint names: " + ", ".join(duplicates))
    canonical_set = set(canonical)
    aliases: dict[str, str] = {}
    for alias, target in joint_aliases.items():
        if not isinstance(alias, str) or not alias or alias != alias.strip():
            raise ValueError("joint aliases must have non-empty names")
        if not isinstance(target, str) or target not in canonical_set:
            raise ValueError(f"joint alias {alias!r} has unknown canonical target {target!r}")
        if alias in canonical_set and alias != target:
            raise ValueError(f"joint alias {alias!r} conflicts with a canonical joint")
        aliases[alias] = target
    return canonical, aliases


def _canonical_joint(name: str, inputs: DeployResolverInputs) -> str:
    """Resolve one canonical joint name or declared alias."""

    if name in inputs.canonical_joint_names:
        return name
    resolved = inputs.joint_aliases.get(name)
    if resolved is None:
        raise ValueError(f"unknown deployment joint {name!r}")
    return resolved


def _canonical_joints(
    names: Sequence[str],
    inputs: DeployResolverInputs,
    *,
    label: str,
) -> tuple[str, ...]:
    """Canonicalize an ordered joint axis and reject alias collisions."""

    resolved = tuple(_canonical_joint(name, inputs) for name in names)
    duplicates = sorted({name for index, name in enumerate(resolved) if name in resolved[:index]})
    if duplicates:
        raise ValueError(f"duplicate {label} after alias resolution: {', '.join(duplicates)}")
    return resolved


def _canonical_joint_map(
    values: Mapping[str, float],
    inputs: DeployResolverInputs,
    *,
    label: str,
) -> dict[str, float]:
    """Canonicalize a joint-keyed map and reject alias collisions."""

    resolved: dict[str, float] = {}
    for name, value in values.items():
        canonical = _canonical_joint(name, inputs)
        if canonical in resolved:
            raise ValueError(f"duplicate {label} key after alias resolution: {canonical}")
        resolved[canonical] = float(value)
    return resolved


def canonicalize_deploy_profile(
    profile: DeployProfile,
    inputs: DeployResolverInputs,
) -> DeployProfile:
    """Resolve every profile joint and effort-map key to robot canonical names."""

    document = profile.model_dump(mode="python", exclude_none=True)
    composition = document["composition"]
    for actuator in composition.get("neural", []):
        for index, mapping in enumerate(actuator["mappings"]):
            mapping["input_joints"] = list(
                _canonical_joints(
                    mapping["input_joints"],
                    inputs,
                    label=f"neural mapping {index} input joints",
                )
            )
            mapping["output_joints"] = list(
                _canonical_joints(
                    mapping["output_joints"],
                    inputs,
                    label=f"neural mapping {index} output joints",
                )
            )
    for actuator in composition.get("pd", []):
        actuator["joints"] = list(
            _canonical_joints(
                actuator["joints"],
                inputs,
                label=f"PD actuator {actuator['name']!r} joints",
            )
        )
        actuator["effort_limits"] = _canonical_joint_map(
            actuator["effort_limits"],
            inputs,
            label=f"PD actuator {actuator['name']!r} effort_limits",
        )
    return DeployProfile.from_dict(document, "resolved deployment profile")


def load_resolved_deploy_profile(
    inputs: DeployResolverInputs,
    profile_name: str = "default",
) -> ResolvedDeployProfile:
    """Load one complete profile and resolve all joint aliases."""

    selected = load_profile(
        inputs.profiles_directory,
        profile_name,
        DeployProfile,
        context=f"deployment profile {profile_name!r}",
    )
    return ResolvedDeployProfile(
        selected=selected,
        resolved=canonicalize_deploy_profile(selected.model, inputs),
    )


def _load_state(path: Path, model_type: type[Any], *, label: str) -> tuple[Any, str]:
    """Load a regular accepted-state YAML file and hash its semantic content."""

    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file: {path}")
    model = model_type.from_dict(load_yaml_mapping(path), path)
    return model, canonical_sha256(model.to_dict())


def _contained_regular_file(
    repository_root: Path,
    relative_path: str,
    *,
    label: str,
) -> Path:
    """Resolve one repository-relative regular file without following a symlink."""

    relative = require_relative_posix_path(relative_path)
    raw = repository_root.joinpath(*PurePosixPath(relative).parts)
    if raw.is_symlink() or not raw.is_file():
        raise ValueError(f"{label} must be a regular file: {relative}")
    return contained_path(repository_root, relative, must_exist=True)


def _contained_run_directory(repository_root: Path, relative_path: str) -> Path:
    """Resolve one repository-relative real run directory."""

    relative = require_relative_posix_path(relative_path)
    raw = repository_root.joinpath(*PurePosixPath(relative).parts)
    if raw.is_symlink() or not raw.is_dir():
        raise ValueError(f"training source run must be a real directory: {relative}")
    return contained_path(repository_root, relative, must_exist=True)


def _manifest_network_entry(manifest: Any) -> Any:
    """Require exactly one authenticated network role at ``network.pt``."""

    matches = [
        entry
        for entry in manifest.generated_files
        if entry.role == "network" and entry.path == "network.pt"
    ]
    if len(matches) != 1:
        raise ValueError(
            "training manifest must contain exactly one 'network' role at 'network.pt'"
        )
    return matches[0]


def _fingerprint_digest(fingerprint: Any, *, label: str) -> str:
    """Read the training phase's conventional content digest."""

    if not isinstance(fingerprint, Mapping):
        raise ValueError(f"training manifest {label} fingerprint must be a mapping")
    for key in ("sha256", f"{label}_sha256", "tree_sha256"):
        value = fingerprint.get(key)
        if isinstance(value, str):
            return validate_sha256(value, label=f"{label} fingerprint {key}")
    raise ValueError(f"training manifest {label} fingerprint is missing its content digest")


def _validate_training_contract(
    model_key: str,
    accepted: AcceptedTrainingModel,
    metadata: RuntimeGRUMetadata,
    manifest: Any,
    inputs: DeployResolverInputs,
) -> None:
    """Bind embedded metadata to the accepted catalog and training manifest."""

    if metadata.hashes.profile_sha256 != manifest.profile.content_sha256:
        raise ValueError(
            f"accepted model {model_key!r} metadata profile hash does not match its source manifest"
        )
    if metadata.hashes.resolved_config_sha256 != manifest.resolved_config_sha256:
        raise ValueError(
            f"accepted model {model_key!r} metadata resolved-config hash does not match its source manifest"
        )
    sysid_hash = validate_sha256(
        accepted.source.sysid_state_sha256,
        label=f"accepted model {model_key!r} sysid_state_sha256",
    )
    if metadata.hashes.sysid_state_sha256 != sysid_hash:
        raise ValueError(
            f"accepted model {model_key!r} metadata SysID hash does not match its accepted source"
        )
    asset_hash = _fingerprint_digest(manifest.fingerprints.get("asset"), label="asset")
    if metadata.hashes.asset_sha256 != asset_hash:
        raise ValueError(
            f"accepted model {model_key!r} metadata asset hash does not match its source manifest"
        )
    source_hash = _fingerprint_digest(manifest.fingerprints.get("dataset"), label="dataset")
    if metadata.hashes.source_sha256 != source_hash:
        raise ValueError(
            f"accepted model {model_key!r} metadata source hash does not match its source manifest"
        )
    target = metadata.target_columns[0]
    if target != accepted.contract.target:
        raise ValueError(f"accepted model {model_key!r} target does not match embedded metadata")
    if metadata.sample_dt_s != accepted.contract.sample_dt_s:
        raise ValueError(f"accepted model {model_key!r} timestep does not match embedded metadata")
    accepted_axes = tuple(
        (mapping.input_joints, mapping.output_joints)
        for mapping in accepted.contract.mappings
    )
    metadata_axes = tuple(
        (mapping.input_joints, mapping.output_joints)
        for mapping in metadata.mappings
    )
    if metadata_axes != accepted_axes:
        raise ValueError(f"accepted model {model_key!r} mappings do not match embedded metadata")
    canonical = set(inputs.canonical_joint_names)
    unknown = sorted(
        {
            joint
            for input_joints, output_joints in metadata_axes
            for joint in (*input_joints, *output_joints)
            if joint not in canonical
        }
    )
    if unknown:
        raise ValueError(
            f"accepted model {model_key!r} metadata contains unknown robot joints: {', '.join(unknown)}"
        )
    unsupported = sorted(
        feature.name
        for feature in metadata.input_features
        if feature.name not in SUPPORTED_DEPLOY_FEATURES
    )
    if unsupported:
        raise ValueError(
            f"accepted model {model_key!r} has unsupported deployment features: {', '.join(unsupported)}"
        )


def _verify_accepted_neural_model(
    model_key: str,
    accepted: AcceptedTrainingModel,
    inputs: DeployResolverInputs,
    metadata_loader: MetadataLoader,
) -> _VerifiedNeuralModel:
    """Authenticate one accepted model run, archive, and embedded contract."""

    require_logical_name(model_key, label="accepted model key")
    if accepted.revision < 1:
        raise ValueError(f"accepted model {model_key!r} has an invalid revision")
    candidate = BenchmarkCandidateReference(
        phase="train",
        robot=inputs.robot,
        run=accepted.source.run,
        run_input_sha256=accepted.source.run_input_sha256,
        run_manifest_sha256=accepted.source.run_manifest_sha256,
    )
    source_parts = PurePosixPath(candidate.run).parts
    if source_parts[3] != model_key:
        raise ValueError(
            f"accepted model {model_key!r} source run must use the same training profile name"
        )
    source_run = _contained_run_directory(inputs.repository_root, candidate.run)
    manifest = verify_run_directory(
        source_run,
        candidate.run_input_sha256,
        expected_run_manifest_sha256=candidate.run_manifest_sha256,
    )
    if manifest.phase != "train" or manifest.robot != inputs.robot:
        raise ValueError(f"accepted model {model_key!r} source has the wrong phase or robot")
    if manifest.profile.name != model_key:
        raise ValueError(f"accepted model {model_key!r} source manifest has a different profile")

    artifact_relative = require_relative_posix_path(accepted.artifact.path)
    expected_artifact = f"{candidate.run}/network.pt"
    if artifact_relative != expected_artifact:
        raise ValueError(
            f"accepted model {model_key!r} artifact must be {expected_artifact!r}"
        )
    artifact_path = _contained_regular_file(
        inputs.repository_root,
        artifact_relative,
        label=f"accepted model {model_key!r} artifact",
    )
    artifact_hash = validate_sha256(
        accepted.artifact.sha256,
        label=f"accepted model {model_key!r} artifact sha256",
    )
    entry = _manifest_network_entry(manifest)
    if entry.sha256 != artifact_hash or sha256_file(artifact_path) != artifact_hash:
        raise ValueError(f"accepted model {model_key!r} artifact hash does not match")

    loaded = metadata_loader(artifact_path)
    expected_metadata_hash = validate_sha256(
        accepted.artifact.metadata_sha256,
        label=f"accepted model {model_key!r} metadata sha256",
    )
    if loaded.metadata_sha256 != expected_metadata_hash:
        raise ValueError(f"accepted model {model_key!r} embedded metadata hash does not match")
    _validate_training_contract(model_key, accepted, loaded.metadata, manifest, inputs)
    return _VerifiedNeuralModel(
        accepted=accepted,
        metadata=loaded.metadata,
        artifact_path=artifact_path,
    )


def _ordered_values(value: StatValue, width: int, *, label: str) -> tuple[float, ...]:
    """Expand one validated scalar/vector metadata value to an ordered tuple."""

    values = value if isinstance(value, tuple) else (value,)
    if len(values) != width:
        raise ValueError(f"{label} must contain {width} value(s)")
    result = tuple(float(item) for item in values)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{label} must contain finite values")
    return result


def _axis_map(names: Sequence[str], values: Sequence[float]) -> dict[str, float]:
    """Pair ordered physical axes with ordered scalar values."""

    return {name: float(value) for name, value in zip(names, values, strict=True)}


def _freeze_neural_mapping(index: int, mapping: RuntimeMapping) -> FrozenNeuralMapping:
    """Copy one selected metadata mapping into the deployment state."""

    pd_baseline = None
    if mapping.pd_baseline is not None:
        pd_baseline = FrozenNeuralPDBaseline(
            kp=_axis_map(
                mapping.input_joints,
                _ordered_values(
                    mapping.pd_baseline.kp,
                    len(mapping.input_joints),
                    label="pd_baseline.kp",
                ),
            ),
            kd=_axis_map(
                mapping.input_joints,
                _ordered_values(
                    mapping.pd_baseline.kd,
                    len(mapping.input_joints),
                    label="pd_baseline.kd",
                ),
            ),
            velocity_target=mapping.pd_baseline.velocity_target,
        )
    friction_baseline = None
    if mapping.friction_baseline is not None:
        friction_baseline = FrozenNeuralFrictionBaseline(
            dry=_axis_map(
                mapping.output_joints,
                _ordered_values(
                    mapping.friction_baseline.dry,
                    len(mapping.output_joints),
                    label="friction_baseline.dry",
                ),
            ),
            viscous=_axis_map(
                mapping.output_joints,
                _ordered_values(
                    mapping.friction_baseline.viscous,
                    len(mapping.output_joints),
                    label="friction_baseline.viscous",
                ),
            ),
        )
    armature = None
    if mapping.armature is not None:
        armature = _axis_map(
            mapping.output_joints,
            _ordered_values(
                mapping.armature.values,
                len(mapping.output_joints),
                label="armature.values",
            ),
        )
    return FrozenNeuralMapping(
        index=index,
        input_joints=mapping.input_joints,
        output_joints=mapping.output_joints,
        pd_baseline=pd_baseline,
        friction_baseline=friction_baseline,
        armature=armature,
        effort_limits=_axis_map(
            mapping.output_joints,
            _ordered_values(
                mapping.effort_limits,
                len(mapping.output_joints),
                label="effort_limits",
            ),
        ),
        clamping=mapping.clamping,
    )


def _freeze_neural_actuator(
    profile_actuator: Any,
    verified: _VerifiedNeuralModel,
) -> FrozenNeuralActuator:
    """Validate selected mapping axes and freeze one neural actuator."""

    metadata = verified.metadata
    mappings: list[FrozenNeuralMapping] = []
    for selected in profile_actuator.mappings:
        if selected.index >= len(metadata.mappings):
            raise ValueError(
                f"neural actuator {profile_actuator.name!r} mapping index {selected.index} "
                "is outside the accepted model"
            )
        mapping = metadata.mappings[selected.index]
        if mapping.input_joints != selected.input_joints:
            raise ValueError(
                f"neural actuator {profile_actuator.name!r} mapping {selected.index} "
                "input joints do not exactly match embedded metadata"
            )
        if mapping.output_joints != selected.output_joints:
            raise ValueError(
                f"neural actuator {profile_actuator.name!r} mapping {selected.index} "
                "output joints do not exactly match embedded metadata"
            )
        mappings.append(_freeze_neural_mapping(selected.index, mapping))
    return FrozenNeuralActuator(
        name=profile_actuator.name,
        model=profile_actuator.model,
        accepted_model_revision=verified.accepted.revision,
        artifact=FrozenNeuralArtifact(
            path=verified.accepted.artifact.path,
            sha256=verified.accepted.artifact.sha256,
            metadata_sha256=verified.accepted.artifact.metadata_sha256,
        ),
        target=metadata.target_columns[0],
        sample_dt_s=metadata.sample_dt_s,
        mappings=tuple(mappings),
    )


def _required_sysid_values(
    state: AcceptedSysIdState,
    parameter: str,
    joints: Sequence[str],
) -> dict[str, float]:
    """Select required finite SysID cells in profile joint order."""

    joint_map = state.parameters.get(parameter)
    if joint_map is None:
        raise ValueError(f"accepted SysID state is missing required parameter {parameter!r}")
    missing = [joint for joint in joints if joint not in joint_map]
    if missing:
        raise ValueError(
            f"accepted SysID parameter {parameter!r} is missing joints: {', '.join(missing)}"
        )
    return {joint: float(joint_map[joint]) for joint in joints}


def _optional_armature(
    state: AcceptedSysIdState,
    joints: Sequence[str],
) -> dict[str, float] | None:
    """Select armature cells when the accepted state defines that parameter."""

    if "armature" not in state.parameters:
        return None
    return _required_sysid_values(state, "armature", joints)


def _freeze_pd_actuator(profile_actuator: Any, state: AcceptedSysIdState) -> FrozenPDActuator:
    """Freeze one explicit PD owner from accepted SysID values."""

    joints = profile_actuator.joints
    return FrozenPDActuator(
        name=profile_actuator.name,
        joints=joints,
        kp=_required_sysid_values(state, "stiffness", joints),
        kd=_required_sysid_values(state, "damping", joints),
        dry_friction=_required_sysid_values(state, "dry_friction", joints),
        viscous_friction=_required_sysid_values(state, "viscous_friction", joints),
        armature=_optional_armature(state, joints),
        effort_limits=dict(profile_actuator.effort_limits),
    )


def materialize_deploy_state(
    inputs: DeployResolverInputs,
    profile: ResolvedDeployProfile,
    *,
    metadata_loader: MetadataLoader = load_embedded_gru_metadata,
) -> FrozenDeployState:
    """Build a frozen state from the accepted files in one source snapshot."""

    training, training_hash = _load_state(
        inputs.accepted_training_state,
        TrainingAcceptedCatalog,
        label="accepted training state",
    )
    sysid, sysid_hash = _load_state(
        inputs.accepted_sysid_state,
        AcceptedSysIdState,
        label="accepted SysID state",
    )

    verified_models: dict[str, _VerifiedNeuralModel] = {}
    neural: list[FrozenNeuralActuator] = []
    expected_dt = 1.0 / profile.resolved.simulation.control_freq
    for actuator in profile.resolved.composition.neural:
        accepted = training.models.get(actuator.model)
        if accepted is None:
            raise ValueError(f"deployment references unaccepted neural model {actuator.model!r}")
        if accepted.revision > training.revision:
            raise ValueError(
                f"accepted model {actuator.model!r} revision exceeds the training catalog revision"
            )
        verified = verified_models.get(actuator.model)
        if verified is None:
            verified = _verify_accepted_neural_model(
                actuator.model,
                accepted,
                inputs,
                metadata_loader,
            )
            verified_models[actuator.model] = verified
        if not math.isclose(
            verified.metadata.sample_dt_s,
            expected_dt,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                f"accepted model {actuator.model!r} timestep does not match deployment control frequency"
            )
        neural.append(_freeze_neural_actuator(actuator, verified))

    pd = tuple(
        _freeze_pd_actuator(actuator, sysid)
        for actuator in profile.resolved.composition.pd
    )
    composition = FrozenDeployComposition(neural=tuple(neural), pd=pd)
    return FrozenDeployState(
        schema_version=1,
        simulation=profile.resolved.simulation,
        ownership=composition.output_owners(),
        composition=composition,
        source=DeploySourceProvenance(
            profile=profile.selected.name,
            profile_sha256=profile.selected.content_sha256,
            sysid_revision=sysid.revision,
            sysid_state_sha256=sysid_hash,
            train_revision=training.revision,
            train_state_sha256=training_hash,
        ),
    )


def resolve_deploy_state(
    inputs: DeployResolverInputs,
    profile_name: str = "default",
    *,
    metadata_loader: MetadataLoader = load_embedded_gru_metadata,
) -> FrozenDeployState:
    """Resolve a selected profile from one locked accepted-state snapshot."""

    profile = load_resolved_deploy_profile(inputs, profile_name)
    with ordered_locks(
        inputs.accepted_training_state,
        inputs.accepted_sysid_state,
        inputs.deployed_state,
    ):
        return materialize_deploy_state(
            inputs,
            profile,
            metadata_loader=metadata_loader,
        )


__all__ = [
    "DeployResolverInputs",
    "LoadedNeuralMetadata",
    "MetadataLoader",
    "ResolvedDeployProfile",
    "SUPPORTED_DEPLOY_FEATURES",
    "canonicalize_deploy_profile",
    "load_embedded_gru_metadata",
    "load_resolved_deploy_profile",
    "materialize_deploy_state",
    "resolve_deploy_state",
]
