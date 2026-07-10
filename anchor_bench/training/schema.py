"""Strict configuration and accepted-state models for actuator training."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import Field, StrictBool, field_validator, model_validator

from anchor_bench.core.config_model import ConfigModel
from anchor_bench.core.hashing import require_relative_posix_path, validate_sha256
from anchor_bench.core.profiles import require_logical_name


TrainingTarget = Literal["torque", "torque_residual"]
FeatureDomain = Literal["input", "output"]
JointSelection = Literal["all"] | tuple[str, ...]
SUPPORTED_FEATURES = frozenset(
    {
        "position",
        "position_error",
        "velocity",
        "solver_pd",
        "dynamic_bias",
        "previous_torque",
    }
)


def _ordered_names(value: Any, *, label: str) -> tuple[str, ...]:
    """Return a non-empty ordered tuple of unique names."""

    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be an ordered list")
    names = tuple(str(item).strip() for item in value)
    if not names or any(not name for name in names):
        raise ValueError(f"{label} must contain non-empty names")
    duplicates = sorted({name for index, name in enumerate(names) if name in names[:index]})
    if duplicates:
        raise ValueError(f"duplicate {label}: {', '.join(duplicates)}")
    return names


def _joint_selection(value: Any, *, label: str) -> JointSelection:
    """Parse either the training-only ``all`` selector or explicit joints."""

    if isinstance(value, str) and value.strip() == "all":
        return "all"
    names = _ordered_names(value, label=label)
    if "all" in names:
        raise ValueError("'all' must be used as a standalone joint selector")
    return names


class TrainingMapping(ConfigModel):
    """One physical application of a shared set of network weights."""

    input_joints: JointSelection
    output_joints: JointSelection | None = None

    @field_validator("input_joints", mode="before")
    @classmethod
    def _parse_inputs(cls, value: Any) -> JointSelection:
        """Parse the mapping's explicit or all-joint input selection."""

        return _joint_selection(value, label="input_joints")

    @field_validator("output_joints", mode="before")
    @classmethod
    def _parse_outputs(cls, value: Any) -> JointSelection | None:
        """Parse an optional output selection, defaulting later to the inputs."""

        return None if value is None else _joint_selection(value, label="output_joints")

    @model_validator(mode="after")
    def _validate_subset(self) -> "TrainingMapping":
        """Require explicit outputs to be a subset of the mapping inputs."""

        outputs = self.output_joints
        if self.input_joints == "all":
            if outputs not in (None, "all"):
                raise ValueError("output_joints must be omitted or 'all' when input_joints is 'all'")
            return self
        if outputs == "all":
            return self
        resolved_outputs = self.input_joints if outputs is None else outputs
        missing = [name for name in resolved_outputs if name not in self.input_joints]
        if missing:
            raise ValueError("output_joints must be a subset of input_joints: " + ", ".join(missing))
        return self
    @property
    def output_selection(self) -> JointSelection:
        """Return the effective output selection before robot-name resolution."""

        return self.input_joints if self.output_joints is None else self.output_joints


class TrainingFeature(ConfigModel):
    """One ordered network feature block and its axis/component semantics."""

    name: str
    domain: FeatureDomain
    components: tuple[str, ...] = ("value",)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        """Validate the feature name against the supported training set."""

        name = value.strip()
        if name not in SUPPORTED_FEATURES:
            raise ValueError(f"unsupported training feature {name!r}")
        return name

    @field_validator("components", mode="before")
    @classmethod
    def _validate_components(cls, value: Any) -> tuple[str, ...]:
        """Parse the feature's ordered component axis."""

        return _ordered_names(value, label="feature components")

    @model_validator(mode="after")
    def _validate_domain(self) -> "TrainingFeature":
        """Match each feature to its input/output domain and component contract."""

        expected = "output" if self.name == "previous_torque" else "input"
        if self.domain != expected:
            raise ValueError(f"feature {self.name!r} must use domain {expected!r}")
        if self.name == "previous_torque" and self.components != ("value",):
            raise ValueError("previous_torque must contain exactly the 'value' component")
        return self


class TrainingNetwork(ConfigModel):
    """GRU architecture selected independently from torque target semantics."""

    type: Literal["gru"] = "gru"
    hidden_size: int = Field(default=64, ge=1)
    num_layers: int = Field(default=1, ge=1)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)


class TrainingFit(ConfigModel):
    """Reproducible fitting and optional hyperparameter-search settings."""

    skip_optuna: StrictBool = True
    trials: int = Field(default=20, ge=1)
    trial_epochs: int = Field(default=5, ge=1)
    final_epochs: int = Field(default=50, ge=1)
    noise_std: float = Field(default=0.0, ge=0.0)
    seed: int = 42


class TrainingProfile(ConfigModel):
    """A mutable recipe that resolves to exactly one shared-weight artifact."""

    schema_version: Literal[1] = 1
    sysid_state: str = "default"
    data: str
    sample_dt_s: float = Field(gt=0.0)
    network: TrainingNetwork = Field(default_factory=TrainingNetwork)
    target: TrainingTarget
    mappings: tuple[TrainingMapping, ...] = Field(min_length=1)
    input_features: tuple[TrainingFeature, ...] = Field(min_length=1)
    fit: TrainingFit = Field(default_factory=TrainingFit)

    @field_validator("sysid_state", "data")
    @classmethod
    def _nonempty_text(cls, value: str) -> str:
        """Normalize required SysID-state and dataset references."""

        result = value.strip()
        if not result:
            raise ValueError("must not be empty")
        return result

    @model_validator(mode="after")
    def _validate_contract(self) -> "TrainingProfile":
        """Validate cadence, feature uniqueness, and shared-weight mapping shapes."""

        if not math.isfinite(self.sample_dt_s):
            raise ValueError("sample_dt_s must be finite")
        feature_names = [feature.name for feature in self.input_features]
        duplicates = sorted({name for index, name in enumerate(feature_names) if name in feature_names[:index]})
        if duplicates:
            raise ValueError("duplicate input features: " + ", ".join(duplicates))

        identities = [(mapping.input_joints, mapping.output_selection) for mapping in self.mappings]
        duplicate_mappings = [identity for index, identity in enumerate(identities) if identity in identities[:index]]
        if duplicate_mappings:
            raise ValueError("mappings must not contain duplicates")

        if any(mapping.input_joints == "all" for mapping in self.mappings):
            if len(self.mappings) != 1:
                raise ValueError("input_joints 'all' is allowed only in a single coupled mapping")
            return self

        first = self.mappings[0]
        assert first.input_joints != "all"
        first_outputs = first.output_selection
        assert first_outputs != "all"
        expected = (len(first.input_joints), len(first_outputs))
        for index, mapping in enumerate(self.mappings[1:], start=1):
            assert mapping.input_joints != "all"
            outputs = mapping.output_selection
            assert outputs != "all"
            shape = (len(mapping.input_joints), len(outputs))
            if shape != expected:
                raise ValueError(
                    f"mapping {index} has shape {shape[0]}->{shape[1]}; "
                    f"expected {expected[0]}->{expected[1]}"
                )
        return self


class ResolvedTrainingMapping(ConfigModel):
    """Canonical physical axes used by episodes, metadata, and catalogs."""

    input_joints: tuple[str, ...]
    output_joints: tuple[str, ...]

    @field_validator("input_joints", "output_joints", mode="before")
    @classmethod
    def _validate_joints(cls, value: Any, info: Any) -> tuple[str, ...]:
        """Parse one canonical resolved training axis."""

        return _ordered_names(value, label=info.field_name)

    @model_validator(mode="after")
    def _validate_subset(self) -> "ResolvedTrainingMapping":
        """Require canonical output joints to be a subset of canonical inputs."""

        missing = [name for name in self.output_joints if name not in self.input_joints]
        if missing:
            raise ValueError("output_joints must be a subset of input_joints: " + ", ".join(missing))
        return self


class TrainingArtifactReference(ConfigModel):
    """Immutable network archive and its authenticated metadata hashes."""

    path: str
    sha256: str
    metadata_sha256: str

    @field_validator("path")
    @classmethod
    def _artifact_path(cls, value: str) -> str:
        """Require a normalized repository-relative artifact path."""

        return require_relative_posix_path(value)

    @field_validator("sha256", "metadata_sha256")
    @classmethod
    def _artifact_hashes(cls, value: str, info: Any) -> str:
        """Require exact lowercase artifact and metadata digests."""

        return validate_sha256(value, label=info.field_name)


class AcceptedTrainingContract(ConfigModel):
    """Runtime-relevant summary recorded in the accepted model catalog."""

    target: TrainingTarget
    sample_dt_s: float = Field(gt=0.0)
    mappings: tuple[ResolvedTrainingMapping, ...] = Field(min_length=1)


class TrainingBenchmarkEvidence(ConfigModel):
    """One exact benchmark run attached to an accepted training model."""

    run: str
    run_manifest_sha256: str

    @field_validator("run")
    @classmethod
    def _run_path(cls, value: str) -> str:
        """Require a normalized repository-relative benchmark path."""

        return require_relative_posix_path(value)

    @field_validator("run_manifest_sha256")
    @classmethod
    def _manifest_hash(cls, value: str) -> str:
        """Require an exact lowercase benchmark manifest digest."""

        return validate_sha256(value, label="run_manifest_sha256")


class TrainingSourceReference(ConfigModel):
    """Immutable training run and optional exact benchmark evidence."""

    run: str
    run_input_sha256: str
    run_manifest_sha256: str
    sysid_state_sha256: str
    benchmarks: tuple[TrainingBenchmarkEvidence, ...] = ()

    @field_validator("run")
    @classmethod
    def _source_run(cls, value: str) -> str:
        """Require a normalized repository-relative source-run path."""

        return require_relative_posix_path(value)

    @field_validator(
        "run_input_sha256",
        "run_manifest_sha256",
        "sysid_state_sha256",
    )
    @classmethod
    def _source_hashes(cls, value: str, info: Any) -> str:
        """Require exact lowercase source identity digests."""

        return validate_sha256(value, label=info.field_name)

    @model_validator(mode="after")
    def _unique_benchmarks(self) -> "TrainingSourceReference":
        """Reject duplicate exact benchmark evidence."""

        identities = [
            (item.run, item.run_manifest_sha256)
            for item in self.benchmarks
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("benchmarks contains duplicate evidence")
        return self


class AcceptedTrainingModel(ConfigModel):
    """One complete accepted model-key entry."""

    revision: int = Field(ge=1)
    artifact: TrainingArtifactReference
    contract: AcceptedTrainingContract
    source: TrainingSourceReference


class TrainingAcceptedCatalog(ConfigModel):
    """Machine-managed accepted neural model catalog for one robot."""

    schema_version: Literal[1] = 1
    revision: int = Field(default=0, ge=0)
    models: Mapping[str, AcceptedTrainingModel] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _reject_output_overlap(self) -> "TrainingAcceptedCatalog":
        """Require accepted model keys to own disjoint physical output joints."""

        owner: dict[str, str] = {}
        for key, model in self.models.items():
            require_logical_name(key, label="model key")
            outputs = {
                joint
                for mapping in model.contract.mappings
                for joint in mapping.output_joints
            }
            for joint in sorted(outputs):
                previous = owner.get(joint)
                if previous is not None and previous != key:
                    raise ValueError(
                        f"accepted models {previous!r} and {key!r} both own output joint {joint!r}"
                    )
                owner[joint] = key
        return self
