"""Schema-version-3 runtime contract embedded in GRU TorchScript archives."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypeAlias

from pydantic import Field, field_validator, model_validator

from anchor_bench.core.config_model import ConfigModel
from anchor_bench.training.schema import SUPPORTED_FEATURES, FeatureDomain, TrainingTarget


StatValue: TypeAlias = float | tuple[float, ...]
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _names(value: Any, *, label: str) -> tuple[str, ...]:
    """Return a non-empty ordered tuple of unique strings."""

    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be an ordered list")
    result = tuple(str(item).strip() for item in value)
    if not result or any(not item for item in result):
        raise ValueError(f"{label} must contain non-empty names")
    if len(set(result)) != len(result):
        raise ValueError(f"{label} must not contain duplicates")
    return result


def _values(
    value: StatValue | None,
    width: int,
    *,
    label: str,
    positive: bool = False,
    nonnegative: bool = False,
) -> tuple[float, ...]:
    """Validate and expand one scalar/vector metadata value."""

    raw = (value,) if not isinstance(value, tuple) else value
    if len(raw) != width:
        raise ValueError(f"{label} must contain {width} value(s)")
    result = tuple(float(item) for item in raw)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{label} must contain finite values")
    if positive and not all(item > 0.0 for item in result):
        raise ValueError(f"{label} must contain positive values")
    if nonnegative and not all(item >= 0.0 for item in result):
        raise ValueError(f"{label} must contain non-negative values")
    return result


def _stat_value(value: Any) -> StatValue:
    """Parse a scalar or ordered vector without changing width-one encoding."""

    if isinstance(value, bool):
        raise ValueError("must be numeric")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result = tuple(float(item) for item in value)
        if not result:
            raise ValueError("must not be empty")
        return result
    return float(value)


class RuntimeFeature(ConfigModel):
    """One flattened runtime feature block."""

    name: str
    domain: FeatureDomain
    components: tuple[str, ...]
    width: int = Field(ge=1)

    @field_validator("name")
    @classmethod
    def _feature_name(cls, value: str) -> str:
        """Validate the runtime feature against the supported feature set."""

        name = value.strip()
        if name not in SUPPORTED_FEATURES:
            raise ValueError(f"unsupported runtime feature {name!r}")
        return name

    @field_validator("components", mode="before")
    @classmethod
    def _components(cls, value: Any) -> tuple[str, ...]:
        """Parse the feature's ordered component axis."""

        return _names(value, label="feature components")

    @model_validator(mode="after")
    def _feature_domain(self) -> "RuntimeFeature":
        """Match each feature to its input/output domain and component contract."""

        expected = "output" if self.name == "previous_torque" else "input"
        if self.domain != expected:
            raise ValueError(f"feature {self.name!r} must use domain {expected!r}")
        if self.name == "previous_torque" and self.components != ("value",):
            raise ValueError("previous_torque must contain exactly the 'value' component")
        return self

class PDBaseline(ConfigModel):
    """Ordered per-input-joint PD values frozen from the selected SysID state."""

    joint_names: tuple[str, ...]
    kp: StatValue
    kd: StatValue
    velocity_target: Literal["zero"] = "zero"

    @field_validator("joint_names", mode="before")
    @classmethod
    def _joint_names(cls, value: Any) -> tuple[str, ...]:
        """Parse the PD baseline's ordered input joints."""

        return _names(value, label="pd_baseline.joint_names")

    @field_validator("kp", "kd", mode="before")
    @classmethod
    def _stats(cls, value: Any) -> StatValue:
        """Parse scalar or vector PD baseline values."""

        return _stat_value(value)


class FrictionBaseline(ConfigModel):
    """Ordered per-output-joint friction values for a residual mapping."""

    joint_names: tuple[str, ...]
    dry: StatValue
    viscous: StatValue

    @field_validator("joint_names", mode="before")
    @classmethod
    def _joint_names(cls, value: Any) -> tuple[str, ...]:
        """Parse the friction baseline's ordered output joints."""

        return _names(value, label="friction_baseline.joint_names")

    @field_validator("dry", "viscous", mode="before")
    @classmethod
    def _stats(cls, value: Any) -> StatValue:
        """Parse scalar or vector friction baseline values."""

        return _stat_value(value)


class ArmatureValues(ConfigModel):
    """Optional ordered per-output-joint armature snapshot."""

    joint_names: tuple[str, ...]
    values: StatValue

    @field_validator("joint_names", mode="before")
    @classmethod
    def _joint_names(cls, value: Any) -> tuple[str, ...]:
        """Parse the armature snapshot's ordered output joints."""

        return _names(value, label="armature.joint_names")

    @field_validator("values", mode="before")
    @classmethod
    def _stats(cls, value: Any) -> StatValue:
        """Parse scalar or vector armature values."""

        return _stat_value(value)


class RuntimeMapping(ConfigModel):
    """One physical mapping and all mapping-specific runtime constants."""

    input_joints: tuple[str, ...]
    output_joints: tuple[str, ...]
    pd_baseline: PDBaseline | None = None
    friction_baseline: FrictionBaseline | None = None
    armature: ArmatureValues | None = None
    effort_limits: StatValue
    clamping: Literal["total_output", "residual_output"]

    @field_validator("input_joints", "output_joints", mode="before")
    @classmethod
    def _axes(cls, value: Any, info: Any) -> tuple[str, ...]:
        """Parse one physical runtime mapping axis."""

        return _names(value, label=info.field_name)

    @field_validator("effort_limits", mode="before")
    @classmethod
    def _effort_limits(cls, value: Any) -> StatValue:
        """Parse scalar or vector output effort limits."""

        return _stat_value(value)

    @model_validator(mode="after")
    def _mapping_contract(self) -> "RuntimeMapping":
        """Validate physical axes and mapping-specific runtime constants."""

        missing = [name for name in self.output_joints if name not in self.input_joints]
        if missing:
            raise ValueError("output_joints must be a subset of input_joints: " + ", ".join(missing))
        _values(self.effort_limits, len(self.output_joints), label="effort_limits", positive=True)
        if self.pd_baseline is not None:
            if self.pd_baseline.joint_names != self.input_joints:
                raise ValueError("pd_baseline.joint_names must match input_joints in order")
            _values(
                self.pd_baseline.kp,
                len(self.input_joints),
                label="pd_baseline.kp",
                nonnegative=True,
            )
            _values(
                self.pd_baseline.kd,
                len(self.input_joints),
                label="pd_baseline.kd",
                nonnegative=True,
            )
        if self.friction_baseline is not None:
            if self.friction_baseline.joint_names != self.output_joints:
                raise ValueError("friction_baseline.joint_names must match output_joints in order")
            _values(
                self.friction_baseline.dry,
                len(self.output_joints),
                label="friction_baseline.dry",
                nonnegative=True,
            )
            _values(
                self.friction_baseline.viscous,
                len(self.output_joints),
                label="friction_baseline.viscous",
                nonnegative=True,
            )
        if self.armature is not None:
            if self.armature.joint_names != self.output_joints:
                raise ValueError("armature.joint_names must match output_joints in order")
            _values(
                self.armature.values,
                len(self.output_joints),
                label="armature.values",
                nonnegative=True,
            )
        return self


class NormalizationValues(ConfigModel):
    """Named scalar/vector normalization values."""

    mean: Mapping[str, StatValue]
    std: Mapping[str, StatValue]

    @field_validator("mean", "std", mode="before")
    @classmethod
    def _parse_values(cls, value: Any) -> Mapping[str, StatValue]:
        """Parse named scalar/vector normalization values."""

        if not isinstance(value, Mapping):
            raise ValueError("must be a mapping")
        return {str(key): _stat_value(item) for key, item in value.items()}


class RuntimeNormalization(ConfigModel):
    """Training-domain input and target normalization statistics."""

    inputs: NormalizationValues
    targets: NormalizationValues


class GRURecurrentDimensions(ConfigModel):
    """Recurrent architecture needed to independently verify the archive."""

    hidden_size: int = Field(ge=1)
    num_layers: int = Field(ge=1)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)


class DelayContract(ConfigModel):
    """Learned-delay declaration; no external runtime delay is allowed."""

    handling: Literal["learned"] = "learned"
    external_delay_s: Literal[0.0] = 0.0


class PreviousTorqueDerivation(ConfigModel):
    """Autoregressive feedback semantics shared by training and runtime."""

    source: Literal["previous_target"] = "previous_target"
    runtime_source: Literal["previous_raw_network_output_pre_clamp"] = "previous_raw_network_output_pre_clamp"
    initialization: Literal["physical_zero"] = "physical_zero"
    reset_boundaries: tuple[Literal["recorded_episode", "joint_mapping"], ...] = (
        "recorded_episode",
        "joint_mapping",
    )
    value_space: Literal["physical_torque", "residual_torque"]

    @model_validator(mode="after")
    def _reset_contract(self) -> "PreviousTorqueDerivation":
        """Require resets at both recorded-episode and physical-mapping boundaries."""

        if self.reset_boundaries != ("recorded_episode", "joint_mapping"):
            raise ValueError("previous_torque reset boundaries must be recorded_episode then joint_mapping")
        return self


class DynamicBiasDerivation(ConfigModel):
    """Provider identity and reproducibility details for MuJoCo qfrc_bias."""

    source: Literal["newton.solvers.SolverMuJoCo.eval_dynamic_bias"]
    provenance: Mapping[str, Any]


class RuntimeHashes(ConfigModel):
    """Hashes binding the archive to its resolved training inputs."""

    profile_sha256: str
    resolved_config_sha256: str
    sysid_state_sha256: str
    asset_sha256: str
    asset_package_sha256: str
    dataset_sha256: str
    source_sha256: str

    @field_validator(
        "profile_sha256",
        "resolved_config_sha256",
        "sysid_state_sha256",
        "asset_sha256",
        "asset_package_sha256",
        "dataset_sha256",
        "source_sha256",
    )
    @classmethod
    def _hash(cls, value: str) -> str:
        """Validate one lowercase runtime identity digest."""

        result = value.strip().lower()
        if not _SHA256.fullmatch(result):
            raise ValueError("must be a lowercase SHA-256 digest")
        return result


def _validate_runtime_header(metadata: "RuntimeGRUMetadata") -> TrainingTarget:
    """Validate fixed runtime layout fields and return the target semantic."""

    if not math.isfinite(metadata.sample_dt_s):
        raise ValueError("sample_dt_s must be finite")
    if metadata.flattening_order != ("feature", "joint", "component"):
        raise ValueError("flattening_order must be feature, joint, component")
    if len(metadata.target_columns) != 1:
        raise ValueError("target_columns must contain one torque semantic")
    return metadata.target_columns[0]


def _validate_runtime_mappings(
    metadata: "RuntimeGRUMetadata",
    target: TrainingTarget,
) -> tuple[int, int]:
    """Validate mapping compatibility and return one mapping's axis widths."""

    first = metadata.mappings[0]
    input_width = len(first.input_joints)
    output_width = len(first.output_joints)
    seen_mappings: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
    owned_outputs: set[str] = set()
    needs_pd = target == "torque_residual" or any(
        feature.name == "solver_pd" for feature in metadata.input_features
    )
    expected_clamp = "residual_output" if target == "torque_residual" else "total_output"
    for mapping in metadata.mappings:
        if (len(mapping.input_joints), len(mapping.output_joints)) != (input_width, output_width):
            raise ValueError("all mappings must have identical input and output widths")
        identity = (mapping.input_joints, mapping.output_joints)
        if identity in seen_mappings:
            raise ValueError("mappings must not contain duplicates")
        seen_mappings.add(identity)
        overlap = owned_outputs.intersection(mapping.output_joints)
        if overlap:
            raise ValueError("mappings contain duplicate physical outputs: " + ", ".join(sorted(overlap)))
        owned_outputs.update(mapping.output_joints)
        if needs_pd and mapping.pd_baseline is None:
            raise ValueError("solver_pd and residual targets require a mapping PD baseline")
        if target == "torque_residual" and mapping.friction_baseline is None:
            raise ValueError("residual targets require a mapping friction baseline")
        if mapping.clamping != expected_clamp:
            raise ValueError(f"target {target!r} requires clamping {expected_clamp!r}")
    return input_width, output_width


def _validate_runtime_features(
    metadata: "RuntimeGRUMetadata",
    *,
    input_width: int,
    output_width: int,
) -> tuple[str, ...]:
    """Validate generic feature blocks and flattened tensor dimensions."""

    feature_names = tuple(feature.name for feature in metadata.input_features)
    if len(set(feature_names)) != len(feature_names):
        raise ValueError("input_features must not contain duplicate names")
    expected_input_size = 0
    for feature in metadata.input_features:
        axis = output_width if feature.domain == "output" else input_width
        expected_width = axis * len(feature.components)
        if feature.width != expected_width:
            raise ValueError(f"feature {feature.name!r} width is {feature.width}, expected {expected_width}")
        expected_input_size += expected_width
    if metadata.input_size != expected_input_size:
        raise ValueError(f"input_size is {metadata.input_size}, expected {expected_input_size}")
    if metadata.output_size != output_width:
        raise ValueError(f"output_size is {metadata.output_size}, expected {output_width}")
    return feature_names


def _validate_runtime_normalization(
    metadata: "RuntimeGRUMetadata",
    *,
    feature_names: tuple[str, ...],
    target: TrainingTarget,
    output_width: int,
) -> None:
    """Validate input-block and output-target normalization statistics."""

    input_mean = metadata.normalization.inputs.mean
    input_std = metadata.normalization.inputs.std
    if set(input_mean) != set(feature_names) or set(input_std) != set(feature_names):
        raise ValueError("input normalization keys must exactly match input_features")
    for feature in metadata.input_features:
        _values(input_mean[feature.name], feature.width, label=f"normalization.inputs.mean.{feature.name}")
        _values(
            input_std[feature.name],
            feature.width,
            label=f"normalization.inputs.std.{feature.name}",
            positive=True,
        )

    if set(metadata.normalization.targets.mean) != {target} or set(metadata.normalization.targets.std) != {target}:
        raise ValueError("target normalization must contain exactly the target semantic")
    _values(metadata.normalization.targets.mean[target], output_width, label=f"normalization.targets.mean.{target}")
    _values(
        metadata.normalization.targets.std[target],
        output_width,
        label=f"normalization.targets.std.{target}",
        positive=True,
    )


def _validate_runtime_derivations(
    metadata: "RuntimeGRUMetadata",
    *,
    feature_names: tuple[str, ...],
    target: TrainingTarget,
) -> None:
    """Validate feature-dependent previous-torque and dynamic-bias declarations."""

    has_previous = "previous_torque" in feature_names
    if has_previous != (metadata.previous_torque_derivation is not None):
        raise ValueError("previous_torque feature and derivation must be configured together")
    if metadata.previous_torque_derivation is not None:
        expected_space = "residual_torque" if target == "torque_residual" else "physical_torque"
        if metadata.previous_torque_derivation.value_space != expected_space:
            raise ValueError(f"previous_torque value_space must be {expected_space!r}")

    has_bias = "dynamic_bias" in feature_names
    if has_bias != (metadata.dynamic_bias_derivation is not None):
        raise ValueError("dynamic_bias feature and derivation must be configured together")


class RuntimeGRUMetadata(ConfigModel):
    """Complete schema-v3 contract consumed without auxiliary files."""

    schema_version: Literal[3] = 3
    model_type: Literal["gru"] = "gru"
    runtime_dtype: Literal["float32"] = "float32"
    input_size: int = Field(ge=1)
    output_size: int = Field(ge=1)
    recurrent: GRURecurrentDimensions
    input_features: tuple[RuntimeFeature, ...] = Field(min_length=1)
    flattening_order: tuple[str, ...] = ("feature", "joint", "component")
    mappings: tuple[RuntimeMapping, ...] = Field(min_length=1)
    target_columns: tuple[TrainingTarget, ...]
    sample_dt_s: float = Field(gt=0.0)
    normalization: RuntimeNormalization
    delay: DelayContract = Field(default_factory=DelayContract)
    previous_torque_derivation: PreviousTorqueDerivation | None = None
    dynamic_bias_derivation: DynamicBiasDerivation | None = None
    hashes: RuntimeHashes

    @model_validator(mode="after")
    def _runtime_contract(self) -> "RuntimeGRUMetadata":
        """Validate tensor dimensions, mappings, statistics, targets, and derivations."""

        target = _validate_runtime_header(self)
        input_width, output_width = _validate_runtime_mappings(self, target)
        feature_names = _validate_runtime_features(
            self,
            input_width=input_width,
            output_width=output_width,
        )
        _validate_runtime_normalization(
            self,
            feature_names=feature_names,
            target=target,
            output_width=output_width,
        )
        _validate_runtime_derivations(self, feature_names=feature_names, target=target)
        return self
