"""Strict, host-safe deployment profile and frozen-state schemas."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal, Self, TypeAlias

from pydantic import Field, FiniteFloat, field_validator, model_validator

from anchor_bench.core.config_model import ConfigModel
from anchor_bench.core.hashing import require_relative_posix_path, validate_sha256
from anchor_bench.core.profiles import require_logical_name


NeuralTarget = Literal["torque", "torque_residual"]
ClampSemantic = Literal["total_output", "residual_output"]
JointValueMap: TypeAlias = dict[str, FiniteFloat]
_SELECTION_KEYWORDS = frozenset({"all", "remaining", "exclude"})


def _logical_name(value: str, *, label: str) -> str:
    """Validate and return a deployment logical name."""

    return require_logical_name(value, label=label)


def _selector(value: str) -> bool:
    """Return whether text is a forbidden implicit joint selector."""

    return value.strip().lower() in _SELECTION_KEYWORDS


def _ordered_joints(value: Any, *, label: str) -> tuple[str, ...]:
    """Parse an explicit ordered list of unique physical joints."""

    if isinstance(value, str):
        if _selector(value):
            raise ValueError(
                f"{label} must be explicit; selection keyword {value.strip()!r} is not allowed"
            )
        raise ValueError(f"{label} must be an explicit ordered list")
    if not isinstance(value, Sequence):
        raise ValueError(f"{label} must be an explicit ordered list")
    joints = tuple(value)
    if not joints:
        raise ValueError(f"{label} must contain at least one joint")
    if any(not isinstance(joint, str) or not joint or joint != joint.strip() for joint in joints):
        raise ValueError(f"{label} must contain non-empty joint names with no surrounding whitespace")
    selectors = sorted({joint for joint in joints if _selector(joint)})
    if selectors:
        raise ValueError(
            f"{label} must be explicit; selection keyword(s) {', '.join(selectors)} are not allowed"
        )
    duplicates = sorted({joint for index, joint in enumerate(joints) if joint in joints[:index]})
    if duplicates:
        raise ValueError(f"duplicate joints in {label}: {', '.join(duplicates)}")
    return joints


def _joint_map(value: Any, *, label: str) -> Any:
    """Validate the keys and primitive values of a per-joint mapping."""

    if not isinstance(value, Mapping):
        return value
    if not value:
        raise ValueError(f"{label} must contain at least one joint value")
    for joint, item in value.items():
        if not isinstance(joint, str) or not joint or joint != joint.strip():
            raise ValueError(f"{label} keys must be non-empty joint names")
        if _selector(joint):
            raise ValueError(
                f"{label} must be explicit; selection keyword {joint!r} is not allowed"
            )
        if isinstance(item, bool):
            raise ValueError(f"{label}.{joint} must be numeric")
    return value


def _require_exact_keys(values: Mapping[str, float], joints: Sequence[str], *, label: str) -> None:
    """Require a value map to cover exactly the selected joints."""

    expected = set(joints)
    actual = set(values)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append("extra " + ", ".join(extra))
        raise ValueError(f"{label} keys must exactly match joints ({'; '.join(details)})")


def _require_positive(values: Mapping[str, float], *, label: str) -> None:
    """Require every value in a per-joint map to be positive."""

    invalid = sorted(joint for joint, value in values.items() if value <= 0.0)
    if invalid:
        raise ValueError(f"{label} must contain positive values for: {', '.join(invalid)}")


class DeploySimulationConfig(ConfigModel):
    """Complete simulation settings frozen by deployment promotion."""

    physics_freq: int = Field(ge=1)
    control_freq: int = Field(ge=1)
    render_freq: int = Field(ge=1)
    integrator: str
    solver_iterations: int = Field(ge=1)
    line_search_iterations: int = Field(ge=1)

    @field_validator("integrator")
    @classmethod
    def _integrator(cls, value: str) -> str:
        """Validate the deployment integrator name."""

        if not value or value != value.strip():
            raise ValueError("integrator must be non-empty and have no surrounding whitespace")
        return value


class DeployNeuralMapping(ConfigModel):
    """Exact accepted-model mapping selected by a deployment profile."""

    index: int = Field(ge=0)
    input_joints: tuple[str, ...]
    output_joints: tuple[str, ...]

    @field_validator("input_joints", "output_joints", mode="before")
    @classmethod
    def _axes(cls, value: Any, info: Any) -> tuple[str, ...]:
        """Parse an explicit neural mapping axis."""

        return _ordered_joints(value, label=info.field_name)

    @model_validator(mode="after")
    def _output_subset(self) -> Self:
        """Require each neural output joint to also be available as an input."""

        missing = [joint for joint in self.output_joints if joint not in self.input_joints]
        if missing:
            raise ValueError("output_joints must be a subset of input_joints: " + ", ".join(missing))
        return self


class DeployNeuralActuator(ConfigModel):
    """Named neural actuator referencing one accepted model key."""

    name: str
    model: str
    mappings: tuple[DeployNeuralMapping, ...] = Field(min_length=1)

    @field_validator("name", "model")
    @classmethod
    def _names(cls, value: str, info: Any) -> str:
        """Validate neural actuator and accepted-model logical names."""

        return _logical_name(value, label=info.field_name)

    @model_validator(mode="after")
    def _unique_mappings(self) -> Self:
        """Reject duplicate accepted-model indices and duplicate physical axes."""

        indices = [mapping.index for mapping in self.mappings]
        if len(indices) != len(set(indices)):
            raise ValueError("mappings contain duplicate accepted-model indices")
        axes = [(mapping.input_joints, mapping.output_joints) for mapping in self.mappings]
        if len(axes) != len(set(axes)):
            raise ValueError("mappings contain duplicate joint axes")
        return self


class DeployPDActuator(ConfigModel):
    """Named explicit PD owner and its profile-selected total-effort clamps."""

    name: str
    joints: tuple[str, ...]
    effort_limits: JointValueMap

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        """Validate the explicit PD actuator name."""

        return _logical_name(value, label="PD actuator name")

    @field_validator("joints", mode="before")
    @classmethod
    def _joints(cls, value: Any) -> tuple[str, ...]:
        """Parse the joints owned by an explicit PD actuator."""

        return _ordered_joints(value, label="joints")

    @field_validator("effort_limits", mode="before")
    @classmethod
    def _effort_limit_input(cls, value: Any) -> Any:
        """Validate the raw per-joint effort-limit mapping."""

        return _joint_map(value, label="effort_limits")

    @model_validator(mode="after")
    def _effort_limit_contract(self) -> Self:
        """Require one positive effort limit for every PD-owned joint."""

        _require_exact_keys(self.effort_limits, self.joints, label="effort_limits")
        _require_positive(self.effort_limits, label="effort_limits")
        return self


class DeployComposition(ConfigModel):
    """Complete explicit actuator composition requested by a profile."""

    neural: tuple[DeployNeuralActuator, ...] = ()
    pd: tuple[DeployPDActuator, ...] = ()

    @model_validator(mode="after")
    def _composition_contract(self) -> Self:
        """Require unique actuator names, references, and output ownership."""

        if not self.neural and not self.pd:
            raise ValueError("composition must contain at least one neural or PD actuator")
        names = [entry.name for entry in (*self.neural, *self.pd)]
        if len(names) != len(set(names)):
            raise ValueError("composition contains duplicate actuator names")

        references: set[tuple[str, int]] = set()
        owners: dict[str, str] = {}
        for entry in self.neural:
            for mapping in entry.mappings:
                reference = (entry.model, mapping.index)
                if reference in references:
                    raise ValueError(
                        f"duplicate neural mapping reference {entry.model!r} index {mapping.index}"
                    )
                references.add(reference)
                _claim_outputs(owners, mapping.output_joints, entry.name)
        for entry in self.pd:
            _claim_outputs(owners, entry.joints, entry.name)
        return self


class DeployProfile(ConfigModel):
    """Human-authored explicit composition resolved only from accepted state."""

    schema_version: Literal[1]
    simulation: DeploySimulationConfig
    composition: DeployComposition


class FrozenNeuralArtifact(ConfigModel):
    """Immutable accepted neural archive and authenticated metadata identity."""

    path: str
    sha256: str
    metadata_sha256: str

    @field_validator("path")
    @classmethod
    def _path(cls, value: str) -> str:
        """Require a neural archive path inside immutable training outputs."""

        normalized = require_relative_posix_path(value)
        if not normalized.startswith("outputs/train/"):
            raise ValueError("neural artifact path must be under outputs/train")
        return normalized

    @field_validator("sha256", "metadata_sha256")
    @classmethod
    def _hash(cls, value: str, info: Any) -> str:
        """Validate neural archive and embedded-metadata digests."""

        return validate_sha256(value, label=info.field_name)


class FrozenNeuralPDBaseline(ConfigModel):
    """Per-input-joint PD baseline copied from accepted neural metadata."""

    kp: JointValueMap
    kd: JointValueMap
    velocity_target: Literal["zero"] = "zero"

    @field_validator("kp", "kd", mode="before")
    @classmethod
    def _maps(cls, value: Any, info: Any) -> Any:
        """Validate a per-input-joint PD baseline map."""

        return _joint_map(value, label=info.field_name)

    @model_validator(mode="after")
    def _matching_axes(self) -> Self:
        """Require stiffness and damping baselines to share the same axes."""

        if set(self.kp) != set(self.kd):
            raise ValueError("PD baseline kp and kd keys must match")
        return self


class FrozenNeuralFrictionBaseline(ConfigModel):
    """Per-output-joint friction baseline copied from neural metadata."""

    dry: JointValueMap
    viscous: JointValueMap

    @field_validator("dry", "viscous", mode="before")
    @classmethod
    def _maps(cls, value: Any, info: Any) -> Any:
        """Validate a per-output-joint friction baseline map."""

        return _joint_map(value, label=info.field_name)

    @model_validator(mode="after")
    def _matching_axes(self) -> Self:
        """Require dry and viscous friction baselines to share the same axes."""

        if set(self.dry) != set(self.viscous):
            raise ValueError("friction baseline dry and viscous keys must match")
        return self


class FrozenNeuralMapping(ConfigModel):
    """One resolved neural mapping with all mapping-specific physical values."""

    index: int = Field(ge=0)
    input_joints: tuple[str, ...]
    output_joints: tuple[str, ...]
    pd_baseline: FrozenNeuralPDBaseline | None = None
    friction_baseline: FrozenNeuralFrictionBaseline | None = None
    armature: JointValueMap | None = None
    effort_limits: JointValueMap
    clamping: ClampSemantic

    @field_validator("input_joints", "output_joints", mode="before")
    @classmethod
    def _axes(cls, value: Any, info: Any) -> tuple[str, ...]:
        """Parse one frozen neural mapping axis."""

        return _ordered_joints(value, label=info.field_name)

    @field_validator("armature", "effort_limits", mode="before")
    @classmethod
    def _joint_maps(cls, value: Any, info: Any) -> Any:
        """Validate optional armature and required effort-limit maps."""

        if value is None:
            return None
        return _joint_map(value, label=info.field_name)

    @model_validator(mode="after")
    def _mapping_contract(self) -> Self:
        """Validate mapping axes and exact per-joint physical parameter coverage."""

        missing = [joint for joint in self.output_joints if joint not in self.input_joints]
        if missing:
            raise ValueError("output_joints must be a subset of input_joints: " + ", ".join(missing))
        _require_exact_keys(self.effort_limits, self.output_joints, label="effort_limits")
        _require_positive(self.effort_limits, label="effort_limits")
        if self.pd_baseline is not None:
            _require_exact_keys(self.pd_baseline.kp, self.input_joints, label="pd_baseline.kp")
            _require_exact_keys(self.pd_baseline.kd, self.input_joints, label="pd_baseline.kd")
        if self.friction_baseline is not None:
            _require_exact_keys(
                self.friction_baseline.dry,
                self.output_joints,
                label="friction_baseline.dry",
            )
            _require_exact_keys(
                self.friction_baseline.viscous,
                self.output_joints,
                label="friction_baseline.viscous",
            )
        if self.armature is not None:
            _require_exact_keys(self.armature, self.output_joints, label="armature")
        return self


class FrozenNeuralActuator(ConfigModel):
    """Accepted neural model revision and resolved mappings used at runtime."""

    name: str
    model: str
    accepted_model_revision: int = Field(ge=1)
    artifact: FrozenNeuralArtifact
    target: NeuralTarget
    sample_dt_s: FiniteFloat = Field(gt=0.0)
    mappings: tuple[FrozenNeuralMapping, ...] = Field(min_length=1)

    @field_validator("name", "model")
    @classmethod
    def _names(cls, value: str, info: Any) -> str:
        """Validate frozen neural actuator and accepted-model names."""

        return _logical_name(value, label=info.field_name)

    @model_validator(mode="after")
    def _model_contract(self) -> Self:
        """Match each frozen mapping's baselines and clamp to the model target."""

        indices = [mapping.index for mapping in self.mappings]
        if len(indices) != len(set(indices)):
            raise ValueError("mappings contain duplicate accepted-model indices")
        expected_clamp = "residual_output" if self.target == "torque_residual" else "total_output"
        for mapping in self.mappings:
            if mapping.clamping != expected_clamp:
                raise ValueError(f"target {self.target!r} requires clamping {expected_clamp!r}")
            if self.target == "torque_residual":
                if mapping.pd_baseline is None:
                    raise ValueError("residual targets require a mapping PD baseline")
                if mapping.friction_baseline is None:
                    raise ValueError("residual targets require a mapping friction baseline")
        return self


class FrozenPDActuator(ConfigModel):
    """Resolved explicit PD values and total-effort clamps for owned joints."""

    name: str
    joints: tuple[str, ...]
    kp: JointValueMap
    kd: JointValueMap
    dry_friction: JointValueMap
    viscous_friction: JointValueMap
    armature: JointValueMap | None = None
    effort_limits: JointValueMap

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        """Validate the frozen PD actuator name."""

        return _logical_name(value, label="PD actuator name")

    @field_validator("joints", mode="before")
    @classmethod
    def _joints(cls, value: Any) -> tuple[str, ...]:
        """Parse the frozen PD actuator's owned joints."""

        return _ordered_joints(value, label="joints")

    @field_validator(
        "kp",
        "kd",
        "dry_friction",
        "viscous_friction",
        "armature",
        "effort_limits",
        mode="before",
    )
    @classmethod
    def _maps(cls, value: Any, info: Any) -> Any:
        """Validate each frozen per-joint PD parameter map."""

        if value is None:
            return None
        return _joint_map(value, label=info.field_name)

    @model_validator(mode="after")
    def _resolved_values(self) -> Self:
        """Require complete physical values and positive clamps for every owned joint."""

        maps = {
            "kp": self.kp,
            "kd": self.kd,
            "dry_friction": self.dry_friction,
            "viscous_friction": self.viscous_friction,
            "effort_limits": self.effort_limits,
        }
        if self.armature is not None:
            maps["armature"] = self.armature
        for label, values in maps.items():
            _require_exact_keys(values, self.joints, label=label)
        _require_positive(self.effort_limits, label="effort_limits")
        return self


class FrozenDeployComposition(ConfigModel):
    """Machine-resolved composition with disjoint actuator output ownership."""

    neural: tuple[FrozenNeuralActuator, ...] = ()
    pd: tuple[FrozenPDActuator, ...] = ()

    @model_validator(mode="after")
    def _composition_contract(self) -> Self:
        """Require unique frozen actuator names, references, and output owners."""

        if not self.neural and not self.pd:
            raise ValueError("composition must contain at least one neural or PD actuator")
        names = [entry.name for entry in (*self.neural, *self.pd)]
        if len(names) != len(set(names)):
            raise ValueError("composition contains duplicate actuator names")

        references: set[tuple[str, int]] = set()
        owners: dict[str, str] = {}
        for entry in self.neural:
            for mapping in entry.mappings:
                reference = (entry.model, mapping.index)
                if reference in references:
                    raise ValueError(
                        f"duplicate neural mapping reference {entry.model!r} index {mapping.index}"
                    )
                references.add(reference)
                _claim_outputs(owners, mapping.output_joints, entry.name)
        for entry in self.pd:
            _claim_outputs(owners, entry.joints, entry.name)
        return self

    def output_owners(self) -> dict[str, str]:
        """Return the validated joint-to-actuator ownership map."""

        owners: dict[str, str] = {}
        for entry in self.neural:
            for mapping in entry.mappings:
                owners.update(dict.fromkeys(mapping.output_joints, entry.name))
        for entry in self.pd:
            owners.update(dict.fromkeys(entry.joints, entry.name))
        return owners


class DeploySourceProvenance(ConfigModel):
    """Exact profile and accepted-state revisions used for materialization."""

    profile: str
    profile_sha256: str
    sysid_revision: int = Field(ge=0)
    sysid_state_sha256: str
    train_revision: int = Field(ge=0)
    train_state_sha256: str

    @field_validator("profile")
    @classmethod
    def _profile(cls, value: str) -> str:
        """Validate the source deployment profile name."""

        return _logical_name(value, label="profile name")

    @field_validator("profile_sha256", "sysid_state_sha256", "train_state_sha256")
    @classmethod
    def _hash(cls, value: str, info: Any) -> str:
        """Validate deployment source-state and profile digests."""

        return validate_sha256(value, label=info.field_name)


class FrozenDeployState(ConfigModel):
    """Complete machine-generated runtime composition in ``deploy/default.yaml``."""

    schema_version: Literal[1]
    simulation: DeploySimulationConfig
    ownership: dict[str, str]
    composition: FrozenDeployComposition
    source: DeploySourceProvenance

    @field_validator("ownership", mode="before")
    @classmethod
    def _ownership_input(cls, value: Any) -> Any:
        """Validate the explicit frozen joint-to-actuator ownership map."""

        if not isinstance(value, Mapping):
            return value
        if not value:
            raise ValueError("ownership must contain at least one output joint")
        for joint, owner in value.items():
            if not isinstance(joint, str) or not joint or joint != joint.strip():
                raise ValueError("ownership keys must be non-empty joint names")
            if _selector(joint):
                raise ValueError(f"ownership selection keyword {joint!r} is not allowed")
            if not isinstance(owner, str):
                raise ValueError(f"ownership.{joint} must name an actuator")
            _logical_name(owner, label=f"ownership owner for {joint}")
        return value

    @model_validator(mode="after")
    def _frozen_contract(self) -> Self:
        """Match frozen ownership and neural cadence to the resolved composition."""

        expected_owners = self.composition.output_owners()
        if self.ownership != expected_owners:
            raise ValueError("ownership must exactly match resolved composition outputs")
        expected_dt = 1.0 / self.simulation.control_freq
        for entry in self.composition.neural:
            if not math.isclose(entry.sample_dt_s, expected_dt, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(
                    f"neural actuator {entry.name!r} sample_dt_s must equal "
                    f"1 / control_freq ({expected_dt})"
                )
        return self


def _claim_outputs(owners: dict[str, str], joints: Sequence[str], owner: str) -> None:
    """Assign output joints to one actuator while rejecting ownership overlap."""

    for joint in joints:
        previous = owners.get(joint)
        if previous is not None:
            raise ValueError(
                f"output joint {joint!r} is owned by both {previous!r} and {owner!r}"
            )
        owners[joint] = owner


__all__ = [
    "ClampSemantic",
    "DeployComposition",
    "DeployNeuralActuator",
    "DeployNeuralMapping",
    "DeployPDActuator",
    "DeployProfile",
    "DeploySimulationConfig",
    "DeploySourceProvenance",
    "FrozenDeployComposition",
    "FrozenDeployState",
    "FrozenNeuralActuator",
    "FrozenNeuralArtifact",
    "FrozenNeuralFrictionBaseline",
    "FrozenNeuralMapping",
    "FrozenNeuralPDBaseline",
    "FrozenPDActuator",
    "JointValueMap",
    "NeuralTarget",
]
