"""Time-aligned episode assembly for coupled actuator-network training."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from anchor_bench.data.motion import MotionData
from anchor_bench.training.dynamic_bias import DynamicBiasProvider
from anchor_bench.training.schema import (
    ResolvedTrainingMapping,
    TrainingFeature,
    TrainingTarget,
)


CADENCE_MEDIAN_REL_TOL = 1.0e-2
CADENCE_INTERVAL_REL_TOL = 0.25
_SCALAR_RECORDED_FEATURES = {"position", "position_error", "velocity", "dynamic_bias"}


def _readonly(values: Any, *, dtype: Any, ndim: int, label: str) -> np.ndarray:
    """Return a finite immutable array with a fixed rank."""

    array = np.asarray(values, dtype=dtype)
    if array.ndim != ndim:
        raise ValueError(f"{label} must be a {ndim}-D array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} contains non-finite values")
    result = np.array(array, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class FeatureBlock:
    """One time-by-joint-by-component feature block."""

    values: np.ndarray
    domain: str
    components: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate the block's rank, domain, components, and finite values."""

        values = _readonly(self.values, dtype=np.float32, ndim=3, label="feature values")
        if self.domain not in {"input", "output"}:
            raise ValueError("feature domain must be 'input' or 'output'")
        components = tuple(str(component).strip() for component in self.components)
        if not components or any(not component for component in components):
            raise ValueError("feature components must contain non-empty names")
        if len(set(components)) != len(components):
            raise ValueError("feature components must not contain duplicates")
        if values.shape[2] != len(components):
            raise ValueError("feature component axis does not match declared components")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "components", components)


@dataclass(frozen=True)
class TrainingEpisode:
    """One recurrent sequence for one recorded episode and one mapping."""

    timestamps: np.ndarray
    mapping: ResolvedTrainingMapping
    feature_blocks: Mapping[str, FeatureBlock]
    measured_torque: np.ndarray
    dynamic_bias_provenance: Mapping[str, Any] | None = field(default=None)

    def __post_init__(self) -> None:
        """Validate episode shapes and freeze its aligned recurrent sequence."""

        timestamps = _readonly(self.timestamps, dtype=np.float64, ndim=1, label="episode timestamps")
        torque = _readonly(self.measured_torque, dtype=np.float32, ndim=2, label="measured torque")
        if not len(timestamps):
            raise ValueError("episode must contain at least one row")
        if torque.shape != (len(timestamps), len(self.mapping.output_joints)):
            raise ValueError("measured torque shape must be [time, output_joints]")

        blocks: dict[str, FeatureBlock] = {}
        for name, block in self.feature_blocks.items():
            if len(block.values) != len(timestamps):
                raise ValueError(f"feature {name!r} must contain one row per timestamp")
            axis_width = len(
                self.mapping.output_joints if block.domain == "output" else self.mapping.input_joints
            )
            if block.values.shape[1] != axis_width:
                raise ValueError(f"feature {name!r} joint axis has the wrong width")
            blocks[name] = block

        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "measured_torque", torque)
        object.__setattr__(self, "feature_blocks", MappingProxyType(blocks))
        if self.dynamic_bias_provenance is not None:
            object.__setattr__(
                self,
                "dynamic_bias_provenance",
                MappingProxyType(dict(self.dynamic_bias_provenance)),
            )


@dataclass(frozen=True)
class PreparedEpisodeSource:
    """Full recorded signals aligned once onto the authoritative state grid."""

    motion: MotionData
    timestamps: np.ndarray
    position: np.ndarray
    command: np.ndarray
    velocity: np.ndarray | None
    effort: np.ndarray


def validate_recorded_cadence(timestamps: np.ndarray, sample_dt_s: float, *, source: Path) -> None:
    """Fail when recorded state cadence is incompatible with the configured control step."""

    if not math.isfinite(sample_dt_s) or sample_dt_s <= 0.0:
        raise ValueError("sample_dt_s must be finite and positive")
    if len(timestamps) < 2:
        return
    intervals = np.diff(timestamps)
    median = float(np.median(intervals))
    if not math.isclose(median, sample_dt_s, rel_tol=CADENCE_MEDIAN_REL_TOL):
        raise ValueError(
            f"{source} median state cadence {median} does not match sample_dt_s {sample_dt_s}"
        )
    mismatched = ~np.isclose(intervals, sample_dt_s, rtol=CADENCE_INTERVAL_REL_TOL, atol=0.0)
    if np.any(mismatched):
        index = int(np.flatnonzero(mismatched)[0])
        raise ValueError(
            f"{source} state interval {index}->{index + 1} is {float(intervals[index])}, "
            f"outside {CADENCE_INTERVAL_REL_TOL:.0%} of sample_dt_s {sample_dt_s}"
        )

def prepare_episode_source(motion: MotionData, sample_dt_s: float) -> PreparedEpisodeSource:
    """Align command, velocity, and effort without resampling the state trajectory."""

    timestamps = np.asarray(motion.position.time, dtype=np.float64)
    if not len(timestamps):
        raise ValueError(f"training source has no state rows: {motion.source}")
    validate_recorded_cadence(timestamps, sample_dt_s, source=motion.source)
    if motion.effort is None:
        raise ValueError(f"{motion.source} is missing required measured torque")
    return PreparedEpisodeSource(
        motion=motion,
        timestamps=timestamps,
        position=np.asarray(motion.position.values, dtype=np.float32),
        command=np.asarray(motion.command_position.interpolate(timestamps).values, dtype=np.float32),
        velocity=(
            None
            if motion.velocity is None
            else np.asarray(motion.velocity.interpolate(timestamps).values, dtype=np.float32)
        ),
        effort=np.asarray(motion.effort.interpolate(timestamps).values, dtype=np.float32),
    )


class EpisodeAssembler:
    """Per-run source cache and mapping-specific episode assembler."""

    def __init__(
        self,
        dynamic_bias_provider: DynamicBiasProvider | None = None,
        *,
        canonical_joint_names: Sequence[str] | None = None,
        joint_aliases: Mapping[str, str] | None = None,
    ) -> None:
        """Create an assembler with an optional injectable dynamic-bias provider."""

        if canonical_joint_names is None and joint_aliases:
            raise ValueError("joint_aliases requires canonical_joint_names")
        self._dynamic_bias_provider = dynamic_bias_provider
        self._canonical_joint_names = (
            None if canonical_joint_names is None else tuple(canonical_joint_names)
        )
        self._joint_aliases = dict(joint_aliases or {})
        self._prepared: dict[tuple[Path, float], PreparedEpisodeSource] = {}

    def load(self, path: str | Path, sample_dt_s: float) -> PreparedEpisodeSource:
        """Load one SAGE source once for this training run."""

        source = Path(path)
        key = (source, float(sample_dt_s))
        if key not in self._prepared:
            from anchor_bench.data.sage import load_sage_motion

            motion = load_sage_motion(
                source,
                canonical_joint_names=self._canonical_joint_names,
                joint_aliases=self._joint_aliases,
            )
            self._prepared[key] = prepare_episode_source(motion, sample_dt_s)
        return self._prepared[key]

    def assemble(
        self,
        source: PreparedEpisodeSource,
        *,
        mapping: ResolvedTrainingMapping,
        features: Sequence[TrainingFeature],
        target: TrainingTarget,
        dynamic_bias_cache_key: str | None = None,
    ) -> TrainingEpisode:
        """Build one independent recurrent sequence in configured joint order."""

        motion = source.motion
        input_indices = motion.joints.indices(mapping.input_joints)
        output_indices = motion.joints.indices(mapping.output_joints)
        position = source.position[:, input_indices]
        command = source.command[:, input_indices]
        velocity = None if source.velocity is None else source.velocity[:, input_indices]

        requested_names = [feature.name for feature in features]
        needs_velocity = target == "torque_residual" or bool(
            {"velocity", "solver_pd", "dynamic_bias"}.intersection(requested_names)
        )
        if needs_velocity and velocity is None:
            raise ValueError(f"{motion.source} is missing required measured velocity")

        scalar_values: dict[str, np.ndarray] = {
            "position": position,
            "position_error": command - position,
        }
        if velocity is not None:
            scalar_values["velocity"] = velocity

        provenance: Mapping[str, Any] | None = None
        if "dynamic_bias" in requested_names:
            if self._dynamic_bias_provider is None:
                raise ValueError("dynamic_bias training requires a DynamicBiasProvider")
            assert source.velocity is not None
            result = self._dynamic_bias_provider.compute(
                joint_names=motion.joints.names,
                positions=source.position,
                velocities=source.velocity,
                requested_joint_names=mapping.input_joints,
                cache_key=dynamic_bias_cache_key,
            )
            values = np.asarray(result.values, dtype=np.float32)
            expected = (len(source.timestamps), len(mapping.input_joints))
            if values.shape != expected:
                raise ValueError(f"dynamic-bias provider returned shape {values.shape}, expected {expected}")
            scalar_values["dynamic_bias"] = values
            provenance = result.provenance

        blocks: dict[str, FeatureBlock] = {}
        for feature in features:
            if feature.name in {"solver_pd", "previous_torque"}:
                continue
            if feature.name not in _SCALAR_RECORDED_FEATURES:
                raise ValueError(f"no recorded provider exists for feature {feature.name!r}")
            if feature.components != ("value",):
                raise ValueError(
                    f"recorded feature {feature.name!r} currently provides only the 'value' component"
                )
            blocks[feature.name] = FeatureBlock(
                scalar_values[feature.name][:, :, np.newaxis],
                feature.domain,
                feature.components,
            )

        # Residual target construction is a trainer concern, but its source
        # physics signals must remain available even when they are not inputs.
        if target == "torque_residual" or "solver_pd" in requested_names:
            for name, values in (
                ("position_error", scalar_values["position_error"]),
                ("velocity", scalar_values["velocity"]),
            ):
                blocks.setdefault(name, FeatureBlock(values[:, :, np.newaxis], "input", ("value",)))

        return TrainingEpisode(
            timestamps=source.timestamps,
            mapping=mapping,
            feature_blocks=blocks,
            measured_torque=source.effort[:, output_indices],
            dynamic_bias_provenance=provenance,
        )
