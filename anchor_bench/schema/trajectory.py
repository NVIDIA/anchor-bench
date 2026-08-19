# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic models for generic piecewise actuator trajectories."""

from __future__ import annotations

from typing import Annotated, Dict, List, Literal, Union

from pydantic import Field, StrictFloat, StrictInt, field_validator, model_validator
from typing_extensions import TypeAlias

from anchor_bench.schema.base import BaseConfig, StrictBaseModel

JsonNumber: TypeAlias = Union[StrictInt, StrictFloat]
Identifier: TypeAlias = Annotated[
    str,
    Field(min_length=1, pattern=r"^[A-Za-z][A-Za-z0-9_-]*$"),
]
PositiveInteger: TypeAlias = Annotated[StrictInt, Field(ge=1)]
PositiveNumber: TypeAlias = Annotated[JsonNumber, Field(gt=0.0)]
NonNegativeNumber: TypeAlias = Annotated[JsonNumber, Field(ge=0.0)]
Pose: TypeAlias = Annotated[List[JsonNumber], Field(min_length=1)]
PoseReference: TypeAlias = Union[Identifier, Pose]


class Actuator(StrictBaseModel):
    """One scalar actuator and the unit used by its trajectory values."""

    name: Identifier
    unit: Annotated[
        str,
        Field(min_length=1, pattern=r"^[A-Za-z][A-Za-z0-9_./*-]*$"),
    ]


class HoldPiece(StrictBaseModel):
    """Hold the current value or an explicit pose."""

    type: Literal["hold"]
    duration_s: PositiveNumber
    pose: PoseReference | None = None


class TransitionPiece(StrictBaseModel):
    """Move from the current value to a target pose."""

    type: Literal["transition"]
    duration_s: PositiveNumber
    to: PoseReference
    interpolation: Literal["linear", "quintic_smoothstep"] = "quintic_smoothstep"


class SineComponent(StrictBaseModel):
    """One actuator's sinusoidal offset."""

    actuator: Identifier
    amplitude: JsonNumber
    frequency_hz: NonNegativeNumber
    phase_rad: JsonNumber = 0.0


class ChirpComponent(StrictBaseModel):
    """One actuator's linear-frequency chirp offset."""

    actuator: Identifier
    amplitude: JsonNumber
    start_frequency_hz: NonNegativeNumber
    end_frequency_hz: NonNegativeNumber
    phase_rad: JsonNumber = 0.0


class SinePiece(StrictBaseModel):
    """Oscillate selected actuators sinusoidally about a center pose."""

    type: Literal["sine"]
    duration_s: PositiveNumber
    center: PoseReference | None = None
    ramp_s: NonNegativeNumber = 1.0
    components: Annotated[List[SineComponent], Field(min_length=1)]


class ChirpPiece(StrictBaseModel):
    """Sweep selected actuators linearly in frequency about a center pose."""

    type: Literal["chirp"]
    duration_s: PositiveNumber
    center: PoseReference | None = None
    ramp_s: NonNegativeNumber = 1.0
    components: Annotated[List[ChirpComponent], Field(min_length=1)]


TrajectoryPiece: TypeAlias = Annotated[
    Union[HoldPiece, TransitionPiece, SinePiece, ChirpPiece],
    Field(discriminator="type"),
]


class Trajectory(StrictBaseModel):
    """One named, ordered piecewise trajectory."""

    name: Identifier
    initial_pose: PoseReference
    pieces: Annotated[List[TrajectoryPiece], Field(min_length=1)]


class TrajectoryConfig(BaseConfig):
    """A reusable pack of actuator poses and generated trajectories."""

    type: Literal["trajectory_v1"]  # pyright: ignore[reportIncompatibleVariableOverride]
    name: Identifier
    sample_rate_hz: PositiveInteger
    actuators: Annotated[List[Actuator], Field(min_length=1)]
    poses: Annotated[Dict[Identifier, Pose], Field(min_length=1)]
    trajectories: Annotated[List[Trajectory], Field(min_length=1)]

    @field_validator("actuators")
    @classmethod
    def actuator_names_must_be_unique(
        cls,
        actuators: List[Actuator],
    ) -> List[Actuator]:
        names = [actuator.name for actuator in actuators]
        if len(names) != len(set(names)):
            raise ValueError("actuator names must be unique")
        return actuators

    @field_validator("trajectories")
    @classmethod
    def trajectory_names_must_be_unique(
        cls,
        trajectories: List[Trajectory],
    ) -> List[Trajectory]:
        names = [trajectory.name for trajectory in trajectories]
        if len(names) != len(set(names)):
            raise ValueError("trajectory names must be unique")
        return trajectories

    @model_validator(mode="after")
    def validate_references_and_pose_widths(self) -> TrajectoryConfig:
        actuator_count = len(self.actuators)
        actuator_names = {actuator.name for actuator in self.actuators}

        if "home" not in self.poses:
            raise ValueError("poses must define a 'home' pose")
        for name, pose in self.poses.items():
            self._validate_pose_width(pose, actuator_count, f"pose {name!r}")

        for trajectory in self.trajectories:
            self._validate_pose_reference(
                trajectory.initial_pose,
                actuator_count,
                f"trajectory {trajectory.name!r} initial_pose",
            )
            for index, piece in enumerate(trajectory.pieces):
                location = f"trajectory {trajectory.name!r} piece {index}"
                if isinstance(piece, HoldPiece) and piece.pose is not None:
                    self._validate_pose_reference(
                        piece.pose,
                        actuator_count,
                        f"{location} pose",
                    )
                elif isinstance(piece, TransitionPiece):
                    self._validate_pose_reference(
                        piece.to,
                        actuator_count,
                        f"{location} target",
                    )
                elif isinstance(piece, (SinePiece, ChirpPiece)):
                    if piece.center is not None:
                        self._validate_pose_reference(
                            piece.center,
                            actuator_count,
                            f"{location} center",
                        )
                    unknown = [
                        component.actuator
                        for component in piece.components
                        if component.actuator not in actuator_names
                    ]
                    if unknown:
                        raise ValueError(
                            f"{location} references unknown actuators: "
                            + ", ".join(unknown)
                        )
        return self

    def _validate_pose_reference(
        self,
        reference: PoseReference,
        actuator_count: int,
        location: str,
    ) -> None:
        if isinstance(reference, str):
            if reference not in self.poses:
                raise ValueError(f"{location} references unknown pose {reference!r}")
            return
        self._validate_pose_width(reference, actuator_count, location)

    @staticmethod
    def _validate_pose_width(
        pose: Pose,
        actuator_count: int,
        location: str,
    ) -> None:
        if len(pose) != actuator_count:
            raise ValueError(
                f"{location} must contain exactly {actuator_count} actuator values"
            )
