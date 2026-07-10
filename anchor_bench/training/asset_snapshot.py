"""Immutable, host-safe snapshots of Newton training assets.

This module deliberately does not import Newton, Warp, MuJoCo, or MuJoCo
Warp.  Runtime objects are passed to :func:`build_asset_snapshot` only after a
caller has entered the full simulator environment.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import numpy as np


JointKind = Literal["revolute", "prismatic"]


def _as_numpy(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    """Copy a Warp-like or array-like value to a host NumPy array."""

    source = value.numpy() if hasattr(value, "numpy") else value
    return np.asarray(source, dtype=dtype)


def _leaf_name(label: str) -> str:
    """Return the final path component of a Newton joint label."""

    return label.rsplit("/", maxsplit=1)[-1]


def _loose_name(name: str) -> str:
    """Return the compatibility alias used by recorded SAGE joint names."""

    return _leaf_name(name).removesuffix("_joint")


def _ordered_names(values: Sequence[str], *, label: str) -> tuple[str, ...]:
    """Validate and freeze an ordered sequence of unique joint names."""

    if isinstance(values, (str, bytes)):
        raise ValueError(f"{label} must be an ordered sequence of joint names")
    names = tuple(str(value).strip() for value in values)
    if not names or any(not name for name in names):
        raise ValueError(f"{label} must contain non-empty joint names")
    duplicates = sorted({name for index, name in enumerate(names) if name in names[:index]})
    if duplicates:
        raise ValueError(f"{label} must be unique: {', '.join(duplicates)}")
    return names


def _freeze_mapping(values: Mapping[str, Any]) -> Mapping[str, Any]:
    """Copy a mapping into a read-only proxy."""

    return MappingProxyType(dict(values))


@dataclass(frozen=True)
class ScalarJointMapping:
    """Canonical one-coordinate/one-DOF Newton joint mapping."""

    name: str
    label: str
    joint_index: int
    q_index: int
    dof_index: int
    kind: JointKind

    def provenance(self) -> dict[str, Any]:
        """Serialize the scalar joint's resolved Newton indices."""

        return {
            "name": self.name,
            "label": self.label,
            "joint_index": self.joint_index,
            "q_index": self.q_index,
            "dof_index": self.dof_index,
            "kind": self.kind,
        }


@dataclass(frozen=True)
class AssetSnapshot:
    """Resolved Newton asset identity, scalar mapping, and physical limits.

    The coordinate and DOF indices are for the first replicated world.  They
    are therefore also the per-world indices used after reshaping a batched
    Newton state.
    """

    path: Path
    sha256: str
    scalar_joints: tuple[ScalarJointMapping, ...]
    gravity: tuple[float, float, float]
    versions: Mapping[str, str]
    solver_settings: Mapping[str, Any]
    worlds: int
    coordinates_per_world: int
    dofs_per_world: int
    _default_q: np.ndarray = field(repr=False, compare=False)
    _effort_by_dof: np.ndarray = field(repr=False, compare=False)
    _unsupported_labels: tuple[str, ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        """Validate dimensions and freeze all mutable snapshot payloads."""

        path = Path(self.path).resolve()
        if len(self.sha256) != 64 or any(character not in "0123456789abcdef" for character in self.sha256):
            raise ValueError("asset sha256 must be a lowercase SHA-256 digest")
        if self.worlds < 1 or self.coordinates_per_world < 0 or self.dofs_per_world < 0:
            raise ValueError("asset snapshot dimensions must be non-negative with at least one world")
        if len(self.gravity) != 3 or not all(math.isfinite(value) for value in self.gravity):
            raise ValueError("asset gravity must contain three finite values")

        default_q = np.asarray(self._default_q, dtype=np.float32)
        expected_q = self.worlds * self.coordinates_per_world
        if default_q.shape != (expected_q,) or not np.all(np.isfinite(default_q)):
            raise ValueError(f"model default joint_q must contain {expected_q} finite values")
        default_q = np.array(default_q, dtype=np.float32, copy=True)
        default_q.setflags(write=False)

        effort = np.asarray(self._effort_by_dof, dtype=np.float64)
        expected_effort = self.worlds * self.dofs_per_world
        if effort.shape != (expected_effort,):
            raise ValueError(f"model joint_effort_limit must contain {expected_effort} values")
        effort = np.array(effort, dtype=np.float64, copy=True)
        effort.setflags(write=False)

        canonical_names = [joint.name for joint in self.scalar_joints]
        duplicates = sorted({name for index, name in enumerate(canonical_names) if name in canonical_names[:index]})
        if duplicates:
            raise ValueError("Newton asset has duplicate canonical scalar joint names: " + ", ".join(duplicates))

        object.__setattr__(self, "path", path)
        object.__setattr__(self, "versions", _freeze_mapping(self.versions))
        object.__setattr__(self, "solver_settings", _freeze_mapping(self.solver_settings))
        object.__setattr__(self, "_default_q", default_q)
        object.__setattr__(self, "_effort_by_dof", effort)

    @property
    def asset_sha256(self) -> str:
        """Compatibility spelling used by runtime metadata."""

        return self.sha256

    @property
    def default_q(self) -> np.ndarray:
        """Read-only replicated model-default generalized positions."""

        return self._default_q

    def _matches(self, requested_name: str, joints: Sequence[ScalarJointMapping]) -> list[ScalarJointMapping]:
        """Return exact or compatible scalar-joint matches for a recorded name."""

        exact_label = [joint for joint in joints if joint.label == requested_name]
        if exact_label:
            return exact_label
        exact_leaf = [joint for joint in joints if joint.name == requested_name]
        if exact_leaf:
            return exact_leaf
        loose = _loose_name(requested_name)
        return [joint for joint in joints if _loose_name(joint.name) == loose]

    def resolve_joint(
        self,
        name: str,
        *,
        required: bool = True,
        role: str = "joint",
    ) -> ScalarJointMapping | None:
        """Resolve an exact or unambiguous legacy leaf-name alias."""

        requested = str(name).strip()
        if not requested:
            raise ValueError(f"{role} name must not be empty")
        matches = self._matches(requested, self.scalar_joints)
        if len(matches) > 1:
            labels = ", ".join(joint.label for joint in matches)
            raise ValueError(f"{role} {requested!r} matches multiple Newton scalar joints: {labels}")
        if matches:
            return matches[0]

        unsupported = [
            label
            for label in self._unsupported_labels
            if label == requested
            or _leaf_name(label) == requested
            or _loose_name(label) == _loose_name(requested)
        ]
        if required and unsupported:
            raise ValueError(f"{role} {requested!r} must map to one scalar revolute or prismatic Newton joint")
        if required:
            raise ValueError(f"{role} {requested!r} was not found in the Newton model")
        return None

    def effort_limits(self, joint_names: Sequence[str]) -> tuple[float, ...]:
        """Return finite positive effort limits in the caller's output order."""

        names = _ordered_names(joint_names, label="output joint names")
        values: list[float] = []
        owned_dofs: dict[int, str] = {}
        for name in names:
            joint = self.resolve_joint(name, role="output joint")
            assert joint is not None
            previous = owned_dofs.get(joint.dof_index)
            if previous is not None:
                raise ValueError(
                    f"output joints {previous!r} and {name!r} map to the same Newton scalar joint/DOF"
                )
            owned_dofs[joint.dof_index] = name
            value = float(self._effort_by_dof[joint.dof_index])
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"model.joint_effort_limit for output joint {name!r} must be finite and positive, got {value}"
                )
            values.append(value)
        return tuple(values)

    @property
    def provenance(self) -> Mapping[str, Any]:
        """Common immutable provenance shared by training feature derivations."""

        return MappingProxyType(
            {
                "asset": self.path.name,
                "asset_sha256": self.sha256,
                "newton_version": self.versions["newton"],
                "mujoco_version": self.versions["mujoco"],
                "mujoco_warp_version": self.versions["mujoco_warp"],
                "gravity": list(self.gravity),
                "solver_settings": dict(self.solver_settings),
                "scalar_joint_mapping": [joint.provenance() for joint in self.scalar_joints],
            }
        )


def build_asset_snapshot(
    *,
    asset_path: str | Path,
    asset_sha256: str,
    model: Any,
    template_joint_labels: Sequence[str],
    joints_per_world: int,
    coordinates_per_world: int,
    dofs_per_world: int,
    newton: Any,
    versions: Mapping[str, str],
    solver_settings: Mapping[str, Any],
) -> AssetSnapshot:
    """Build an :class:`AssetSnapshot` from one finalized Newton model."""

    labels = tuple(str(label) for label in template_joint_labels)
    if len(labels) != joints_per_world:
        raise ValueError(
            f"Newton template exposes {len(labels)} labels for {joints_per_world} joints"
        )

    joint_types = _as_numpy(model.joint_type, dtype=np.int64).reshape(-1)
    q_starts = _as_numpy(model.joint_q_start, dtype=np.int64).reshape(-1)
    qd_starts = _as_numpy(model.joint_qd_start, dtype=np.int64).reshape(-1)
    if len(joint_types) < joints_per_world:
        raise ValueError("Newton model joint_type is shorter than the template joint count")
    if len(q_starts) < joints_per_world + 1 or len(qd_starts) < joints_per_world + 1:
        raise ValueError("Newton model joint q/qd starts must include a sentinel")

    revolute = int(newton.JointType.REVOLUTE)
    prismatic = int(newton.JointType.PRISMATIC)
    scalar: list[ScalarJointMapping] = []
    unsupported: list[str] = []
    for joint_index, label in enumerate(labels):
        joint_type = int(joint_types[joint_index])
        q_index = int(q_starts[joint_index])
        dof_index = int(qd_starts[joint_index])
        q_width = int(q_starts[joint_index + 1] - q_index)
        dof_width = int(qd_starts[joint_index + 1] - dof_index)
        if joint_type in (revolute, prismatic) and q_width == 1 and dof_width == 1:
            scalar.append(
                ScalarJointMapping(
                    name=_leaf_name(label),
                    label=label,
                    joint_index=joint_index,
                    q_index=q_index,
                    dof_index=dof_index,
                    kind="revolute" if joint_type == revolute else "prismatic",
                )
            )
        else:
            unsupported.append(label)

    gravity_values = _as_numpy(model.gravity, dtype=np.float64)
    if gravity_values.shape == (3,):
        gravity = gravity_values
    elif gravity_values.ndim == 2 and gravity_values.shape[1] == 3 and len(gravity_values):
        gravity = gravity_values[0]
        if not np.allclose(gravity_values, gravity, rtol=0.0, atol=0.0):
            raise ValueError("replicated Newton training worlds must use identical gravity")
    else:
        raise ValueError("Newton model gravity must have shape (3,) or (worlds, 3)")

    default_q = _as_numpy(model.joint_q, dtype=np.float32).reshape(-1)
    effort = _as_numpy(model.joint_effort_limit, dtype=np.float64).reshape(-1)
    worlds = int(getattr(model, "world_count", 1))
    return AssetSnapshot(
        path=Path(asset_path),
        sha256=asset_sha256,
        scalar_joints=tuple(scalar),
        gravity=tuple(float(value) for value in gravity),
        versions=versions,
        solver_settings=solver_settings,
        worlds=worlds,
        coordinates_per_world=coordinates_per_world,
        dofs_per_world=dofs_per_world,
        _default_q=default_q,
        _effort_by_dof=effort,
        _unsupported_labels=tuple(unsupported),
    )


__all__ = [
    "AssetSnapshot",
    "JointKind",
    "ScalarJointMapping",
    "build_asset_snapshot",
]
