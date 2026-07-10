
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from anchor_bench.core.paths import repo_root, robot_config_root
from anchor_bench.core.schemas import ActuatorFile, RobotRunFile, RobotSpec, SysIdBoundsFile

_ROBOT_CONFIG_ERRORS = (OSError, ValueError, KeyError, yaml.YAMLError)


@dataclass(frozen=True)
class RobotPaths:
    config_dir: Path
    robot: Path
    run: Path
    actuators: Path
    bounds: Path
    asset: Path


@dataclass(frozen=True)
class RobotBundle:
    spec: RobotSpec
    paths: RobotPaths
    run_config: RobotRunFile


def list_robots() -> list[str]:
    root = robot_config_root()
    if not root.exists():
        return []
    return sorted(path.name for path in root.iterdir() if (path / "robot.yaml").exists())


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _load_robot_bundle(name: str) -> RobotBundle:
    cfg_dir = robot_config_root() / name
    robot_file = cfg_dir / "robot.yaml"
    if not robot_file.exists():
        raise KeyError(f"unknown robot '{name}'")
    spec = RobotSpec.from_dict(_load_yaml(robot_file))
    run_file = cfg_dir / "run.yaml"
    run_config = RobotRunFile.from_dict(
        _load_yaml(run_file) if run_file.exists() else {},
        run_file,
    )
    paths = RobotPaths(
        config_dir=cfg_dir,
        robot=robot_file,
        run=run_file,
        actuators=cfg_dir / spec.actuator_file,
        bounds=cfg_dir / spec.bounds_file,
        asset=repo_root() / spec.asset_path,
    )
    return RobotBundle(spec=spec, paths=paths, run_config=run_config)


def _validate_robot_bundle(name: str, bundle: RobotBundle) -> list[str]:
    errors: list[str] = []
    for label in ("robot", "run", "actuators", "bounds", "asset"):
        path = getattr(bundle.paths, label)
        if not path.exists():
            errors.append(f"{name}: missing {label} file {path}")

    try:
        actuator_file = ActuatorFile.from_dict(_load_yaml(bundle.paths.actuators), bundle.paths.actuators)
    except _ROBOT_CONFIG_ERRORS as exc:
        errors.append(f"{name}: invalid actuator file: {exc}")
        actuator_file = None

    if actuator_file is not None:
        valid_joints = set(bundle.spec.joint_names)
        for actuator in actuator_file.actuators:
            unknown = sorted(set(actuator.joints) - valid_joints)
            if unknown:
                errors.append(
                    f"{name}: actuator {actuator.name} joints not in robot.yaml joint_names: "
                    f"{', '.join(unknown)}"
                )

    if bundle.paths.bounds.exists():
        try:
            bounds = SysIdBoundsFile.from_dict(_load_yaml(bundle.paths.bounds), bundle.paths.bounds)
            valid_joint_types = set(bundle.spec.joint_names) | set(bundle.spec.joint_aliases)
            unknown = sorted(set(bounds.joint_types) - valid_joint_types)
            if unknown:
                errors.append(
                    f"{name}: bounds joint_types not in robot.yaml joint_names or joint_aliases: "
                    f"{', '.join(unknown)}"
                )
        except _ROBOT_CONFIG_ERRORS as exc:
            errors.append(f"{name}: invalid bounds file: {exc}")
    return errors


def get_robot(name: str) -> RobotBundle:
    bundle = _load_robot_bundle(name)
    errors = _validate_robot_bundle(name, bundle)
    if errors:
        raise ValueError("; ".join(errors))
    return bundle


def validate_robot(name: str) -> list[str]:
    try:
        bundle = _load_robot_bundle(name)
    except _ROBOT_CONFIG_ERRORS as exc:
        return [str(exc)]
    return _validate_robot_bundle(name, bundle)
