
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from anchor_bench.actuators.loader import load_actuator_file
from anchor_bench.core.paths import repo_root, robot_config_root
from anchor_bench.core.schemas import RobotRunFile, RobotSpec


@dataclass(frozen=True)
class RobotPaths:
    config_dir: Path
    robot: Path
    run: Path
    actuators: Path
    bounds: Path
    joints: Path
    valid_joints: Path
    asset: Path


@dataclass(frozen=True)
class RobotBundle:
    spec: RobotSpec
    paths: RobotPaths
    run_config: dict[str, Any]


def list_robots() -> list[str]:
    root = robot_config_root()
    if not root.exists():
        return []
    return sorted(path.name for path in root.iterdir() if (path / "robot.yaml").exists())


def get_robot(name: str) -> RobotBundle:
    cfg_dir = robot_config_root() / name
    robot_file = cfg_dir / "robot.yaml"
    if not robot_file.exists():
        raise KeyError(f"unknown robot '{name}'")
    spec = RobotSpec.from_dict(yaml.safe_load(robot_file.read_text(encoding="utf-8")) or {})
    run_file = cfg_dir / "run.yaml"
    run_config = RobotRunFile.from_dict(
        yaml.safe_load(run_file.read_text(encoding="utf-8")) if run_file.exists() else {},
        run_file,
    ).to_dict()
    paths = RobotPaths(
        config_dir=cfg_dir,
        robot=robot_file,
        run=run_file,
        actuators=cfg_dir / spec.actuator_file,
        bounds=cfg_dir / spec.bounds_file,
        joints=cfg_dir / "joints.yaml",
        valid_joints=cfg_dir / spec.valid_joints_file,
        asset=repo_root() / spec.asset_path,
    )
    return RobotBundle(spec=spec, paths=paths, run_config=run_config or {})


def validate_robot(name: str) -> list[str]:
    errors: list[str] = []
    try:
        bundle = get_robot(name)
    except Exception as exc:
        return [str(exc)]
    for label in ("robot", "run", "actuators", "bounds", "joints", "valid_joints", "asset"):
        path = getattr(bundle.paths, label)
        if not path.exists():
            errors.append(f"{name}: missing {label} file {path}")
    try:
        load_actuator_file(bundle.paths.actuators)
    except Exception as exc:
        errors.append(f"{name}: invalid actuator file: {exc}")
    if bundle.paths.bounds.exists():
        try:
            from anchor_bench.core.schemas import SysIdBoundsFile

            SysIdBoundsFile.from_dict(
                yaml.safe_load(bundle.paths.bounds.read_text(encoding="utf-8")) or {},
                bundle.paths.bounds,
            )
        except Exception as exc:
            errors.append(f"{name}: invalid bounds file: {exc}")
    valid = bundle.paths.valid_joints.read_text(encoding="utf-8").splitlines() if bundle.paths.valid_joints.exists() else []
    if valid and valid != bundle.spec.joint_names:
        errors.append(f"{name}: valid_joints.txt order does not match robot.yaml joint_names")
    return errors
