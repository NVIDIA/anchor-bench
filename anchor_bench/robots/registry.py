"""Host-safe registry for declarative robot identity documents."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from anchor_bench.core.paths import repo_root, robot_config_root
from anchor_bench.core.profiles import load_yaml_mapping, require_logical_name
from anchor_bench.robots.schema import RobotIdentity


@dataclass(frozen=True)
class RobotPaths:
    """Repository locations owned by one robot identity."""

    config_dir: Path
    robot: Path
    asset: Path


@dataclass(frozen=True)
class RobotBundle:
    """A validated identity paired with its repository paths."""

    spec: RobotIdentity
    paths: RobotPaths


def list_robots() -> list[str]:
    """List robots that provide a declarative ``robot.yaml`` identity."""

    root = robot_config_root()
    if not root.exists():
        return []
    return sorted(path.name for path in root.iterdir() if (path / "robot.yaml").is_file())


def _load_robot_bundle(name: str) -> RobotBundle:
    """Load one identity without importing runtimes or resolving external USD bytes."""

    robot_name = require_logical_name(name, label="robot name")
    config_dir = robot_config_root() / robot_name
    robot_file = config_dir / "robot.yaml"
    if not robot_file.is_file():
        raise KeyError(f"unknown robot {robot_name!r}")
    spec = RobotIdentity.from_dict(load_yaml_mapping(robot_file), robot_file)
    if spec.name != robot_name:
        raise ValueError(
            f"{robot_file}.name must match its configuration directory {robot_name!r}"
        )
    return RobotBundle(
        spec=spec,
        paths=RobotPaths(
            config_dir=config_dir,
            robot=robot_file,
            asset=repo_root() / spec.asset_path,
        ),
    )


def _validate_robot_bundle(name: str, bundle: RobotBundle) -> list[str]:
    """Return identity and manifest errors without requiring the runtime asset itself."""

    errors: list[str] = []
    if not bundle.paths.robot.is_file():
        errors.append(f"{name}: missing robot file {bundle.paths.robot}")
    if not bundle.paths.asset.is_file():
        errors.append(f"{name}: missing asset manifest {bundle.paths.asset}")
    return errors


def get_robot(name: str) -> RobotBundle:
    """Return one strictly validated, host-safe robot identity bundle."""

    bundle = _load_robot_bundle(name)
    errors = _validate_robot_bundle(name, bundle)
    if errors:
        raise ValueError("; ".join(errors))
    return bundle


def validate_robot(name: str) -> list[str]:
    """Return validation errors for one robot instead of raising them."""

    try:
        bundle = _load_robot_bundle(name)
    except (KeyError, OSError, ValueError) as error:
        return [str(error)]
    return _validate_robot_bundle(name, bundle)


__all__ = ["RobotBundle", "RobotPaths", "get_robot", "list_robots", "validate_robot"]
