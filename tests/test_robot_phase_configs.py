"""Validation coverage for checked-in authoritative robot phase YAML."""

from __future__ import annotations

from pathlib import Path

import pytest

from anchor_bench.benchmark.schema import BenchmarkProfile
from anchor_bench.core.paths import robot_config_root
from anchor_bench.core.profiles import load_profile, load_yaml_mapping
from anchor_bench.deploy.schema import DeployProfile
from anchor_bench.robots.registry import get_robot, list_robots
from anchor_bench.sysid.schema import SysIdProfile, validate_profile_joints
from anchor_bench.sysid.state import AcceptedSysIdState
from anchor_bench.training.schema import TrainingAcceptedCatalog, TrainingProfile


ROBOT_NAMES = ("g1", "h1", "so101", "teststand", "ur10e")
PHASE_PROFILE_TYPES = {
    "sysid": SysIdProfile,
    "train": TrainingProfile,
    "benchmark": BenchmarkProfile,
    "deploy": DeployProfile,
}


def _profile_files(robot: str, phase: str) -> tuple[Path, ...]:
    """Return every authored profile for one robot phase."""

    directory = robot_config_root() / robot / phase / "profiles"
    return tuple(sorted(directory.glob("*.yaml"))) if directory.is_dir() else ()


def _explicit_training_joints(profile: TrainingProfile) -> set[str]:
    """Collect physical joint names from mappings that do not use ``all``."""

    joints: set[str] = set()
    for mapping in profile.mappings:
        assert mapping.input_joints != "all"
        assert mapping.output_selection != "all"
        joints.update(mapping.input_joints)
        joints.update(mapping.output_selection)
    return joints


def test_every_robot_has_strict_identity_and_revision_zero_phase_state() -> None:
    """Parse all identity and accepted-state documents with their strict schemas."""

    assert tuple(list_robots()) == ROBOT_NAMES
    for robot in ROBOT_NAMES:
        bundle = get_robot(robot)
        joints = set(bundle.spec.joint_names)
        sysid_path = bundle.paths.config_dir / "sysid/default.yaml"
        sysid = AcceptedSysIdState.from_dict(load_yaml_mapping(sysid_path), sysid_path)
        assert sysid.revision == 0
        assert set(sysid.parameters) == {
            "stiffness",
            "damping",
            "armature",
            "dry_friction",
            "viscous_friction",
        }
        assert all(set(values) == joints for values in sysid.parameters.values())

        train_path = bundle.paths.config_dir / "train/default.yaml"
        catalog = TrainingAcceptedCatalog.from_dict(load_yaml_mapping(train_path), train_path)
        assert catalog.revision == 0
        assert dict(catalog.models) == {}
        assert not (bundle.paths.config_dir / "deploy/default.yaml").exists()


@pytest.mark.parametrize("robot", ROBOT_NAMES)
def test_sysid_profiles_are_complete_and_canonical(robot: str) -> None:
    """Require one complete seeded SysID recipe for every checked-in robot."""

    bundle = get_robot(robot)
    loaded = load_profile(
        bundle.paths.config_dir / "sysid/profiles",
        "default",
        SysIdProfile,
    )
    assert loaded.model.seed == 42
    assert loaded.model.simulation.physics_freq >= loaded.model.simulation.control_freq
    assert validate_profile_joints(loaded.model, bundle.spec.joint_names) == tuple(
        loaded.model.joints
    )


def test_every_available_phase_profile_parses_and_uses_robot_joints() -> None:
    """Strictly parse each authored recipe and reject undeclared joint selectors."""

    for robot in ROBOT_NAMES:
        bundle = get_robot(robot)
        canonical = set(bundle.spec.joint_names)
        for phase, model_type in PHASE_PROFILE_TYPES.items():
            for path in _profile_files(robot, phase):
                loaded = load_profile(path.parent, path.stem, model_type)
                profile = loaded.model
                if isinstance(profile, SysIdProfile):
                    assert set(profile.joints) <= canonical
                elif isinstance(profile, TrainingProfile):
                    assert _explicit_training_joints(profile) <= canonical
                elif isinstance(profile, DeployProfile):
                    outputs = {
                        joint
                        for actuator in profile.composition.pd
                        for joint in actuator.joints
                    }
                    outputs.update(
                        joint
                        for actuator in profile.composition.neural
                        for mapping in actuator.mappings
                        for joint in mapping.output_joints
                    )
                    assert outputs <= canonical


def test_so101_profile_preserves_one_shared_weight_model_with_six_siso_mappings() -> None:
    """Represent the former full-joint shared SISO training behavior explicitly."""

    bundle = get_robot("so101")
    profile = load_profile(
        bundle.paths.config_dir / "train/profiles",
        "default",
        TrainingProfile,
    ).model
    expected = tuple((joint,) for joint in bundle.spec.joint_names)
    assert tuple(mapping.input_joints for mapping in profile.mappings) == expected
    assert tuple(mapping.output_selection for mapping in profile.mappings) == expected
    assert profile.target == "torque"


def test_g1_profile_is_one_residual_siso_artifact() -> None:
    """Keep G1's established right-shoulder residual experiment explicit."""

    bundle = get_robot("g1")
    profile = load_profile(
        bundle.paths.config_dir / "train/profiles",
        "default",
        TrainingProfile,
    ).model
    assert profile.target == "torque_residual"
    assert len(profile.mappings) == 1
    assert profile.mappings[0].input_joints == ("right_shoulder_pitch",)
    assert profile.mappings[0].output_selection == ("right_shoulder_pitch",)


def test_deploy_profiles_are_explicit_pd_compositions_with_legacy_effort_clamps() -> None:
    """Freeze only established PD deployments and their per-joint total clamps."""

    expected_limits = {"g1": 25.0, "h1": 300.0, "so101": 3.35, "ur10e": 150.0}
    for robot, expected_limit in expected_limits.items():
        bundle = get_robot(robot)
        profile = load_profile(
            bundle.paths.config_dir / "deploy/profiles",
            "default",
            DeployProfile,
        ).model
        assert profile.composition.neural == ()
        assert len(profile.composition.pd) == 1
        actuator = profile.composition.pd[0]
        assert set(actuator.joints) == set(bundle.spec.joint_names)
        assert set(actuator.effort_limits.values()) == {expected_limit}

    # The legacy teststand controller was an FMU. A PD profile would silently
    # change runtime semantics, so it requires a future explicit authored recipe.
    assert not _profile_files("teststand", "deploy")


def test_legacy_robot_layout_and_generated_config_artifacts_are_absent() -> None:
    """Keep obsolete phase files and generated binaries out of configuration."""

    forbidden_names = {"run.yaml", "actuators.yaml", "bounds.yaml", "manifest.yaml"}
    forbidden_suffixes = {".pt", ".pth", ".onnx", ".json", ".csv", ".png"}
    for robot in ROBOT_NAMES:
        root = robot_config_root() / robot
        assert not any((root / name).exists() for name in forbidden_names)
        files = tuple(path for path in root.rglob("*") if path.is_file())
        assert files
        assert all(path.suffix == ".yaml" for path in files)
        assert not any(path.suffix in forbidden_suffixes for path in files)
