from __future__ import annotations

import copy
import zipfile
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from anchor_bench.core.artifacts import generated_file, write_manifest
from anchor_bench.core.hashing import canonical_sha256, sha256_file, sha256_hex
from anchor_bench.core.manifest_schema import RunManifest, build_profile_snapshot
from anchor_bench.deploy.promotion import promote_deploy
from anchor_bench.deploy.runner import (
    DeployResolverInputs,
    LoadedNeuralMetadata,
    resolve_deploy_state,
)
from anchor_bench.training.export import serialize_runtime_metadata
from anchor_bench.training.metadata import (
    ArmatureValues,
    FrictionBaseline,
    GRURecurrentDimensions,
    NormalizationValues,
    PDBaseline,
    PreviousTorqueDerivation,
    RuntimeFeature,
    RuntimeGRUMetadata,
    RuntimeHashes,
    RuntimeMapping,
    RuntimeNormalization,
)


def _write_yaml(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


def _deploy_profile(*, control_freq: int = 500) -> dict:
    return {
        "schema_version": 1,
        "simulation": {
            "physics_freq": 500,
            "control_freq": control_freq,
            "render_freq": 500,
            "integrator": "implicitfast",
            "solver_iterations": 10,
            "line_search_iterations": 10,
        },
        "composition": {
            "neural": [
                {
                    "name": "arm_network",
                    "model": "right_arm_gru",
                    "mappings": [
                        {
                            "index": 0,
                            "input_joints": ["sh", "el"],
                            "output_joints": ["el"],
                        }
                    ],
                }
            ],
            "pd": [
                {
                    "name": "shoulder_pd",
                    "joints": ["shoulder"],
                    "effort_limits": {"shoulder": 30.0},
                }
            ],
        },
    }


def _sysid_state() -> dict:
    return {
        "schema_version": 1,
        "revision": 0,
        "parameters": {
            "stiffness": {"shoulder": 40.0, "elbow": 50.0, "untouched": 60.0},
            "damping": {"shoulder": 1.0, "elbow": 1.5, "untouched": 2.0},
            "armature": {"shoulder": 0.01, "elbow": 0.02, "untouched": 0.03},
            "dry_friction": {"shoulder": 0.1, "elbow": 0.2, "untouched": 0.3},
            "viscous_friction": {"shoulder": 0.01, "elbow": 0.02, "untouched": 0.03},
        },
        "provenance": {"history": []},
    }


def _runtime_metadata(
    *,
    profile_sha256: str,
    resolved_config_sha256: str,
    sysid_state_sha256: str,
    sample_dt_s: float = 0.002,
) -> RuntimeGRUMetadata:
    mapping = RuntimeMapping(
        input_joints=("shoulder", "elbow"),
        output_joints=("elbow",),
        pd_baseline=PDBaseline(
            joint_names=("shoulder", "elbow"),
            kp=(11.0, 22.0),
            kd=(0.5, 0.75),
        ),
        friction_baseline=FrictionBaseline(
            joint_names=("elbow",),
            dry=0.12,
            viscous=0.03,
        ),
        armature=ArmatureValues(joint_names=("elbow",), values=0.02),
        effort_limits=25.0,
        clamping="residual_output",
    )
    return RuntimeGRUMetadata(
        input_size=3,
        output_size=1,
        recurrent=GRURecurrentDimensions(hidden_size=8, num_layers=1),
        input_features=(
            RuntimeFeature(
                name="position_error",
                domain="input",
                components=("value",),
                width=2,
            ),
            RuntimeFeature(
                name="previous_torque",
                domain="output",
                components=("value",),
                width=1,
            ),
        ),
        mappings=(mapping,),
        target_columns=("torque_residual",),
        sample_dt_s=sample_dt_s,
        normalization=RuntimeNormalization(
            inputs=NormalizationValues(
                mean={"position_error": (0.0, 0.0), "previous_torque": 0.0},
                std={"position_error": (1.0, 1.0), "previous_torque": 1.0},
            ),
            targets=NormalizationValues(
                mean={"torque_residual": 0.0},
                std={"torque_residual": 1.0},
            ),
        ),
        previous_torque_derivation=PreviousTorqueDerivation(value_space="residual_torque"),
        hashes=RuntimeHashes(
            profile_sha256=profile_sha256,
            resolved_config_sha256=resolved_config_sha256,
            sysid_state_sha256=sysid_state_sha256,
            asset_sha256="a" * 64,
            asset_package_sha256="b" * 64,
            dataset_sha256="c" * 64,
            source_sha256="d" * 64,
        ),
    )


def _setup(
    tmp_path: Path,
    *,
    network_role: str = "network",
) -> tuple[DeployResolverInputs, RuntimeGRUMetadata]:
    profiles = tmp_path / "configs/robots/demo/deploy/profiles"
    _write_yaml(profiles / "default.yaml", _deploy_profile())
    sysid_path = tmp_path / "configs/robots/demo/sysid/default.yaml"
    _write_yaml(sysid_path, _sysid_state())

    training_profile_content = {
        "schema_version": 1,
        "target": "torque_residual",
        "mappings": [
            {
                "input_joints": ["shoulder", "elbow"],
                "output_joints": ["elbow"],
            }
        ],
    }
    training_profile_raw = (
        "# formatting is deliberately not the metadata identity\n"
        + yaml.safe_dump(training_profile_content, sort_keys=True)
    )
    profile_snapshot = build_profile_snapshot(
        name="right_arm_gru",
        path="configs/robots/demo/train/profiles/right_arm_gru.yaml",
        raw_yaml=training_profile_raw,
        content=training_profile_content,
    )
    resolved_config = {
        "schema_version": 1,
        "target": "torque_residual",
        "sample_dt_s": 0.002,
    }
    training_sysid_hash = "c" * 64
    metadata = _runtime_metadata(
        profile_sha256=profile_snapshot.content_sha256,
        resolved_config_sha256=canonical_sha256(resolved_config),
        sysid_state_sha256=training_sysid_hash,
    )
    metadata_bytes = serialize_runtime_metadata(metadata).encode("utf-8")

    fingerprints = {
        "asset": {
            "kind": "runtime_asset_package",
            "runtime_usd": "demo.usd",
            "asset_sha256": "a" * 64,
            "root_usd_sha256": "a" * 64,
            "package_tree_sha256": "b" * 64,
            "package_files": {"demo.usd": "a" * 64},
        },
        "dataset": {"tree_sha256": "c" * 64},
        "source": {"source_sha256": "d" * 64},
    }
    upstream = {"sysid_state_sha256": training_sysid_hash}
    run_input = {
        "phase": "train",
        "robot": "demo",
        "profile": {
            "name": profile_snapshot.name,
            "content": profile_snapshot.content,
            "content_sha256": profile_snapshot.content_sha256,
        },
        "resolved_config": resolved_config,
        "seed": 42,
        "replicate": None,
        "fingerprints": fingerprints,
    }
    run_input_sha256 = canonical_sha256(run_input)
    run_dir = (
        tmp_path
        / "outputs/train/demo/right_arm_gru/seed-42"
        / run_input_sha256
    )
    run_dir.mkdir(parents=True)
    artifact = run_dir / "network.pt"
    with zipfile.ZipFile(artifact, "w") as archive:
        archive.writestr("archive/extra/metadata.json", metadata_bytes)
    manifest = RunManifest.build(
        phase="train",
        robot="demo",
        profile=profile_snapshot,
        resolved_config=resolved_config,
        seed=42,
        run_input=run_input,
        fingerprints=fingerprints,
        upstream=upstream,
        generated_files=[generated_file(run_dir, "network.pt", network_role)],
    )
    write_manifest(run_dir, manifest)

    run_relative = run_dir.relative_to(tmp_path).as_posix()
    artifact_relative = artifact.relative_to(tmp_path).as_posix()
    training_state = {
        "schema_version": 1,
        "revision": 3,
        "models": {
            "right_arm_gru": {
                "revision": 2,
                "artifact": {
                    "path": artifact_relative,
                    "sha256": sha256_file(artifact),
                    "metadata_sha256": sha256_hex(metadata_bytes),
                },
                "contract": {
                    "target": "torque_residual",
                    "sample_dt_s": 0.002,
                    "mappings": [
                        {
                            "input_joints": ["shoulder", "elbow"],
                            "output_joints": ["elbow"],
                        }
                    ],
                },
                "source": {
                    "run": run_relative,
                    "run_input_sha256": manifest.run_input_sha256,
                    "run_manifest_sha256": manifest.run_manifest_sha256,
                    "sysid_state_sha256": training_sysid_hash,
                    "benchmarks": [],
                },
            }
        },
    }
    training_path = tmp_path / "configs/robots/demo/train/default.yaml"
    _write_yaml(training_path, training_state)
    deploy_path = profiles.parent / "default.yaml"
    inputs = DeployResolverInputs(
        repository_root=tmp_path,
        robot="demo",
        profiles_directory=profiles,
        accepted_training_state=training_path,
        accepted_sysid_state=sysid_path,
        deployed_state=deploy_path,
        canonical_joint_names=("shoulder", "elbow", "untouched"),
        joint_aliases={"sh": "shoulder", "el": "elbow"},
        robot_identity_fingerprint={
            "kind": "robot_yaml_identity",
            "raw_sha256": "d" * 64,
            "canonical_sha256": "e" * 64,
        },
        asset_fingerprint={
            "kind": "runtime_asset_package",
            "runtime_usd": "demo.usd",
            "asset_sha256": "a" * 64,
            "root_usd_sha256": "a" * 64,
            "package_tree_sha256": "b" * 64,
            "package_files": {"demo.usd": "a" * 64},
        },
    )
    return inputs, metadata


def _loader(metadata: RuntimeGRUMetadata):
    metadata_sha256 = sha256_hex(serialize_runtime_metadata(metadata).encode("utf-8"))

    def load(_path: Path) -> LoadedNeuralMetadata:
        return LoadedNeuralMetadata(metadata=metadata, metadata_sha256=metadata_sha256)

    return load


def _mutate_yaml(path: Path, mutate) -> None:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    mutate(document)
    _write_yaml(path, document)


def test_resolution_freezes_authenticated_neural_and_explicit_pd_state(tmp_path: Path) -> None:
    inputs, metadata = _setup(tmp_path)

    state = resolve_deploy_state(inputs, metadata_loader=_loader(metadata))

    assert state.source.robot == "demo"
    assert state.source.robot_identity == dict(inputs.robot_identity_fingerprint)
    assert state.source.asset_identity == dict(inputs.asset_fingerprint)
    assert state.ownership == {"elbow": "arm_network", "shoulder": "shoulder_pd"}
    assert "untouched" not in state.ownership
    neural = state.composition.neural[0]
    assert neural.accepted_model_revision == 2
    assert neural.target == "torque_residual"
    assert neural.mappings[0].input_joints == ("shoulder", "elbow")
    assert neural.mappings[0].pd_baseline.kp == {"shoulder": 11.0, "elbow": 22.0}
    assert neural.mappings[0].effort_limits == {"elbow": 25.0}
    assert neural.mappings[0].clamping == "residual_output"
    pd = state.composition.pd[0]
    assert pd.kp == {"shoulder": 40.0}
    assert pd.dry_friction == {"shoulder": 0.1}
    assert pd.armature == {"shoulder": 0.01}
    assert pd.effort_limits == {"shoulder": 30.0}


def test_new_deployment_requires_authoritative_robot_and_asset_identity(tmp_path: Path) -> None:
    """Never promote a new frozen state whose runtime package is unbound."""

    inputs, metadata = _setup(tmp_path)
    with pytest.raises(ValueError, match="robot identity fingerprint"):
        resolve_deploy_state(
            replace(inputs, robot_identity_fingerprint=None),
            metadata_loader=_loader(metadata),
        )
    with pytest.raises(ValueError, match="authoritative asset fingerprint"):
        resolve_deploy_state(
            replace(inputs, asset_fingerprint=None),
            metadata_loader=_loader(metadata),
        )


def test_default_metadata_loader_is_host_safe(tmp_path: Path) -> None:
    inputs, _metadata = _setup(tmp_path)

    state = resolve_deploy_state(inputs)

    assert state.composition.neural[0].mappings[0].output_joints == ("elbow",)


def test_metadata_profile_binding_uses_semantic_content_hash(tmp_path: Path) -> None:
    inputs, metadata = _setup(tmp_path)
    train_manifest = next((tmp_path / "outputs/train/demo/right_arm_gru/seed-42").glob("*/manifest.yaml"))
    manifest_document = yaml.safe_load(train_manifest.read_text(encoding="utf-8"))

    assert metadata.hashes.profile_sha256 == manifest_document["profile"]["content_sha256"]
    assert metadata.hashes.profile_sha256 != manifest_document["profile"]["raw_sha256"]
    resolve_deploy_state(inputs, metadata_loader=_loader(metadata))


def test_deploy_profile_alias_collisions_and_canonical_output_overlap_fail(tmp_path: Path) -> None:
    inputs, metadata = _setup(tmp_path)
    profile_path = inputs.profiles_directory / "default.yaml"
    _mutate_yaml(
        profile_path,
        lambda data: data["composition"]["neural"][0]["mappings"][0].update(
            input_joints=["shoulder", "sh"], output_joints=["sh"]
        ),
    )
    with pytest.raises(ValueError, match="duplicate neural mapping 0 input joints after alias"):
        resolve_deploy_state(inputs, metadata_loader=_loader(metadata))

    _write_yaml(profile_path, _deploy_profile())
    _mutate_yaml(
        profile_path,
        lambda data: data["composition"]["pd"][0].update(
            joints=["elbow"], effort_limits={"elbow": 30.0}
        ),
    )
    with pytest.raises(ValueError, match="output joint 'elbow' is owned by both"):
        resolve_deploy_state(inputs, metadata_loader=_loader(metadata))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda data: data["composition"]["neural"][0]["mappings"][0].update(index=1),
            "mapping index 1 is outside",
        ),
        (
            lambda data: data["composition"]["neural"][0]["mappings"][0].update(
                input_joints=["el", "sh"]
            ),
            "input joints do not exactly match",
        ),
        (
            lambda data: data["simulation"].update(control_freq=250),
            "timestep does not match deployment control frequency",
        ),
    ],
)
def test_selected_mapping_index_axes_and_timestep_are_exact(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    inputs, metadata = _setup(tmp_path)
    _mutate_yaml(inputs.profiles_directory / "default.yaml", mutation)

    with pytest.raises(ValueError, match=message):
        resolve_deploy_state(inputs, metadata_loader=_loader(metadata))


def test_artifact_and_embedded_metadata_hashes_are_authenticated(tmp_path: Path) -> None:
    inputs, metadata = _setup(tmp_path)
    _mutate_yaml(
        inputs.accepted_training_state,
        lambda data: data["models"]["right_arm_gru"]["artifact"].update(sha256="0" * 64),
    )
    with pytest.raises(ValueError, match="artifact hash does not match"):
        resolve_deploy_state(inputs, metadata_loader=_loader(metadata))

    inputs, metadata = _setup(tmp_path / "metadata")

    def wrong_loader(_path):
        """Return valid metadata under an intentionally wrong embedded hash."""

        return LoadedNeuralMetadata(
            metadata=metadata,
            metadata_sha256="0" * 64,
        )

    with pytest.raises(ValueError, match="embedded metadata hash does not match"):
        resolve_deploy_state(inputs, metadata_loader=wrong_loader)


def test_training_manifest_requires_the_network_role_at_network_path(tmp_path: Path) -> None:
    inputs, metadata = _setup(tmp_path, network_role="checkpoint")

    with pytest.raises(ValueError, match="'network' role at 'network.pt'"):
        resolve_deploy_state(inputs, metadata_loader=_loader(metadata))


def test_metadata_must_match_accepted_catalog_and_source_manifest(tmp_path: Path) -> None:
    inputs, metadata = _setup(tmp_path)
    wrong_hashes = metadata.hashes.model_copy(update={"profile_sha256": "0" * 64})
    wrong_metadata = metadata.model_copy(update={"hashes": wrong_hashes})
    wrong_digest = sha256_hex(serialize_runtime_metadata(wrong_metadata).encode("utf-8"))
    _mutate_yaml(
        inputs.accepted_training_state,
        lambda data: data["models"]["right_arm_gru"]["artifact"].update(
            metadata_sha256=wrong_digest
        ),
    )

    with pytest.raises(ValueError, match="metadata profile hash does not match"):
        resolve_deploy_state(inputs, metadata_loader=_loader(wrong_metadata))

    inputs, metadata = _setup(tmp_path / "contract")
    _mutate_yaml(
        inputs.accepted_training_state,
        lambda data: data["models"]["right_arm_gru"]["contract"].update(
            target="torque"
        ),
    )
    with pytest.raises(ValueError, match="target does not match"):
        resolve_deploy_state(inputs, metadata_loader=_loader(metadata))


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("asset_sha256", "metadata asset hash does not match"),
        ("asset_package_sha256", "metadata asset package hash does not match"),
        ("dataset_sha256", "metadata dataset hash does not match"),
        ("source_sha256", "metadata Anchor source hash does not match"),
    ],
)
def test_metadata_asset_and_source_hashes_bind_to_training_manifest(
    tmp_path: Path,
    field: str,
    message: str,
) -> None:
    inputs, metadata = _setup(tmp_path)
    hashes = metadata.hashes.model_copy(update={field: "0" * 64})
    mismatched = metadata.model_copy(update={"hashes": hashes})
    digest = sha256_hex(serialize_runtime_metadata(mismatched).encode("utf-8"))
    _mutate_yaml(
        inputs.accepted_training_state,
        lambda data: data["models"]["right_arm_gru"]["artifact"].update(
            metadata_sha256=digest
        ),
    )

    with pytest.raises(ValueError, match=message):
        resolve_deploy_state(inputs, metadata_loader=_loader(mismatched))


def test_full_torque_network_freezes_total_output_clamping_without_baselines(
    tmp_path: Path,
) -> None:
    inputs, residual = _setup(tmp_path)
    document = residual.model_dump(mode="json", exclude_none=True)
    document["target_columns"] = ["torque"]
    document["normalization"]["targets"] = {
        "mean": {"torque": 0.0},
        "std": {"torque": 1.0},
    }
    document["previous_torque_derivation"]["value_space"] = "physical_torque"
    mapping = document["mappings"][0]
    mapping.pop("pd_baseline")
    mapping.pop("friction_baseline")
    mapping["clamping"] = "total_output"
    metadata = RuntimeGRUMetadata.from_dict(document, "full torque metadata")
    metadata_digest = sha256_hex(serialize_runtime_metadata(metadata).encode("utf-8"))

    def accept_full_torque(data: dict) -> None:
        model = data["models"]["right_arm_gru"]
        model["contract"]["target"] = "torque"
        model["artifact"]["metadata_sha256"] = metadata_digest

    _mutate_yaml(inputs.accepted_training_state, accept_full_torque)

    state = resolve_deploy_state(inputs, metadata_loader=_loader(metadata))

    frozen = state.composition.neural[0].mappings[0]
    assert state.composition.neural[0].target == "torque"
    assert frozen.pd_baseline is None
    assert frozen.friction_baseline is None
    assert frozen.clamping == "total_output"


def test_pd_requires_all_physical_cells_but_armature_is_optional(tmp_path: Path) -> None:
    inputs, metadata = _setup(tmp_path)
    _mutate_yaml(
        inputs.accepted_sysid_state,
        lambda data: data["parameters"]["dry_friction"].pop("shoulder"),
    )
    with pytest.raises(ValueError, match="dry_friction.*missing joints: shoulder"):
        resolve_deploy_state(inputs, metadata_loader=_loader(metadata))

    inputs, metadata = _setup(tmp_path / "no-armature")
    _mutate_yaml(
        inputs.accepted_sysid_state,
        lambda data: data["parameters"].pop("armature"),
    )
    state = resolve_deploy_state(inputs, metadata_loader=_loader(metadata))
    assert state.composition.pd[0].armature is None


def test_pd_only_deployment_does_not_load_neural_metadata(tmp_path: Path) -> None:
    inputs, _metadata = _setup(tmp_path)
    _mutate_yaml(
        inputs.profiles_directory / "default.yaml",
        lambda data: data["composition"].update(neural=[]),
    )
    _write_yaml(
        inputs.accepted_training_state,
        {"schema_version": 1, "revision": 0, "models": {}},
    )

    def unexpected_loader(_path: Path) -> LoadedNeuralMetadata:
        raise AssertionError("PD-only deployment must not inspect TorchScript")

    state = resolve_deploy_state(inputs, metadata_loader=unexpected_loader)

    assert state.composition.neural == ()
    assert state.ownership == {"shoulder": "shoulder_pd"}


def test_resolver_rejects_cross_robot_accepted_state_paths(tmp_path: Path) -> None:
    inputs, _metadata = _setup(tmp_path)
    other_sysid = tmp_path / "configs/robots/other/sysid/default.yaml"
    _write_yaml(other_sysid, _sysid_state())

    with pytest.raises(ValueError, match="accepted_sysid_state must be .*robot 'demo'"):
        DeployResolverInputs(
            repository_root=tmp_path,
            robot="demo",
            profiles_directory=inputs.profiles_directory,
            accepted_training_state=inputs.accepted_training_state,
            accepted_sysid_state=other_sysid,
            deployed_state=inputs.deployed_state,
            canonical_joint_names=inputs.canonical_joint_names,
            joint_aliases=inputs.joint_aliases,
        )


def test_promotion_uses_fixed_lock_order_and_identical_state_is_noop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs, metadata = _setup(tmp_path)
    lock_calls: list[tuple[Path, ...]] = []

    @contextmanager
    def capture_locks(*paths: Path):
        lock_calls.append(tuple(paths))
        yield

    monkeypatch.setattr("anchor_bench.deploy.promotion.ordered_locks", capture_locks)
    first = promote_deploy(inputs, metadata_loader=_loader(metadata))
    original = inputs.deployed_state.read_bytes()
    second = promote_deploy(inputs, metadata_loader=_loader(metadata))

    expected_order = (
        inputs.accepted_training_state,
        inputs.accepted_sysid_state,
        inputs.deployed_state,
    )
    assert lock_calls == [expected_order, expected_order]
    assert first.no_op is False
    assert second.no_op is True
    assert inputs.deployed_state.read_bytes() == original


def test_dry_run_validates_without_writing_and_profile_defaults_are_overridable(
    tmp_path: Path,
) -> None:
    inputs, metadata = _setup(tmp_path)
    alternate = copy.deepcopy(_deploy_profile())
    alternate["composition"]["neural"][0]["name"] = "alternate_network"
    _write_yaml(inputs.profiles_directory / "alternate.yaml", alternate)

    outcome = promote_deploy(
        inputs,
        "alternate",
        metadata_loader=_loader(metadata),
        dry_run=True,
    )

    assert outcome.dry_run is True
    assert outcome.no_op is False
    assert outcome.state.source.profile == "alternate"
    assert outcome.state.ownership["elbow"] == "alternate_network"
    assert not inputs.deployed_state.exists()
    assert not list(tmp_path.rglob("*.lock"))
