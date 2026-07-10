from __future__ import annotations

import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml
from pydantic import ValidationError

from anchor_bench.core.artifacts import (
    generated_files,
    verify_run_directory,
    write_manifest_last,
)
from anchor_bench.core.hashing import canonical_sha256, sha256_file, tree_fingerprint
from anchor_bench.core.manifest_schema import RunManifest, build_profile_snapshot
from anchor_bench.sysid.state import (
    AcceptedSysIdState,
    SysIdCandidateSource,
    SysIdFittedPatch,
    build_candidate_state,
)
from anchor_bench.training.export import serialize_runtime_metadata
from anchor_bench.training.metadata import (
    GRURecurrentDimensions,
    NormalizationValues,
    RuntimeFeature,
    RuntimeGRUMetadata,
    RuntimeHashes,
    RuntimeMapping,
    RuntimeNormalization,
)
from anchor_bench.training.promotion import promote_training_run
from anchor_bench.training.runner import (
    TrainingInvocation,
    TrainingResolverInputs,
    _assemble_episodes,
    execute_training_run,
    resolve_training_run,
)
from anchor_bench.training.schema import (
    TrainingAcceptedCatalog,
    TrainingArtifactReference,
    TrainingBenchmarkEvidence,
    TrainingFeature,
    TrainingSourceReference,
)


_ASSET_HASH = "a" * 64
_ASSET_PACKAGE_HASH = "b" * 64


def _profile(*, mapping=None, target="torque"):
    """Return a compact profile document suitable for phase tests."""

    return {
        "schema_version": 1,
        "sysid_state": "default",
        "data": "inputs/training",
        "sample_dt_s": 0.01,
        "network": {"type": "gru", "hidden_size": 4, "num_layers": 1},
        "target": target,
        "mappings": mapping or [
            {"input_joints": ["AliasA"], "output_joints": ["AliasA"]}
        ],
        "input_features": [
            {"name": "position", "domain": "input", "components": ["value"]}
        ],
        "fit": {"skip_optuna": True, "final_epochs": 1, "seed": 42},
    }


class TrainingPhaseFixture:
    """Build one isolated repository-shaped Training fixture."""

    def __init__(self, root: Path, *, profile=None):
        self.root = root
        self.profiles = root / "configs/robots/demo/train/profiles"
        self.profiles.mkdir(parents=True)
        self.dataset = root / "inputs/training"
        self.dataset.mkdir(parents=True)
        self.accepted_sysid = root / "configs/robots/demo/sysid/default.yaml"
        self.accepted_sysid.parent.mkdir(parents=True)
        self.write_profile("arm", profile or _profile())
        self.accepted_sysid.write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "revision": 0,
                    "parameters": {
                        "stiffness": {"a": 20.0, "b": 30.0},
                        "damping": {"a": 0.5, "b": 0.75},
                        "armature": {"a": 0.01, "b": 0.02},
                        "dry_friction": {"a": 0.1, "b": 0.2},
                        "viscous_friction": {"a": 0.01, "b": 0.02},
                    },
                    "provenance": {"history": []},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

    def write_profile(self, name: str, document) -> None:
        """Write one complete Training profile."""

        (self.profiles / f"{name}.yaml").write_text(
            yaml.safe_dump(document, sort_keys=False),
            encoding="utf-8",
        )

    def inputs(self) -> TrainingResolverInputs:
        """Return injected host-safe resolver inputs."""

        def run_directory(phase, *, robot, profile, seed, run_input_sha256):
            """Derive a fixture-local content-addressed run directory."""

            return (
                self.root
                / "outputs"
                / phase
                / robot
                / profile
                / f"seed-{seed}"
                / run_input_sha256
            )

        return TrainingResolverInputs(
            repository_root=self.root,
            profiles_directory=self.profiles,
            accepted_sysid_state=self.accepted_sysid,
            canonical_joint_names=("a", "b"),
            joint_aliases={"AliasA": "a", "AliasB": "b"},
            robot_identity_fingerprint={
                "kind": "robot_yaml_identity",
                "raw_sha256": "e" * 64,
                "canonical_sha256": "f" * 64,
            },
            asset_fingerprint={
                "kind": "runtime_asset_package",
                "runtime_usd": "demo.usd",
                "asset_sha256": _ASSET_HASH,
                "root_usd_sha256": _ASSET_HASH,
                "package_tree_sha256": _ASSET_PACKAGE_HASH,
                "package_files": {"demo.usd": _ASSET_HASH},
            },
            source_fingerprint={"source_sha256": "c" * 64},
            dependency_fingerprints={"torch": "test", "newton": "test"},
            dataset_fingerprint=tree_fingerprint(self.dataset),
            run_directory_resolver=run_directory,
        )


def _runtime_metadata(resolved, staging: Path) -> RuntimeGRUMetadata:
    """Construct matching schema-v3 metadata for a fake host backend."""

    input_width = len(resolved.mappings[0].input_joints)
    output_width = len(resolved.mappings[0].output_joints)

    def scalar_or_vector(values):
        """Encode one-element metadata vectors as scalars."""

        return values[0] if len(values) == 1 else tuple(values)

    mappings = tuple(
        RuntimeMapping(
            input_joints=mapping.input_joints,
            output_joints=mapping.output_joints,
            effort_limits=scalar_or_vector([50.0] * output_width),
            clamping="total_output",
        )
        for mapping in resolved.mappings
    )
    feature_width = input_width
    return RuntimeGRUMetadata(
        input_size=feature_width,
        output_size=output_width,
        recurrent=GRURecurrentDimensions(hidden_size=4, num_layers=1),
        input_features=(
            RuntimeFeature(
                name="position",
                domain="input",
                components=("value",),
                width=feature_width,
            ),
        ),
        mappings=mappings,
        target_columns=("torque",),
        sample_dt_s=resolved.resolved_config.sample_dt_s,
        normalization=RuntimeNormalization(
            inputs=NormalizationValues(
                mean={"position": scalar_or_vector([0.0] * feature_width)},
                std={"position": scalar_or_vector([1.0] * feature_width)},
            ),
            targets=NormalizationValues(
                mean={"torque": scalar_or_vector([0.0] * output_width)},
                std={"torque": scalar_or_vector([1.0] * output_width)},
            ),
        ),
        hashes=RuntimeHashes(
            profile_sha256=resolved.profile_snapshot.content_sha256,
            resolved_config_sha256=canonical_sha256(
                resolved.resolved_config.to_dict()
            ),
            sysid_state_sha256=sha256_file(staging / "sysid_state.yaml"),
            asset_sha256=_ASSET_HASH,
            asset_package_sha256=_ASSET_PACKAGE_HASH,
            dataset_sha256=resolved.fingerprints["dataset"]["tree_sha256"],
            source_sha256=resolved.fingerprints["source"]["source_sha256"],
        ),
    )


def _fake_backend(calls: list[Path]):
    """Return a backend that emits the exact fixed Training file surface."""

    def backend(resolved, staging):
        """Write a metadata-bearing zip and lightweight checkpoint/stats files."""

        calls.append(staging)
        checkpoint = staging / "checkpoints/best.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"checkpoint")
        stats = staging / "stats.json"
        stats.write_text("{}\n", encoding="utf-8")
        network = staging / "network.pt"
        with zipfile.ZipFile(network, "w") as archive:
            archive.writestr(
                "archive/extra/metadata.json",
                serialize_runtime_metadata(_runtime_metadata(resolved, staging)),
            )
        return SimpleNamespace(
            checkpoint_path=checkpoint,
            stats_path=stats,
            network_path=network,
        )

    return backend


def _benchmark_run(root: Path, source_run: Path, name: str = "candidate") -> Path:
    """Create one completed benchmark that references an exact Training run."""

    source = verify_run_directory(source_run)
    run_dir = root / "outputs/benchmark/demo/train/default" / name
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text("{}\n", encoding="utf-8")
    profile = build_profile_snapshot(
        name="default",
        path="configs/robots/demo/benchmark/profiles/default.yaml",
        raw_yaml="schema_version: 1\n",
        content={"schema_version": 1},
    )
    source_relative = source_run.relative_to(root).as_posix()
    source_reference = {
        "phase": "train",
        "robot": "demo",
        "run": source_relative,
        "run_input_sha256": source.run_input_sha256,
        "run_manifest_sha256": source.run_manifest_sha256,
    }
    resolved_config = {"schema_version": 1}
    run_input = {
        "phase": "benchmark",
        "robot": "demo",
        "profile": {
            "name": profile.name,
            "content": profile.content,
            "content_sha256": profile.content_sha256,
        },
        "resolved_config": resolved_config,
        "seed": None,
        "replicate": None,
        "fingerprints": {},
        "candidate": {"source": source_reference},
        "name": name,
    }
    manifest = RunManifest.build(
        phase="benchmark",
        robot="demo",
        profile=profile,
        resolved_config=resolved_config,
        run_input=run_input,
        generated_files=generated_files(
            run_dir,
            {"metrics.json": "metrics"},
        ),
        upstream={"source": source_reference},
    )
    write_manifest_last(run_dir, manifest)
    return run_dir


def _sysid_candidate(fixture: TrainingPhaseFixture) -> Path:
    """Create one authenticated unpromoted SysID candidate state."""

    profile = build_profile_snapshot(
        name="default",
        path="configs/robots/demo/sysid/profiles/default.yaml",
        raw_yaml="schema_version: 1\n",
        content={"schema_version": 1},
    )
    resolved_config = {"schema_version": 1}
    run_input = {
        "phase": "sysid",
        "robot": "demo",
        "profile": {
            "name": profile.name,
            "content": profile.content,
            "content_sha256": profile.content_sha256,
        },
        "resolved_config": resolved_config,
        "seed": 42,
        "replicate": None,
        "fingerprints": {},
        "candidate": "unpromoted",
    }
    run_input_sha256 = canonical_sha256(run_input)
    run_dir = (
        fixture.root
        / "outputs/sysid/demo/default/seed-42"
        / run_input_sha256
    )
    run_dir.mkdir(parents=True)
    parent = AcceptedSysIdState.from_dict(
        yaml.safe_load(fixture.accepted_sysid.read_text(encoding="utf-8")),
        fixture.accepted_sysid,
    )
    patch = SysIdFittedPatch(
        schema_version=1,
        artifact_type="sysid_fitted_patch",
        robot="demo",
        parameters={"stiffness": {"a": 25.0}},
    )
    (run_dir / "parent_state.yaml").write_text(
        yaml.safe_dump(parent.to_dict(), sort_keys=False),
        encoding="utf-8",
    )
    (run_dir / "fitted_patch.yaml").write_text(
        yaml.safe_dump(patch.to_dict(), sort_keys=False),
        encoding="utf-8",
    )
    candidate = build_candidate_state(
        parent,
        patch,
        SysIdCandidateSource(
            parent_state_sha256=sha256_file(run_dir / "parent_state.yaml"),
            fitted_patch_sha256=sha256_file(run_dir / "fitted_patch.yaml"),
            run_input_sha256=run_input_sha256,
        ),
    )
    (run_dir / "sysid_state.yaml").write_text(
        yaml.safe_dump(candidate.to_dict(), sort_keys=False),
        encoding="utf-8",
    )
    manifest = RunManifest.build(
        phase="sysid",
        robot="demo",
        profile=profile,
        resolved_config=resolved_config,
        seed=42,
        run_input=run_input,
        generated_files=generated_files(
            run_dir,
            {
                "parent_state.yaml": "parent_state",
                "fitted_patch.yaml": "fitted_patch",
                "sysid_state.yaml": "sysid_state",
            },
        ),
    )
    write_manifest_last(run_dir, manifest)
    return run_dir / "sysid_state.yaml"


class TrainingResolutionTests(unittest.TestCase):
    def test_resolution_rejects_components_builtin_providers_cannot_produce(self):
        """Keep generic feature blocks while failing unsupported recipes before runtime."""

        with tempfile.TemporaryDirectory() as temporary:
            fixture = TrainingPhaseFixture(Path(temporary))
            with self.assertRaisesRegex(
                ValueError,
                "built-in training feature 'velocity'.*only the 'value' component",
            ):
                resolve_training_run(
                    TrainingInvocation(
                        robot="demo",
                        profile="arm",
                        input_features=(
                            TrainingFeature(
                                name="velocity",
                                domain="input",
                                components=("vx", "vy", "vz"),
                            ),
                        ),
                    ),
                    fixture.inputs(),
                )

    def test_cli_mapping_and_fit_overrides_replace_yaml_and_resolve_aliases(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = TrainingPhaseFixture(Path(temporary))
            resolved = resolve_training_run(
                TrainingInvocation(
                    robot="demo",
                    profile="arm",
                    input_joints=("AliasB",),
                    output_joints=("AliasB",),
                    seed=7,
                ),
                fixture.inputs(),
            )

            self.assertEqual(("b",), resolved.mappings[0].input_joints)
            self.assertEqual(7, resolved.resolved_config.fit.seed)
            self.assertIn("seed-7", resolved.run_dir.parts)
            self.assertEqual(("AliasA",), fixture_profile_joint(resolved))

    def test_alias_and_canonical_cli_spellings_have_same_semantic_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = TrainingPhaseFixture(Path(temporary))
            alias = resolve_training_run(
                TrainingInvocation(robot="demo", profile="arm", input_joints=("AliasB",)),
                fixture.inputs(),
            )
            canonical = resolve_training_run(
                TrainingInvocation(robot="demo", profile="arm", input_joints=("b",)),
                fixture.inputs(),
            )

            self.assertEqual(alias.run_input_sha256, canonical.run_input_sha256)

    def test_all_is_one_coupled_mapping_in_canonical_robot_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = TrainingPhaseFixture(Path(temporary))
            resolved = resolve_training_run(
                TrainingInvocation(robot="demo", profile="arm", input_joints="all"),
                fixture.inputs(),
            )

            self.assertEqual(("a", "b"), resolved.mappings[0].input_joints)
            self.assertEqual(("a", "b"), resolved.mappings[0].output_joints)

    def test_residual_resolution_requires_every_mapping_baseline_cell(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = TrainingPhaseFixture(
                Path(temporary),
                profile=_profile(
                    target="torque_residual",
                    mapping=[
                        {
                            "input_joints": ["AliasA", "AliasB"],
                            "output_joints": ["AliasB"],
                        }
                    ],
                ),
            )
            document = yaml.safe_load(fixture.accepted_sysid.read_text(encoding="utf-8"))
            del document["parameters"]["damping"]["b"]
            fixture.accepted_sysid.write_text(yaml.safe_dump(document), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "damping/b"):
                resolve_training_run(
                    TrainingInvocation(robot="demo", profile="arm"),
                    fixture.inputs(),
                )

    def test_manifest_backed_sysid_candidate_is_frozen_and_path_normalized(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = TrainingPhaseFixture(Path(temporary))
            candidate = _sysid_candidate(fixture)
            absolute = resolve_training_run(
                TrainingInvocation(
                    robot="demo",
                    profile="arm",
                    sysid_state=candidate,
                ),
                fixture.inputs(),
            )
            relative = resolve_training_run(
                TrainingInvocation(
                    robot="demo",
                    profile="arm",
                    sysid_state=candidate.relative_to(fixture.root).as_posix(),
                ),
                fixture.inputs(),
            )

            self.assertEqual(25.0, absolute.sysid_state.parameters["stiffness"]["a"])
            self.assertEqual(absolute.run_input_sha256, relative.run_input_sha256)
            self.assertFalse(Path(absolute.resolved_config.sysid_state).is_absolute())


def fixture_profile_joint(resolved) -> tuple[str, ...]:
    """Read the immutable raw profile semantics retained in the snapshot."""

    return tuple(resolved.profile_snapshot.content["mappings"][0]["input_joints"])


class TrainingExecutionTests(unittest.TestCase):
    def test_episode_split_precedes_mapping_expansion(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = TrainingPhaseFixture(
                Path(temporary),
                profile=_profile(
                    mapping=[
                        {"input_joints": ["AliasA"], "output_joints": ["AliasA"]},
                        {"input_joints": ["AliasB"], "output_joints": ["AliasB"]},
                    ]
                ),
            )
            resolved = resolve_training_run(
                TrainingInvocation(robot="demo", profile="arm"),
                fixture.inputs(),
            )
            sources = [SimpleNamespace(path=fixture.dataset / name) for name in ("one", "two")]
            for source in sources:
                source.path.mkdir()
            split_calls = []

            def split(values, *, seed):
                """Record the pre-expansion source count and separate episodes."""

                split_calls.append((len(values), seed))
                return [values[0]], [values[1]]

            assembler = SimpleNamespace(
                load=lambda path, sample_dt: path,
                assemble=lambda prepared, **kwargs: (prepared, kwargs["mapping"]),
            )
            with (
                patch("anchor_bench.training.data.discover_training_sources", return_value=sources),
                patch("anchor_bench.training.data.split_training_sources", side_effect=split),
            ):
                train, validation = _assemble_episodes(resolved, assembler)

            self.assertEqual([(2, 42)], split_calls)
            self.assertEqual(2, len(train))
            self.assertEqual(2, len(validation))

    def test_immutable_execution_installs_once_and_reuses_verified_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = TrainingPhaseFixture(Path(temporary))
            resolved = resolve_training_run(
                TrainingInvocation(robot="demo", profile="arm", dry_run=False),
                fixture.inputs(),
            )
            calls = []
            first = execute_training_run(resolved, _fake_backend(calls))
            second = execute_training_run(resolved, _fake_backend(calls))

            self.assertFalse(first.no_op)
            self.assertTrue(second.no_op)
            self.assertEqual(1, len(calls))
            self.assertEqual("train", verify_run_directory(resolved.run_dir).phase)
            self.assertEqual(
                {
                    "checkpoint",
                    "network",
                    "resolved_config",
                    "stats",
                    "sysid_state",
                },
                {entry.role for entry in first.manifest.generated_files},
            )

    def test_dry_run_never_calls_backend_or_writes_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = TrainingPhaseFixture(Path(temporary))
            resolved = resolve_training_run(
                TrainingInvocation(robot="demo", profile="arm", dry_run=True),
                fixture.inputs(),
            )
            calls = []
            outcome = execute_training_run(resolved, _fake_backend(calls))

            self.assertTrue(outcome.dry_run)
            self.assertEqual([], calls)
            self.assertFalse(resolved.run_dir.exists())

    def test_dataset_mutation_after_resolution_fails_before_episode_loading(self):
        """Reject dataset bytes changed between immutable resolution and execution."""

        with tempfile.TemporaryDirectory() as temporary:
            fixture = TrainingPhaseFixture(Path(temporary))
            resolved = resolve_training_run(
                TrainingInvocation(robot="demo", profile="arm", dry_run=False),
                fixture.inputs(),
            )
            (fixture.dataset / "late.csv").write_text("changed\n", encoding="utf-8")

            def backend(candidate, _staging):
                """Cross the normal episode-assembly boundary without simulator imports."""

                return _assemble_episodes(candidate, SimpleNamespace())

            with self.assertRaisesRegex(RuntimeError, "training dataset changed after resolution"):
                execute_training_run(resolved, backend)
            self.assertFalse(resolved.run_dir.exists())


class TrainingPromotionTests(unittest.TestCase):
    def _completed(self, fixture: TrainingPhaseFixture, profile="arm"):
        """Execute one fake immutable candidate and return its directory."""

        resolved = resolve_training_run(
            TrainingInvocation(robot="demo", profile=profile, dry_run=False),
            fixture.inputs(),
        )
        execute_training_run(resolved, _fake_backend([]))
        return resolved.run_dir

    def test_promotion_references_network_in_place_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = TrainingPhaseFixture(Path(temporary))
            run_dir = self._completed(fixture)
            catalog = fixture.root / "configs/robots/demo/train/default.yaml"
            catalog.write_text("schema_version: 1\nrevision: 0\nmodels: {}\n", encoding="utf-8")

            first = promote_training_run(
                run_dir,
                catalog,
                repository_root=fixture.root,
            )
            second = promote_training_run(
                run_dir,
                catalog,
                repository_root=fixture.root,
            )

            self.assertFalse(first.no_op)
            self.assertTrue(second.no_op)
            model = first.catalog.models["arm"]
            self.assertEqual((run_dir / "network.pt").relative_to(fixture.root).as_posix(), model.artifact.path)
            self.assertFalse((catalog.parent / "network.pt").exists())

    def test_matching_benchmark_adds_exact_evidence_without_becoming_a_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = TrainingPhaseFixture(Path(temporary))
            run_dir = self._completed(fixture)
            benchmark = _benchmark_run(fixture.root, run_dir)
            catalog = fixture.root / "configs/robots/demo/train/default.yaml"
            catalog.write_text("schema_version: 1\nrevision: 0\nmodels: {}\n", encoding="utf-8")
            promote_training_run(run_dir, catalog, repository_root=fixture.root)

            attached = promote_training_run(
                run_dir,
                catalog,
                repository_root=fixture.root,
                benchmark_runs=(benchmark,),
            )
            repeated = promote_training_run(
                run_dir,
                catalog,
                repository_root=fixture.root,
                benchmark_runs=(benchmark,),
            )

            evidence = attached.catalog.models["arm"].source.benchmarks
            self.assertEqual(1, len(evidence))
            self.assertEqual(verify_run_directory(benchmark).run_manifest_sha256, evidence[0].run_manifest_sha256)
            self.assertTrue(repeated.no_op)

    def test_cross_key_output_overlap_is_a_hard_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = TrainingPhaseFixture(Path(temporary))
            fixture.write_profile("other", _profile())
            first_run = self._completed(fixture, "arm")
            second_run = self._completed(fixture, "other")
            catalog = fixture.root / "configs/robots/demo/train/default.yaml"
            catalog.write_text("schema_version: 1\nrevision: 0\nmodels: {}\n", encoding="utf-8")
            promote_training_run(first_run, catalog, repository_root=fixture.root)

            with self.assertRaisesRegex(ValidationError, "both own output joint 'a'"):
                promote_training_run(second_run, catalog, repository_root=fixture.root)

    def test_dry_run_validates_but_does_not_mutate_catalog(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = TrainingPhaseFixture(Path(temporary))
            run_dir = self._completed(fixture)
            catalog = fixture.root / "configs/robots/demo/train/default.yaml"
            initial = "schema_version: 1\nrevision: 0\nmodels: {}\n"
            catalog.write_text(initial, encoding="utf-8")

            outcome = promote_training_run(
                run_dir,
                catalog,
                repository_root=fixture.root,
                dry_run=True,
            )

            self.assertTrue(outcome.dry_run)
            self.assertEqual(initial, catalog.read_text(encoding="utf-8"))
            self.assertEqual(1, outcome.catalog.revision)
            self.assertFalse(catalog.with_name("default.yaml.lock").exists())

    def test_promotion_rejects_a_copied_completed_run(self):
        """Require the exact manifest-derived canonical Training directory."""

        with tempfile.TemporaryDirectory() as temporary:
            fixture = TrainingPhaseFixture(Path(temporary))
            run_dir = self._completed(fixture)
            copied = fixture.root / "outputs/train/demo/copied-run"
            shutil.copytree(run_dir, copied)
            catalog = fixture.root / "configs/robots/demo/train/default.yaml"
            initial = "schema_version: 1\nrevision: 0\nmodels: {}\n"
            catalog.write_text(initial, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "canonical run directory"):
                promote_training_run(
                    copied,
                    catalog,
                    repository_root=fixture.root,
                )
            self.assertEqual(initial, catalog.read_text(encoding="utf-8"))

    def test_promotion_rejects_a_catalog_for_another_robot(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = TrainingPhaseFixture(Path(temporary))
            run_dir = self._completed(fixture)
            wrong_catalog = fixture.root / "configs/robots/other/train/default.yaml"
            wrong_catalog.parent.mkdir(parents=True)
            wrong_catalog.write_text(
                "schema_version: 1\nrevision: 0\nmodels: {}\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "configs/robots/demo/train"):
                promote_training_run(
                    run_dir,
                    wrong_catalog,
                    repository_root=fixture.root,
                )
            self.assertFalse(wrong_catalog.with_name("default.yaml.lock").exists())

    def test_benchmark_evidence_schema_rejects_duplicates_and_unsafe_paths(self):
        evidence = {
            "run": "outputs/benchmark/demo/run",
            "run_manifest_sha256": "e" * 64,
        }
        with self.assertRaisesRegex(ValidationError, "duplicate evidence"):
            TrainingSourceReference.model_validate(
                {
                    "run": "outputs/train/demo/run",
                    "run_input_sha256": "a" * 64,
                    "run_manifest_sha256": "b" * 64,
                    "sysid_state_sha256": "c" * 64,
                    "benchmarks": [evidence, evidence],
                }
            )
        with self.assertRaisesRegex(ValidationError, "relative POSIX"):
            TrainingBenchmarkEvidence.model_validate(
                {**evidence, "run": "../escape"}
            )

    def test_accepted_reference_schema_rejects_unsafe_paths_and_bad_hashes(self):
        with self.assertRaisesRegex(ValidationError, "relative POSIX"):
            TrainingArtifactReference.model_validate(
                {
                    "path": "../network.pt",
                    "sha256": "a" * 64,
                    "metadata_sha256": "b" * 64,
                }
            )
        with self.assertRaisesRegex(ValidationError, "lowercase SHA-256"):
            TrainingSourceReference.model_validate(
                {
                    "run": "outputs/train/demo/run",
                    "run_input_sha256": "invalid",
                    "run_manifest_sha256": "b" * 64,
                    "sysid_state_sha256": "c" * 64,
                }
            )

    def test_accepted_catalog_rejects_non_logical_model_keys(self):
        entry = {
            "revision": 1,
            "artifact": {
                "path": "outputs/train/demo/run/network.pt",
                "sha256": "a" * 64,
                "metadata_sha256": "b" * 64,
            },
            "contract": {
                "target": "torque",
                "sample_dt_s": 0.01,
                "mappings": [
                    {"input_joints": ["a"], "output_joints": ["a"]}
                ],
            },
            "source": {
                "run": "outputs/train/demo/run",
                "run_input_sha256": "c" * 64,
                "run_manifest_sha256": "d" * 64,
                "sysid_state_sha256": "e" * 64,
            },
        }
        with self.assertRaisesRegex(ValidationError, "invalid model key"):
            TrainingAcceptedCatalog.model_validate(
                {"schema_version": 1, "revision": 1, "models": {"Bad Key": entry}}
            )


if __name__ == "__main__":
    unittest.main()
