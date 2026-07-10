from __future__ import annotations

import hashlib
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "prepare_huggingface_parquets.py"


def load_script():
    spec = importlib.util.spec_from_file_location("prepare_huggingface_parquets", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class PrepareHuggingFaceParquetsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_script()

    def test_parses_huggingface_dataset_folder_and_derives_layout(self):
        folder = self.module._parse_huggingface_folder(
            "https://huggingface.co/datasets/nvidia/Anchor-Lab/tree/main/data/g1_arm_multijoint"
        )
        layout = self.module._import_layout(folder, robot=None, download_dir=None, output_root=None)

        self.assertEqual("nvidia/Anchor-Lab", folder.repo_id)
        self.assertEqual("main", folder.revision)
        self.assertEqual("data/g1_arm_multijoint", folder.path)
        self.assertEqual("g1", layout.robot)
        self.assertEqual(Path("inputs/datasets/g1/g1_arm_multijoint"), layout.output_root)
        self.assertEqual(
            Path("inputs/datasets/g1/_parquet/g1_arm_multijoint"),
            layout.download_dir,
        )

    def test_lists_generic_filenames_and_preserves_upstream_splits(self):
        payload = [
            {
                "type": "file",
                "path": "data/robot_trials/robot-heldout-square-pattern.parquet",
                "lfs": {"oid": "b" * 64},
            },
            {"type": "file", "path": "data/robot_trials/README.md"},
            {
                "type": "file",
                "path": "data/robot_trials/g1-arm-antiphase-trial-004.parquet",
                "lfs": {"oid": "a" * 64},
            },
            {
                "type": "file",
                "path": "data/robot_trials/nested/robot-validation-chirp.parquet",
                "lfs": {"oid": "c" * 64},
            },
        ]
        api = mock.Mock()
        api.list_repo_tree.return_value = payload

        entries = self.module._list_huggingface_parquets(
            "https://huggingface.co/datasets/org/repo/tree/main/data/robot_trials",
            api=api,
        )

        self.assertEqual(["heldout", "train", "validation"], sorted(entry.split for entry in entries))
        by_split = {entry.split: entry for entry in entries}
        self.assertEqual("g1-arm-antiphase-trial-004", by_split["train"].motion)
        self.assertEqual("nested--chirp", by_split["validation"].motion)
        self.assertEqual("square-pattern", by_split["heldout"].motion)
        self.assertTrue(api.list_repo_tree.call_args.kwargs["recursive"])

    def test_regular_non_lfs_parquet_is_supported_without_expected_checksum(self):
        payload = [
            {
                "type": "file",
                "path": "data/demo/robot-trial.parquet",
                "oid": "git-blob-id",
            }
        ]

        api = mock.Mock()
        api.list_repo_tree.return_value = payload
        entries = self.module._list_huggingface_parquets(
            "https://huggingface.co/datasets/org/repo/tree/main/data/demo",
            api=api,
        )

        self.assertIsNone(entries[0].sha256)

    def test_download_verifies_published_lfs_sha256_and_keeps_relative_path(self):
        content = b"parquet fixture"
        digest = hashlib.sha256(content).hexdigest()
        entry = self.module.RemoteParquet(
            path="data/demo/nested/robot-train-demo.parquet",
            relative_path="nested/robot-train-demo.parquet",
            url="https://example.invalid/demo.parquet",
            sha256=digest,
            split="train",
            motion="nested--demo",
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            folder = self.module.HuggingFaceFolder("org/repo", "main", "data/demo")

            def fake_download(**kwargs):
                self.assertEqual(entry.path, kwargs["filename"])
                cached = root / "hub-cache.parquet"
                cached.write_bytes(content)
                return str(cached)

            source = self.module._verified_source(entry, folder, root, no_download=False, downloader=fake_download)

            self.assertEqual(root / "nested" / "robot-train-demo.parquet", source)
            self.assertEqual(content, source.read_bytes())
            source.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                self.module._verified_source(entry, folder, root, no_download=True)

    def test_discovers_so101_g1_and_h1_signal_conventions(self):
        so101 = self.module._discover_signal_fields(
            [
                "rotation/command_q",
                "rotation/actual_q",
                "rotation/dq",
                "rotation/tau_abs",
                "rotation/present_load_raw",
            ]
        )
        g1 = self.module._discover_signal_fields(
            [
                "right_elbow/command_q",
                "right_elbow/actual_q",
                "right_elbow/dq",
                "right_elbow/tau_est",
            ]
        )
        h1 = self.module._discover_signal_fields(["elbow_q", "elbow_dq", "elbow_tau"])

        self.assertEqual(("rotation",), so101.source_joints)
        self.assertEqual("tau_abs", so101.effort_kind)
        self.assertFalse(so101.control_fallback)
        self.assertEqual("tau_est", g1.effort_kind)
        self.assertFalse(g1.control_fallback)
        self.assertEqual("tau", h1.effort_kind)
        self.assertTrue(h1.control_fallback)
        self.assertEqual(h1.position, h1.command)

    def test_maps_known_robot_joints_and_preserves_unknown_joints(self):
        so101 = self.module._canonical_joint_pairs(["elbow", "rotation"], "so101")
        g1 = self.module._canonical_joint_pairs(["elbow", "right_shoulder_pitch"], "g1")
        unknown = self.module._canonical_joint_pairs(["left_ankle_pitch"], "unconfigured_robot")

        self.assertEqual([("rotation", "Rotation"), ("elbow", "Elbow")], so101)
        self.assertEqual(
            [("right_shoulder_pitch", "right_shoulder_pitch"), ("elbow", "right_elbow")],
            g1,
        )
        self.assertEqual([("left_ankle_pitch", "left_ankle_pitch")], unknown)

    def test_bulk_import_routes_outputs_and_passes_robot_provenance(self):
        entry = self.module.RemoteParquet(
            path="data/g1_arm_multijoint/g1-arm-trial.parquet",
            relative_path="g1-arm-trial.parquet",
            url="https://example.invalid/g1-arm-trial.parquet",
            sha256="a" * 64,
            split="train",
            motion="g1-arm-trial",
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            converted = []

            def fake_source(remote, folder, download_dir, *, no_download):
                return download_dir / remote.relative_path

            def fake_convert(source, output, **provenance):
                converted.append((source, output, provenance))

            with (
                mock.patch.object(self.module, "_list_huggingface_parquets", return_value=[entry]),
                mock.patch.object(self.module, "_verified_source", side_effect=fake_source),
                mock.patch.object(self.module, "convert_parquet", side_effect=fake_convert),
            ):
                outputs = self.module.import_huggingface_folder(
                    "https://huggingface.co/datasets/nvidia/Anchor-Lab/tree/main/data/g1_arm_multijoint",
                    download_dir=root / "raw",
                    output_root=root / "sage",
                )

        self.assertEqual([root / "sage" / "train" / "g1-arm-trial"], outputs)
        self.assertEqual("g1", converted[0][2]["robot"])
        self.assertEqual("train", converted[0][2]["source_split"])
        self.assertEqual("g1-arm-trial", converted[0][2]["source_motion"])
        self.assertEqual("main", converted[0][2]["source_revision"])


if __name__ == "__main__":
    unittest.main()
