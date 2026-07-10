from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from anchor_bench.core.fingerprints import source_fingerprint


def _git(root: Path, *args: str) -> None:
    subprocess.run(("git", *args), cwd=root, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


class SourceFingerprintTests(unittest.TestCase):
    def _repository(self, root: Path) -> None:
        _git(root, "init", "-q")
        _git(root, "config", "user.name", "Anchor Test")
        _git(root, "config", "user.email", "anchor@example.com")
        (root / "anchor_bench").mkdir()
        (root / "anchor_bench/module.py").write_text("VALUE = 1\n", encoding="utf-8")
        (root / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
        (root / "versions").mkdir()
        (root / "versions/compatibility.toml").write_text("version='1'\n", encoding="utf-8")
        _git(root, "add", ".")
        _git(root, "commit", "-qm", "initial")

    def test_clean_fingerprint_is_stable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._repository(root)

            first = source_fingerprint(root)
            second = source_fingerprint(root)

        self.assertEqual(first, second)
        self.assertEqual({}, first["untracked_files"])
        self.assertEqual(40, len(first["git_commit"]))

    def test_tracked_and_untracked_source_changes_affect_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._repository(root)
            clean = source_fingerprint(root)

            (root / "anchor_bench/module.py").write_text("VALUE = 2\n", encoding="utf-8")
            tracked = source_fingerprint(root)
            (root / "anchor_bench/new.py").write_text("NEW = True\n", encoding="utf-8")
            untracked = source_fingerprint(root)

        self.assertNotEqual(clean["source_sha256"], tracked["source_sha256"])
        self.assertNotEqual(tracked["source_sha256"], untracked["source_sha256"])
        self.assertIn("anchor_bench/new.py", untracked["untracked_files"])

    def test_dataset_changes_are_not_duplicated_into_source_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._repository(root)
            before = source_fingerprint(root)
            (root / "inputs").mkdir()
            (root / "inputs/data.bin").write_bytes(b"dataset")
            after = source_fingerprint(root)

        self.assertEqual(before, after)

    def test_source_paths_reject_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._repository(root)
            with self.assertRaisesRegex(ValueError, "safe repository-relative"):
                source_fingerprint(root, source_paths=("../outside",))

    def test_untracked_source_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._repository(root)
            target = root / "target.py"
            target.write_text("VALUE = 1\n", encoding="utf-8")
            (root / "anchor_bench/link.py").symlink_to(target)

            with self.assertRaisesRegex(ValueError, "must not be a symlink"):
                source_fingerprint(root)


if __name__ == "__main__":
    unittest.main()
