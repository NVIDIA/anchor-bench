
import unittest

from anchor_bench.core.versions import load_external_versions


def _major_minor(version: str) -> str:
    """Return the leading ``major.minor`` of a dotted version string."""
    parts = version.split(".")
    return ".".join(parts[:2])


class VersionManifestTests(unittest.TestCase):
    def test_manifest_pins_newton_isaac_sim_and_isaac_lab(self):
        versions = load_external_versions()
        # Pin the major.minor compatibility line but allow patch/build bumps in
        # versions/compatibility.toml without a test edit.
        self.assertEqual(_major_minor(versions.isaac_sim_version), "6.0")
        self.assertEqual(_major_minor(versions.newton_version), "1.2")
        # Isaac Lab tracks a branch + exact commit, not a semver, so keep these exact.
        self.assertEqual(versions.isaac_lab_branch, "develop")
        self.assertEqual(versions.isaac_lab_commit, "482d21c936b863e323f0a135c8e8e8e8a9c166b2")


if __name__ == "__main__":
    unittest.main()
