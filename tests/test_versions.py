
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
        self.assertEqual(versions.newton_version, "1.4.0rc1")
        self.assertEqual(versions.newton_package, "newton")
        self.assertEqual(versions.newton_repo, "https://github.com/newton-physics/newton.git")
        self.assertEqual(versions.newton_tag, "v1.4.0rc1")
        # Isaac Lab tracks a branch + exact commit, not a semver, so keep these exact.
        self.assertEqual(versions.isaac_lab_branch, "develop")
        self.assertEqual(versions.isaac_lab_commit, "5f2db9902baa724427b7aa654835aa559fce64e8")


if __name__ == "__main__":
    unittest.main()
