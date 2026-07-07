import unittest
from unittest.mock import patch
from pathlib import Path
import tomllib

from anchor_bench.core.versions import load_external_versions


REPO = Path(__file__).resolve().parents[1]


class VersionManifestTests(unittest.TestCase):
    def test_manifest_pins_newton_isaac_sim_and_isaac_lab(self):
        versions = load_external_versions()
        self.assertEqual(versions.isaac_sim_version, "6.0.0.0")
        self.assertEqual(versions.newton_version, "1.4.0.dev0")
        self.assertEqual(versions.newton_package, "newton")
        self.assertEqual(
            versions.newton_repo,
            "https://github.com/vidurv-nvidia/newton.git",
        )
        self.assertEqual(
            versions.newton_commit, "ae365981f0ba40371d2b19489e4ba07a5e5767ce"
        )
        self.assertEqual(versions.warp_package, "warp-lang")
        self.assertEqual(versions.warp_version, "1.15.0.dev20260626")
        self.assertEqual(versions.isaac_lab_branch, "develop")
        self.assertEqual(
            versions.isaac_lab_commit, "5f2db9902baa724427b7aa654835aa559fce64e8"
        )

    def test_newton_environment_validation_checks_commit_and_capability(self):
        from check_environment import _newton_environment_errors

        errors = _newton_environment_errors(
            "1.3.0",
            "wrong-commit",
            "1.16.0.dev0",
            False,
            False,
            None,
            required_version="1.4.0.dev0",
            required_commit="expected-commit",
            required_warp_version="1.15.0.dev20260626",
        )

        self.assertEqual(5, len(errors))
        self.assertIn("1.4.0.dev0 is required", errors[0])
        self.assertIn("expected-commit", errors[1])
        self.assertIn("ControllerNeuralGRU", errors[2])
        self.assertIn("eval_dynamic_bias", errors[3])
        self.assertIn("warp-lang 1.15.0.dev20260626", errors[4])

    def test_full_extra_matches_manifest_pins(self):
        versions = load_external_versions()
        project = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
        requirements = project["project"]["optional-dependencies"]["full"]

        self.assertIn(f"warp-lang=={versions.warp_version}", requirements)
        self.assertTrue(
            any(
                requirement.startswith("newton[sim] @ git+")
                and requirement.endswith(f"@{versions.newton_commit}")
                for requirement in requirements
            )
        )

    def test_newton_environment_validation_accepts_exact_runtime(self):
        from check_environment import _newton_environment_errors

        errors = _newton_environment_errors(
            "1.4.0.dev0",
            "ABC123",
            "1.15.0.dev20260626",
            True,
            True,
            None,
            required_version="1.4.0.dev0",
            required_commit="abc123",
            required_warp_version="1.15.0.dev20260626",
        )

        self.assertEqual([], errors)

    def test_environment_dry_run_does_not_probe_runtime_packages(self):
        from check_environment import main

        with (
            patch("check_environment._distribution_version") as distribution_version,
            patch("check_environment._distribution_vcs_commit") as vcs_commit,
            patch("check_environment._newton_capabilities") as capabilities,
            patch("check_environment._module_version") as module_version,
        ):
            self.assertEqual(0, main(["--dry-run"]))

        distribution_version.assert_not_called()
        vcs_commit.assert_not_called()
        capabilities.assert_not_called()
        module_version.assert_not_called()

    def test_kitless_profile_enforces_runtime_without_importing_isaac(self):
        from types import SimpleNamespace

        from check_environment import main

        versions = load_external_versions()

        def installed_version(package):
            return {
                versions.newton_package: versions.newton_version,
                versions.warp_package: versions.warp_version,
            }[package]

        with (
            patch("check_environment.list_robots", return_value=[]),
            patch(
                "check_environment._distribution_version", side_effect=installed_version
            ),
            patch(
                "check_environment._distribution_vcs_commit",
                return_value=versions.newton_commit,
            ),
            patch(
                "check_environment._newton_capabilities",
                return_value=(True, True, None),
            ),
            patch("check_environment._module_version") as module_version,
            patch(
                "check_environment.sys.version_info",
                SimpleNamespace(major=3, minor=12, micro=0),
            ),
        ):
            self.assertEqual(0, main(["--profile", "kitless"]))

        module_version.assert_not_called()

    def test_direct_url_commit_is_read_from_distribution_metadata(self):
        from check_environment import _distribution_vcs_commit

        class FakeDistribution:
            @staticmethod
            def read_text(name):
                if name == "direct_url.json":
                    return '{"vcs_info": {"commit_id": "abc123"}}'
                return None

        with patch("check_environment.distribution", return_value=FakeDistribution()):
            self.assertEqual("abc123", _distribution_vcs_commit("newton"))


if __name__ == "__main__":
    unittest.main()
