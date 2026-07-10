from __future__ import annotations

import unittest
from pathlib import Path

from anchor_bench.core.paths import (
    accepted_state_path,
    benchmark_run_dir,
    phase_profiles_dir,
    phase_run_dir,
    repo_root,
)


class PhasePathTests(unittest.TestCase):
    def test_phase_configuration_paths_are_robot_scoped(self):
        self.assertEqual(
            repo_root() / "configs/robots/g1/train/profiles",
            phase_profiles_dir("g1", "train"),
        )
        self.assertEqual(
            repo_root() / "configs/robots/g1/sysid/default.yaml",
            accepted_state_path("g1", "sysid"),
        )

    def test_sysid_and_training_run_paths_are_content_addressed(self):
        digest = "a" * 64
        self.assertEqual(
            repo_root() / f"outputs/train/g1/right_arm/seed-42/{digest}",
            phase_run_dir(
                "train",
                robot="g1",
                profile="right_arm",
                seed=42,
                run_input_sha256=digest,
            ),
        )

    def test_benchmark_output_root_does_not_change_identity_suffix(self):
        source = "b" * 64
        benchmark = "c" * 64
        first = benchmark_run_dir(
            output_directory=Path("one"),
            robot="g1",
            source_phase="train",
            profile="default",
            source_manifest_sha256=source,
            benchmark_input_sha256=benchmark,
        )
        second = benchmark_run_dir(
            output_directory=Path("two"),
            robot="g1",
            source_phase="train",
            profile="default",
            source_manifest_sha256=source,
            benchmark_input_sha256=benchmark,
        )

        self.assertEqual(first.parts[-5:], second.parts[-5:])

    def test_logical_names_and_hashes_are_validated(self):
        with self.assertRaises(ValueError):
            phase_profiles_dir("../g1", "train")
        with self.assertRaises(ValueError):
            phase_run_dir(
                "train",
                robot="g1",
                profile="default",
                seed=42,
                run_input_sha256="not-a-hash",
            )


if __name__ == "__main__":
    unittest.main()
