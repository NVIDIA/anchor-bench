import os
import subprocess
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def run_setup(
    *args: str,
    dry_run: bool = False,
    env_overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    if dry_run:
        env["ANCHOR_BENCH_SETUP_DRY_RUN"] = "1"
    if env_overrides:
        env.update(env_overrides)
    return subprocess.run(
        ["./setup.sh", *args],
        cwd=REPO,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )


class DockerSetupTests(unittest.TestCase):
    def test_setup_help_describes_docker_first_modes(self):
        result = run_setup("--help")
        self.assertIn("Docker-first", result.stdout)
        self.assertIn("Usage: ./setup.sh", result.stdout)
        self.assertIn("--kitless", result.stdout)
        self.assertIn("--full", result.stdout)
        self.assertIn("--shell", result.stdout)
        self.assertIn("does not create a host .venv", result.stdout)

    def test_kitless_dry_run_builds_docker_image_without_host_venv(self):
        venv_existed = (REPO / ".venv").exists()
        result = run_setup("--kitless", dry_run=True)
        self.assertIn("docker build", result.stdout)
        self.assertIn("--target kitless", result.stdout)
        self.assertIn("anchor-bench:kitless", result.stdout)
        self.assertIn("--build-arg NEWTON_VERSION=1.2.0", result.stdout)
        self.assertIn("--build-arg ISAACSIM_VERSION=6.0.0.0", result.stdout)
        self.assertIn(
            "--build-arg ISAACLAB_REF=482d21c936b863e323f0a135c8e8e8e8a9c166b2",
            result.stdout,
        )
        self.assertNotIn("-m venv", result.stdout)
        self.assertEqual(venv_existed, (REPO / ".venv").exists())

    def test_check_dry_run_runs_environment_check_inside_kitless_image(self):
        result = run_setup("--check", dry_run=True)
        self.assertIn("docker image inspect anchor-bench:kitless", result.stdout)
        self.assertIn("python check_environment.py --dry-run", result.stdout)

    def test_shell_dry_run_delegates_to_internal_persistent_helper(self):
        result = run_setup("--shell", dry_run=True)
        self.assertIn("docker/run-persist.sh", result.stdout)
        self.assertNotIn("anchor-bench:kitless", result.stdout)

    def test_shell_dry_run_prints_persistent_helper_without_invoking_it(self):
        result = run_setup("--shell", dry_run=True)
        self.assertNotIn("python check_environment.py", result.stdout)
        self.assertNotIn("Container anchor-bench-persist loaded properly", result.stdout)
        self.assertNotIn("Running environment check", result.stdout)
        self.assertNotIn("docker exec", result.stdout)

    def test_shell_third_party_isaaclab_source_dry_run_bootstraps_local_checkout(self):
        result = run_setup(
            "--shell",
            "anchor-bench",
            "robots",
            "list",
            dry_run=True,
            env_overrides={"ANCHOR_BENCH_ISAACLAB_SOURCE": "third_party"},
        )

        self.assertIn("Preparing third_party/IsaacLab", result.stdout)
        self.assertIn("third_party/IsaacLab", result.stdout)
        self.assertIn("git clone --branch develop --single-branch", result.stdout)
        self.assertIn("https://github.com/isaac-sim/IsaacLab.git", result.stdout)
        isaaclab_dir = REPO / "third_party" / "IsaacLab"
        self.assertIn(
            f"git -C {isaaclab_dir} fetch origin 482d21c936b863e323f0a135c8e8e8e8a9c166b2",
            result.stdout,
        )
        self.assertIn(
            f"git -C {isaaclab_dir} switch --detach 482d21c936b863e323f0a135c8e8e8e8a9c166b2",
            result.stdout,
        )
        self.assertIn(
            "ANCHOR_BENCH_ISAACLAB_DIR=/workspace/anchor-bench/third_party/IsaacLab",
            result.stdout,
        )
        self.assertIn("docker/run-persist.sh", result.stdout)

    def test_persistent_shell_check_has_quiet_success_output(self):
        run_persist_text = (REPO / "docker" / "run-persist.sh").read_text(encoding="utf-8")
        self.assertNotIn("Running environment check", run_persist_text)

    def test_readme_uses_setup_shell_as_public_persistent_interface(self):
        readme = (REPO / "README.md").read_text(encoding="utf-8")
        cli = (REPO / "docs" / "cli.md").read_text(encoding="utf-8")
        self.assertIn("./setup.sh --full", readme)
        self.assertIn("| Command | Purpose |", readme)
        self.assertIn("`./setup.sh --shell`", readme)
        self.assertIn("host UID/GID", readme)
        self.assertIn("[CLI commands](docs/cli.md)", readme)
        self.assertNotIn("ANCHOR_BENCH_ISAACLAB_SOURCE=third_party", readme)
        self.assertNotIn("ANCHOR_BENCH_RUN_AS_ROOT=1", readme)
        self.assertIn("ANCHOR_BENCH_ISAACLAB_SOURCE=third_party", cli)
        self.assertIn("ANCHOR_BENCH_RUN_AS_ROOT=1", cli)
        self.assertIn("/opt/IsaacLab", cli)
        self.assertIn("`anchor-bench --help`", readme)
        self.assertIn("anchor-bench robots list", readme)
        self.assertIn("./docker/run-gui.sh", readme)
        self.assertIn("./docker/run-headless.sh", readme)
        self.assertNotIn("Executables per robot/dataset", readme)
        self.assertNotIn("Single instance of Docker containers", readme)
        self.assertNotIn("bash setup.sh", readme)
        self.assertNotIn("./docker/run-persist.sh", readme)

    def test_readme_links_to_reference_docs(self):
        readme = (REPO / "README.md").read_text(encoding="utf-8")

        self.assertIn("[CLI commands](docs/cli.md)", readme)
        self.assertIn("[repo structure](docs/repo-structure.md)", readme)
        self.assertIn("[Artifacts](docs/artifacts.md)", readme)
        self.assertIn("[Robot plugins](docs/robot-plugin.md)", readme)
        self.assertIn("run.yaml", readme)
        self.assertIn("prepare_huggingface_parquets.py", readme)

    def test_reference_docs_document_inputs_outputs_and_config_boundary(self):
        repo_structure = (REPO / "docs" / "repo-structure.md").read_text(encoding="utf-8")
        cli = (REPO / "docs" / "cli.md").read_text(encoding="utf-8")

        self.assertIn("## Configs", repo_structure)
        self.assertIn("## Assets", repo_structure)
        self.assertIn("ANCHOR_BENCH_ASSET_DIR", repo_structure)
        self.assertIn("Do not put robot USDs", repo_structure)
        self.assertIn("## Inputs", repo_structure)
        self.assertIn("## Outputs", repo_structure)
        self.assertIn("optional vendor checkouts", repo_structure)
        self.assertIn("container image uses /opt/IsaacLab", repo_structure)
        self.assertIn("inputs/datasets/", repo_structure)
        self.assertIn("inputs/actuator_params/", repo_structure)
        self.assertIn("configs/robots/<robot>/actuators.yaml", repo_structure)
        self.assertIn("outputs/sysid/.../best_params.yaml", repo_structure)
        self.assertIn("anchor-bench robots list", cli)
        self.assertIn("anchor-bench experiment benchmark", cli)

    def test_third_party_artifact_layout_is_documented(self):
        gitignore = (REPO / ".gitignore").read_text(encoding="utf-8")
        dockerignore = (REPO / ".dockerignore").read_text(encoding="utf-8")
        setup_text = (REPO / "setup.sh").read_text(encoding="utf-8")
        check_environment = (REPO / "check_environment.py").read_text(encoding="utf-8")
        teststand_actuators = (REPO / "configs" / "robots" / "teststand" / "actuators.yaml").read_text(encoding="utf-8")

        self.assertTrue((REPO / "third_party" / "README.md").exists())
        self.assertIn("third_party/artifacts/h1/nvidia-motor.fmu", teststand_actuators)
        self.assertIn("third_party/artifacts/h1", setup_text)
        self.assertIn("third_party", dockerignore)
        self.assertIn("third_party/*", gitignore)
        self.assertIn("repo_root() / \"third_party\" / \"IsaacLab\"", check_environment)
        self.assertNotIn("external" + "/artifacts", teststand_actuators)
        self.assertNotIn("repo_root() / \"external\"", check_environment)

    def test_required_docker_assets_are_present(self):
        dockerfile = REPO / "docker" / "Dockerfile"
        compose = REPO / "docker-compose.yaml"
        entrypoint = REPO / "docker" / "entrypoint.sh"
        dockerignore = REPO / ".dockerignore"
        run_gui = REPO / "docker" / "run-gui.sh"
        run_headless = REPO / "docker" / "run-headless.sh"
        run_persist = REPO / "docker" / "run-persist.sh"
        x11 = REPO / "docker" / "x11.yaml"

        for path in (dockerfile, compose, entrypoint, dockerignore, run_gui, run_headless, run_persist, x11):
            self.assertTrue(path.exists(), f"{path} should exist")
        for path in (run_gui, run_headless, run_persist):
            self.assertTrue(os.access(path, os.X_OK), f"{path} should be executable")

        dockerfile_text = dockerfile.read_text(encoding="utf-8")
        self.assertIn("AS kitless", dockerfile_text)
        self.assertIn("AS full", dockerfile_text)
        self.assertIn("ARG ISAACSIM_VERSION=6.0.0.0", dockerfile_text)
        self.assertIn("isaacsim[all,extscache]==${ISAACSIM_VERSION}", dockerfile_text)
        self.assertIn("--extra-index-url https://download.pytorch.org/whl/cu128", dockerfile_text)
        self.assertIn("--extra-index-url https://pypi.nvidia.com", dockerfile_text)
        self.assertIn("--editable source/isaaclab", dockerfile_text)
        self.assertIn("source/isaaclab_newton", dockerfile_text)
        self.assertIn("${NEWTON_PACKAGE}[sim]==${NEWTON_VERSION}", dockerfile_text)
        self.assertIn("ACCEPT_EULA=Y", dockerfile_text)
        self.assertIn("OMNI_KIT_ACCEPT_EULA=YES", dockerfile_text)
        self.assertIn("libxt6", dockerfile_text)
        self.assertIn("libglu1-mesa", dockerfile_text)
        self.assertIn("482d21c936b863e323f0a135c8e8e8e8a9c166b2", dockerfile_text)

        compose_text = compose.read_text(encoding="utf-8")
        self.assertIn("/workspace/anchor-bench", compose_text)
        self.assertIn("ACCEPT_EULA: Y", compose_text)
        self.assertIn("OMNI_KIT_ACCEPT_EULA: YES", compose_text)
        self.assertIn("NVIDIA_DRIVER_CAPABILITIES", compose_text)
        self.assertIn("ANCHOR_BENCH_ASSET_DIR", compose_text)

        run_gui_text = run_gui.read_text(encoding="utf-8")
        self.assertIn("ANCHOR_BENCH_DOCKER_TARGET=full", run_gui_text)
        self.assertIn("ANCHOR_BENCH_IMAGE=", run_gui_text)
        self.assertIn("x11.yaml", run_gui_text)
        self.assertIn("ANCHOR_BENCH_DATA_DIR", run_gui_text)

        run_headless_text = run_headless.read_text(encoding="utf-8")
        self.assertIn("HEADLESS=1", run_headless_text)
        self.assertIn("run-gui.sh", run_headless_text)

        run_persist_text = run_persist.read_text(encoding="utf-8")
        self.assertIn("Internal helper for ./setup.sh --shell", run_persist_text)
        self.assertIn("./setup.sh --shell anchor-bench benchmark", run_persist_text)
        self.assertIn("CONTAINER=anchor-bench-persist", run_persist_text)
        self.assertIn("IMAGE=", run_persist_text)
        self.assertIn("anchor-bench:full", run_persist_text)
        self.assertIn("ACCEPT_EULA=Y", run_persist_text)
        self.assertIn("OMNI_KIT_ACCEPT_EULA=YES", run_persist_text)
        self.assertIn("ANCHOR_BENCH_DATA_DIR", run_persist_text)
        self.assertIn("/workspace/anchor-bench", run_persist_text)
        self.assertIn("PYTHONPATH=/workspace/anchor-bench:${PYTHONPATH:-}", run_persist_text)
        self.assertIn("ANCHOR_BENCH_EXTRA_PYTHONPATH", run_persist_text)
        self.assertIn("source/isaaclab_visualizers", run_persist_text)

        x11_text = x11.read_text(encoding="utf-8")
        self.assertIn("anchor-bench:", x11_text)
        self.assertIn("XAUTHORITY", x11_text)

    def test_docker_cache_defaults_use_non_root_anchor_home(self):
        dockerfile_text = (REPO / "docker" / "Dockerfile").read_text(encoding="utf-8")
        compose_text = (REPO / "docker-compose.yaml").read_text(encoding="utf-8")
        run_persist_text = (REPO / "docker" / "run-persist.sh").read_text(encoding="utf-8")

        self.assertIn("HOME=/home/anchor", dockerfile_text)
        self.assertIn("XDG_CACHE_HOME=/home/anchor/.cache", dockerfile_text)
        self.assertIn("target: /home/anchor/.cache/pip", compose_text)
        self.assertNotIn("target: /root/.cache/pip", compose_text)
        self.assertIn("-v anchor-bench-cache-pip:/home/anchor/.cache/pip", run_persist_text)
        self.assertIn("-v anchor-bench-cache-compute:/home/anchor/.nv/ComputeCache", run_persist_text)
        self.assertIn("-v anchor-bench-docs:/home/anchor/Documents", run_persist_text)
        self.assertNotIn("/root/.cache", run_persist_text)
        self.assertNotIn("/root/.nv", run_persist_text)
        self.assertNotIn("/root/Documents", run_persist_text)

    def test_container_wrappers_default_to_host_uid_gid_for_bind_mount_writes(self):
        entrypoint_text = (REPO / "docker" / "entrypoint.sh").read_text(encoding="utf-8")
        check_environment = (REPO / "check_environment.py").read_text(encoding="utf-8")
        run_gui_text = (REPO / "docker" / "run-gui.sh").read_text(encoding="utf-8")
        run_persist_text = (REPO / "docker" / "run-persist.sh").read_text(encoding="utf-8")

        self.assertIn('"git", "-c", f"safe.directory={lab_dir}", "-C"', check_environment)
        self.assertIn('mkdir -p "${HOME}" "${XDG_CACHE_HOME}" "${MPLCONFIGDIR}"', entrypoint_text)
        self.assertIn('mkdir -p "${HOME}" "${XDG_CACHE_HOME}" "${MPLCONFIGDIR}" && sleep infinity', run_persist_text)
        for script_text in (run_gui_text, run_persist_text):
            self.assertIn("HOST_UID=\"$(id -u)\"", script_text)
            self.assertIn("HOST_GID=\"$(id -g)\"", script_text)
            self.assertIn("--user \"${HOST_UID}:${HOST_GID}\"", script_text)
            self.assertIn("-e HOME=/tmp/anchor-bench-home", script_text)
            self.assertIn("-e XDG_CACHE_HOME=/tmp/anchor-bench-cache", script_text)
        self.assertIn("CONTAINER_IDENTITY_NAME=anchor-bench", run_persist_text)
        self.assertIn("prepare_container_identity_files", run_persist_text)
        self.assertIn("Anchor Bench Host User:/tmp/anchor-bench-home:/bin/bash", run_persist_text)
        self.assertIn("-e USER=\"${CONTAINER_IDENTITY_NAME}\"", run_persist_text)
        self.assertIn("-v \"${CONTAINER_PASSWD_PATH}:/etc/passwd:ro\"", run_persist_text)
        self.assertIn("-v \"${CONTAINER_GROUP_PATH}:/etc/group:ro\"", run_persist_text)


if __name__ == "__main__":
    unittest.main()
