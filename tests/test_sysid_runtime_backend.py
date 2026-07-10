from __future__ import annotations

import builtins
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml


def _resolved(tmp_path: Path):
    """Resolve one host-only recipe with explicit per-joint and shared fields."""

    from anchor_bench.sysid.runner import (
        SysIdInvocation,
        SysIdResolverInputs,
        resolve_sysid_run,
    )

    profile = {
        "schema_version": 1,
        "data": "inputs/demo",
        "joints": ["joint_a", "joint_b"],
        "simulation": {
            "physics_freq": 1000,
            "control_freq": 500,
            "render_freq": 500,
            "integrator": "implicitfast",
            "solver_iterations": 7,
            "line_search_iterations": 3,
        },
        "num_envs": 2,
        "buffer_time": 0.25,
        "seed": 42,
        "optimizer": {
            "type": "cmaes",
            "sigma": 0.5,
            "max_iterations": 1,
            "epsilon": 0.001,
        },
        "parameters": {
            "stiffness": {
                "mode": "per_joint",
                "bounds": {"lower": 0.0, "upper": 100.0},
            },
            "damping": {
                "mode": "shared",
                "bounds": {"lower": 0.0, "upper": 10.0},
            },
        },
        "loss": {"position": 2.0, "velocity": 0.5},
    }
    accepted = {
        "schema_version": 1,
        "revision": 0,
        "parameters": {
            "stiffness": {"joint_a": 10.0, "joint_b": 20.0},
            "damping": {"joint_a": 1.0, "joint_b": 3.0},
            "armature": {"joint_a": 0.01, "joint_b": 0.02},
        },
        "provenance": {"history": []},
    }
    profiles = tmp_path / "configs/robots/demo/sysid/profiles"
    profiles.mkdir(parents=True)
    (profiles / "default.yaml").write_text(
        yaml.safe_dump(profile, sort_keys=False),
        encoding="utf-8",
    )
    accepted_path = profiles.parent / "default.yaml"
    accepted_path.write_text(
        yaml.safe_dump(accepted, sort_keys=False),
        encoding="utf-8",
    )

    def run_directory(_phase: str, **kwargs) -> Path:
        """Place the deterministic output below the test repository."""

        return tmp_path / "outputs" / kwargs["run_input_sha256"]

    inputs = SysIdResolverInputs(
        repository_root=tmp_path,
        profiles_directory=profiles,
        accepted_state=accepted_path,
        canonical_joint_names=("joint_a", "joint_b"),
        joint_aliases={},
        asset_fingerprint={"sha256": "a" * 64},
        source_fingerprint={"git": "b" * 40},
        dependency_fingerprints={"newton": "1.4.0.dev0"},
        dataset_fingerprint={"tree_sha256": "d" * 64},
        run_directory_resolver=run_directory,
    )
    return resolve_sysid_run(
        SysIdInvocation(robot="demo", dry_run=False),
        inputs,
    )


class _FakeOptimizer:
    """Deterministic two-candidate optimizer used to observe weighted scores."""

    def __init__(self) -> None:
        """Initialize one explicit population and empty result state."""

        self.population = [
            {
                "stiffness": {"joint_a": 12.0, "joint_b": 22.0},
                "damping": {"joint_a": 2.0, "joint_b": 2.0},
            },
            {
                "stiffness": {"joint_a": 30.0, "joint_b": 40.0},
                "damping": {"joint_a": 4.0, "joint_b": 4.0},
            },
        ]
        self.best_params = None
        self.best_score = float("inf")
        self.generation = 0
        self.scores: np.ndarray | None = None

    @property
    def normalized_mean(self) -> list[float]:
        """Expose deterministic parent-derived evidence for serialization."""

        return [-0.8, -0.6, -0.6]

    def sample_population(self):
        """Return the configured sparse fitted patches."""

        return self.population

    def tell(self, scores) -> bool:
        """Record scores and select their exact minimum."""

        self.scores = np.asarray(scores, dtype=np.float64)
        index = int(np.argmin(self.scores))
        self.best_score = float(self.scores[index])
        self.best_params = self.population[index]
        self.generation += 1
        return False


class _FakeAdapter:
    """Capture complete parent-overlaid states and return fixed components."""

    def __init__(self) -> None:
        """Initialize capture and lifecycle state."""

        self.population = None
        self.closed = False

    def evaluate_population(self, population):
        """Return errors whose ranking depends on both configured weights."""

        from anchor_bench.sysid.runtime import PopulationEvaluation

        self.population = population
        return PopulationEvaluation(
            position_mse=[1.0, 4.0],
            velocity_mse=[9.0, 1.0],
        )

    def close(self) -> None:
        """Record deterministic resource cleanup."""

        self.closed = True


def test_resolved_backend_uses_parent_layout_both_losses_and_returns_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the backend without importing or writing through the simulator."""

    from anchor_bench.sysid.runtime import optimize_resolved_sysid

    resolved = _resolved(tmp_path)
    optimizer = _FakeOptimizer()
    adapter = _FakeAdapter()
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        """Fail if the injected host path attempts a full-runtime import."""

        if name.split(".", 1)[0] in {
            "isaaclab",
            "isaaclab_newton",
            "newton",
            "torch",
            "warp",
        }:
            raise AssertionError(f"unexpected simulator import: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    def adapter_factory(received):
        """Assert the adapter receives the complete resolved simulation recipe."""

        assert received is resolved
        assert received.resolved_config.simulation.solver_iterations == 7
        assert received.resolved_config.simulation.line_search_iterations == 3
        assert received.resolved_config.buffer_time == 0.25
        return adapter

    result = optimize_resolved_sysid(
        resolved,
        adapter_factory=adapter_factory,
        optimizer_factory=lambda received: optimizer,
    )

    assert adapter.closed is True
    assert optimizer.scores == pytest.approx([6.5, 8.5])
    assert result.fitted_parameters == optimizer.population[0]
    assert adapter.population[0]["armature"] == {
        "joint_a": 0.01,
        "joint_b": 0.02,
    }
    assert adapter.population[0]["stiffness"] == {
        "joint_a": 12.0,
        "joint_b": 22.0,
    }
    assert adapter.population[0]["damping"] == {
        "joint_a": 2.0,
        "joint_b": 2.0,
    }
    assert result.metrics == {
        "best_loss": 6.5,
        "best_position_mse": 1.0,
        "best_velocity_mse": 9.0,
        "generations_completed": 1,
        "converged": False,
        "num_envs": 2,
        "loss_weights": {"position": 2.0, "velocity": 0.5},
    }
    assert set(result.optimizer_artifacts) == {"history.csv", "optimizer.json"}
    optimizer_evidence = json.loads(result.optimizer_artifacts["optimizer.json"])
    assert optimizer_evidence["parameter_layout"] == [
        "stiffness/joint_a",
        "stiffness/joint_b",
        "damping/shared",
    ]
    assert optimizer_evidence["normalized_parent_mean"] == [-0.8, -0.6, -0.6]
    assert result.optimizer_artifacts["history.csv"].splitlines()[1] == (
        "0,6.5,7.5,1,9"
    )
    assert not resolved.run_dir.exists()


@pytest.mark.parametrize(
    ("position", "velocity", "message"),
    [
        ([1.0], [1.0, 1.0], "position_mse must have shape"),
        ([1.0, 2.0], [1.0, float("nan")], "velocity_mse must contain only finite"),
        ([1.0, -2.0], [1.0, 1.0], "position_mse must contain only non-negative"),
    ],
)
def test_backend_rejects_invalid_simulator_components_and_closes(
    tmp_path: Path,
    position: list[float],
    velocity: list[float],
    message: str,
) -> None:
    """Reject malformed simulator output while still closing its resources."""

    from anchor_bench.sysid.runtime import (
        PopulationEvaluation,
        optimize_resolved_sysid,
    )

    resolved = _resolved(tmp_path)
    optimizer = _FakeOptimizer()

    class InvalidAdapter(_FakeAdapter):
        """Return one test-provided invalid component pair."""

        def evaluate_population(self, population):
            """Capture the population before returning invalid values."""

            self.population = population
            return PopulationEvaluation(position, velocity)

    adapter = InvalidAdapter()
    with pytest.raises(ValueError, match=message):
        optimize_resolved_sysid(
            resolved,
            adapter_factory=lambda received: adapter,
            optimizer_factory=lambda received: optimizer,
        )
    assert adapter.closed is True


def test_backend_rejects_shared_parameter_disagreement_before_simulation(
    tmp_path: Path,
) -> None:
    """Enforce the profile's shared-variable expansion at the adapter boundary."""

    from anchor_bench.sysid.runtime import optimize_resolved_sysid

    resolved = _resolved(tmp_path)
    optimizer = _FakeOptimizer()
    optimizer.population[0]["damping"]["joint_b"] = 3.0
    adapter = _FakeAdapter()

    with pytest.raises(ValueError, match="shared fitted parameter 'damping'"):
        optimize_resolved_sysid(
            resolved,
            adapter_factory=lambda received: adapter,
            optimizer_factory=lambda received: optimizer,
        )
    assert adapter.population is None
    assert adapter.closed is True


def test_robot_identity_loader_ignores_legacy_phase_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read robot.yaml even when legacy runtime configuration is unusable."""

    from anchor_bench.sysid import runtime

    config_dir = tmp_path / "configs/robots/demo"
    config_dir.mkdir(parents=True)
    (config_dir / "robot.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "demo",
                "display_name": "Demo",
                "fixed_base": True,
                "asset_path": "assets/robots/demo/manifest.yaml",
                "runtime_usd": "demo.usd",
                "joint_names": ["joint_a", "joint_b"],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    for name in ("run.yaml", "bounds.yaml", "actuators.yaml"):
        (config_dir / name).write_text("not: [valid", encoding="utf-8")
    monkeypatch.setattr(runtime, "robot_config_dir", lambda robot: config_dir)

    spec = runtime._load_robot_identity("demo")

    assert spec.name == "demo"
    assert spec.joint_names == ("joint_a", "joint_b")
    assert spec.runtime_usd == "demo.usd"


def test_runtime_rechecks_asset_and_dataset_fingerprints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject input bytes that change between resolution and simulation."""

    from anchor_bench.core.hashing import sha256_file, sha256_tree
    from anchor_bench.sysid import runtime

    resolved = _resolved(tmp_path)
    dataset = tmp_path / resolved.resolved_config.data
    dataset.mkdir(parents=True)
    data_file = dataset / "episode.csv"
    data_file.write_text("time,value\n0,0\n", encoding="utf-8")
    usd = tmp_path / "demo.usd"
    usd.write_bytes(b"usd-v1")
    resolved = replace(
        resolved,
        fingerprints={
            **resolved.fingerprints,
            "asset": {"asset_sha256": sha256_file(usd)},
            "dataset": {"tree_sha256": sha256_tree(dataset)},
        },
    )
    monkeypatch.setattr(runtime, "repo_root", lambda: tmp_path)

    runtime._verify_runtime_inputs(resolved, usd)
    usd.write_bytes(b"usd-v2")
    with pytest.raises(RuntimeError, match="asset changed after resolution"):
        runtime._verify_runtime_inputs(resolved, usd)

    usd.write_bytes(b"usd-v1")
    data_file.write_text("time,value\n0,1\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="dataset changed after resolution"):
        runtime._verify_runtime_inputs(resolved, usd)
