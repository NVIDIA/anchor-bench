from __future__ import annotations

import builtins
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

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
        robot_identity_fingerprint={
            "kind": "robot_yaml_identity",
            "raw_sha256": "e" * 64,
            "canonical_sha256": "f" * 64,
        },
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


def _write_sysid_axis_motion(root: Path, names: tuple[str, ...]) -> None:
    """Write one two-row SysID motion with a caller-selected three-column axis."""

    from anchor_bench.data.sage import write_sage_motion

    write_sage_motion(
        root,
        names,
        control_rows=(
            ("CONTROL", 0.0, [1.0, 2.0, 9.0]),
            ("CONTROL", 0.002, [1.1, 2.1, 9.1]),
        ),
        state_rows=(
            ("STATE_MOTOR", 0.0, [3.0, 4.0, 8.0], [0.1, 0.2, 0.3], [5.0, 6.0, 7.0]),
            ("STATE_MOTOR", 0.002, [3.1, 4.1, 8.1], [0.4, 0.5, 0.6], [5.1, 6.1, 7.1]),
        ),
        metadata={
            "time_unit": "seconds",
            "position_unit": "rad",
            "velocity_unit": "rad/s",
            "torque_unit": "N*m",
        },
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


def test_sysid_motion_canonicalizes_aliases_before_joint_selection(tmp_path: Path) -> None:
    """Accept alias-backed inputs while preserving collision and missing failures."""

    from anchor_bench.robots.schema import RobotIdentity
    from anchor_bench.sysid import runtime

    resolved = replace(
        _resolved(tmp_path),
        canonical_joint_names=("joint_a", "joint_b"),
        joint_aliases={"AliasA": "joint_a"},
    )
    spec = RobotIdentity.from_dict(
        {
            "schema_version": 1,
            "name": "demo",
            "display_name": "Demo",
            "fixed_base": True,
            "asset_path": "assets/robots/demo/manifest.yaml",
            "runtime_usd": "demo.usd",
            "joint_names": ["joint_a", "joint_b"],
            "joint_aliases": {"AliasA": "joint_a"},
        },
        "test robot",
    )
    data = tmp_path / resolved.resolved_config.data
    _write_sysid_axis_motion(data, ("AliasA", "joint_b", "context"))

    motion = runtime._load_sysid_motion(resolved, spec, target_dt=0.002)
    assert motion.joints.names == ("joint_a", "joint_b")
    np.testing.assert_allclose(motion.position.values[0], [3.0, 4.0])

    _write_sysid_axis_motion(data, ("AliasA", "joint_a", "context"))
    with pytest.raises(ValueError, match="maps columns 'AliasA' and 'joint_a'"):
        runtime._load_sysid_motion(resolved, spec, target_dt=0.002)

    _write_sysid_axis_motion(data, ("AliasA", "context", "outside"))
    with pytest.raises(ValueError, match="joints not found.*joint_b"):
        runtime._load_sysid_motion(resolved, spec, target_dt=0.002)


def test_runtime_rechecks_asset_and_dataset_fingerprints(
    tmp_path: Path,
) -> None:
    """Reject robot, package, or dataset changes before simulation."""

    from anchor_bench.core.hashing import sha256_tree
    from anchor_bench.robots.fingerprints import (
        load_robot_identity_fingerprint,
        runtime_asset_fingerprint,
    )
    from anchor_bench.sysid import runtime

    resolved = _resolved(tmp_path)
    dataset = tmp_path / resolved.resolved_config.data
    dataset.mkdir(parents=True)
    data_file = dataset / "episode.csv"
    data_file.write_text("time,value\n0,0\n", encoding="utf-8")
    config_dir = tmp_path / "configs/robots/demo"
    robot_path = config_dir / "robot.yaml"
    robot_document = {
        "schema_version": 1,
        "name": "demo",
        "display_name": "Demo",
        "fixed_base": True,
        "asset_path": "assets/robots/demo/manifest.yaml",
        "runtime_usd": "demo.usd",
        "joint_names": ["joint_a", "joint_b"],
        "joint_aliases": {"A": "joint_a"},
    }
    robot_path.parent.mkdir(parents=True, exist_ok=True)
    robot_path.write_text(yaml.safe_dump(robot_document, sort_keys=False), encoding="utf-8")
    robot_bytes = robot_path.read_bytes()
    manifest = tmp_path / "assets/robots/demo/manifest.yaml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("name: demo\n", encoding="utf-8")
    usd = manifest.parent / "files/demo.usd"
    usd.parent.mkdir()
    usd.write_bytes(b"usd-v1")
    dependency = usd.parent / "mesh.bin"
    dependency.write_bytes(b"mesh-v1")
    _spec, robot_fingerprint = load_robot_identity_fingerprint(
        robot_path,
        expected_name="demo",
        repository_root=tmp_path,
    )
    resolved = replace(
        resolved,
        fingerprints={
            **resolved.fingerprints,
            "robot": robot_fingerprint,
            "asset": runtime_asset_fingerprint(usd, runtime_usd="demo.usd"),
            "dataset": {"tree_sha256": sha256_tree(dataset)},
        },
    )

    runtime._verify_runtime_inputs(resolved, usd)
    dependency.write_bytes(b"mesh-v2")
    with pytest.raises(RuntimeError, match="asset package changed after resolution"):
        runtime._verify_runtime_inputs(resolved, usd)

    dependency.write_bytes(b"mesh-v1")
    robot_document["fixed_base"] = False
    robot_path.write_text(yaml.safe_dump(robot_document, sort_keys=False), encoding="utf-8")
    with pytest.raises(RuntimeError, match="robot identity changed after resolution"):
        runtime._verify_runtime_inputs(resolved, usd)

    robot_path.write_bytes(robot_bytes)
    usd.write_bytes(b"usd-v2")
    with pytest.raises(RuntimeError, match="asset package changed after resolution"):
        runtime._verify_runtime_inputs(resolved, usd)

    usd.write_bytes(b"usd-v1")
    data_file.write_text("time,value\n0,1\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="dataset changed after resolution"):
        runtime._verify_runtime_inputs(resolved, usd)


class _FakeTensor:
    """Small NumPy-backed tensor implementing reset-path operations only."""

    def __init__(self, values) -> None:
        """Copy finite test values into a mutable array."""

        self.values = np.asarray(values, dtype=float).copy()

    def __getitem__(self, key):
        """Return one copied tensor slice."""

        return _FakeTensor(self.values[key])

    def __setitem__(self, key, value) -> None:
        """Assign a tensor or array-like slice."""

        self.values[key] = value.values if isinstance(value, _FakeTensor) else value

    def zero_(self):
        """Clear this tensor in place."""

        self.values.fill(0.0)
        return self

    def unsqueeze(self, axis: int):
        """Insert one singleton dimension."""

        return _FakeTensor(np.expand_dims(self.values, axis=axis))

    def expand(self, *shape: int):
        """Broadcast into the requested Torch-like shape."""

        resolved = tuple(
            current if requested == -1 else requested
            for current, requested in zip(self.values.shape, shape, strict=True)
        )
        return _FakeTensor(np.broadcast_to(self.values, resolved))


def test_sysid_rollout_uses_command_t_and_measurement_t_plus_one() -> None:
    """Score exactly T-1 intervals without command-T or measurement-zero leakage."""

    from anchor_bench.sysid import runtime

    adapter = object.__new__(runtime.IsaacNewtonRuntimeAdapter)
    adapter._row_count = 4
    events = []
    adapter._set_command_row = lambda row: events.append(("command", row))
    adapter._advance_control_interval = lambda: events.append(("advance", None))

    def accumulate(measured_row, position_error, velocity_error):
        """Record the compared row and expose error accumulator flow."""

        events.append(("measure", measured_row))
        return position_error + measured_row, velocity_error + 10 * measured_row

    adapter._accumulate_measurement_error = accumulate
    position_error, velocity_error = adapter._rollout_errors(0, 0)

    assert events == [
        ("command", 0),
        ("advance", None),
        ("measure", 1),
        ("command", 1),
        ("advance", None),
        ("measure", 2),
        ("command", 2),
        ("advance", None),
        ("measure", 3),
    ]
    assert position_error == 6
    assert velocity_error == 60
    assert runtime._rollout_sample_count(4, 2) == 6


def test_sysid_settle_restores_exact_measured_row_zero_without_stepping() -> None:
    """Warm the configured buffer, then make the final operation an exact restore."""

    from anchor_bench.sysid import runtime

    events: list[str] = []
    states = []
    targets = []

    def write_state(position, velocity) -> None:
        """Capture exact state values and their place in reset order."""

        states.append((position.values.copy(), velocity.values.copy()))
        events.append("state")

    def set_target(target) -> None:
        """Capture the full-layout warm-up target."""

        targets.append(target.values.copy())
        events.append("target")

    adapter = object.__new__(runtime.IsaacNewtonRuntimeAdapter)
    adapter._resolved = SimpleNamespace(
        resolved_config=SimpleNamespace(num_envs=2, buffer_time=0.004)
    )
    adapter._timing = SimpleNamespace(
        control_dt=0.002,
        control_divisor=2,
        physics_dt=0.001,
    )
    adapter._joint_ids_long = np.asarray([1, 3], dtype=int)
    adapter._measured_position = _FakeTensor([[10.0, 20.0], [11.0, 21.0]])
    adapter._measured_velocity = _FakeTensor([[1.0, 2.0], [1.1, 2.1]])
    adapter._commands = _FakeTensor([[0.5, 0.6], [0.7, 0.8]])
    adapter._joint_position = _FakeTensor(np.full((2, 4), 99.0))
    adapter._joint_velocity = _FakeTensor(np.full((2, 4), 88.0))
    adapter._target = _FakeTensor(np.full((2, 4), 77.0))
    adapter._robot = SimpleNamespace(
        write_joint_state_to_sim=write_state,
        set_joint_position_target=set_target,
    )
    adapter._scene = SimpleNamespace(
        write_data_to_sim=lambda: events.append("write")
    )
    adapter._advance_physics_step = lambda: events.append("step")

    adapter._reset_trajectory()

    expected_position = np.asarray([[0, 10, 0, 20], [0, 10, 0, 20]], dtype=float)
    expected_velocity = np.asarray([[0, 1, 0, 2], [0, 1, 0, 2]], dtype=float)
    for position, velocity in states:
        np.testing.assert_allclose(position, expected_position)
        np.testing.assert_allclose(velocity, expected_velocity)
    np.testing.assert_allclose(
        targets[0],
        [[0, 0.5, 0, 0.6], [0, 0.5, 0, 0.6]],
    )
    assert events == [
        "state",
        "write",
        "target",
        "step",
        "step",
        "step",
        "step",
        "state",
        "write",
    ]


def test_sysid_rollout_requires_two_synchronized_rows() -> None:
    """Reject a one-row or mismatched command/state rollout before simulation."""

    from anchor_bench.sysid import runtime

    one_row = np.asarray([0.0])
    motion = SimpleNamespace(
        command_position=SimpleNamespace(time=one_row),
        position=SimpleNamespace(time=one_row),
        velocity=SimpleNamespace(time=one_row),
    )
    with pytest.raises(ValueError, match="at least 2 synchronized"):
        runtime._synchronized_rollout_rows(motion)

    motion.command_position.time = np.asarray([0.0, 0.003])
    motion.position.time = np.asarray([0.0, 0.002])
    motion.velocity.time = np.asarray([0.0, 0.002])
    with pytest.raises(ValueError, match="requires synchronized"):
        runtime._synchronized_rollout_rows(motion)
