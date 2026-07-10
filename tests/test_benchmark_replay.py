"""Host-safe behavioral tests for candidate benchmark replay."""

from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import yaml

from anchor_bench.benchmark.phase import (
    BenchmarkMotionSource,
    SysIdBenchmarkCandidate,
    TrainingBenchmarkCandidate,
)
from anchor_bench.benchmark.replay import (
    ReplayState,
    candidate_joint_overrides,
    run_benchmark_replay,
)
from anchor_bench.benchmark.schema import BenchmarkProfile
from anchor_bench.core.hashing import tree_fingerprint
from anchor_bench.training.metadata import RuntimeGRUMetadata


SHA_A = "a" * 64
SHA_B = "b" * 64


def _write_motion(
    root: Path,
    *,
    joints: tuple[str, ...] = ("joint_a", "joint_b"),
    command: tuple[tuple[float, ...], ...] = ((0.0, 10.0), (2.0, 14.0)),
    position: tuple[tuple[float, ...], ...] = ((0.0, 10.0), (1.0, 12.0)),
    velocity: tuple[tuple[float, ...], ...] = ((0.0, 1.0), (0.2, 1.4)),
    timestamps: tuple[float, ...] = (0.0, 0.02),
) -> None:
    """Write a minimal SAGE motion whose intermediate row requires interpolation."""

    root.mkdir(parents=True)
    (root / "joint_list.txt").write_text("\n".join(joints) + "\n", encoding="utf-8")
    with (root / "control.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["type", "timestamp", "positions"])
        for timestamp, values in zip(timestamps, command, strict=True):
            writer.writerow(["CONTROL", timestamp, list(values)])
    with (root / "state_motor.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["type", "timestamp", "positions", "velocities"])
        for timestamp, positions, velocities in zip(
            timestamps, position, velocity, strict=True
        ):
            writer.writerow(["STATE_MOTOR", timestamp, list(positions), list(velocities)])
    (root / "metadata.yaml").write_text(
        yaml.safe_dump(
            {
                "time_unit": "seconds",
                "position_unit": "rad",
                "velocity_unit": "rad/s",
                "torque_unit": "N*m",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _motion_source(path: Path, name: str = "reach") -> BenchmarkMotionSource:
    """Build a resolved motion reference for a test SAGE directory."""

    return BenchmarkMotionSource(
        name=name,
        path=path,
        repository_path=f"inputs/{name}",
        joint_names=("joint_a", "joint_b"),
        canonical_joint_names=("joint_a", "joint_b"),
        fingerprint=tree_fingerprint(path),
    )


def _profile(*, signals: tuple[str, ...] = ("position", "velocity")) -> BenchmarkProfile:
    """Build one 100 Hz benchmark profile with initial-state synchronization."""

    return BenchmarkProfile.from_dict(
        {
            "schema_version": 1,
            "data": "inputs",
            "motions": ["reach"],
            "simulation": {
                "physics_freq": 500,
                "control_freq": 100,
                "render_freq": 100,
                "integrator": "implicitfast",
                "solver_iterations": 10,
                "line_search_iterations": 5,
            },
            "num_envs": 1,
            "buffer_time": 0.0,
            "sync_initial_state": True,
            "signals": list(signals),
            "output_root": "outputs/benchmark",
        },
        "test benchmark profile",
    )


def _sysid_candidate() -> SysIdBenchmarkCandidate:
    """Build one reconstructed scalar SysID candidate with complete physics."""

    state = SimpleNamespace(
        parameters={
            "stiffness": {"joint_a": 20.0},
            "damping": {"joint_a": 2.0},
            "dry_friction": {"joint_a": 0.3},
            "viscous_friction": {"joint_a": 0.4},
            "armature": {"joint_a": 0.05},
        }
    )
    return SysIdBenchmarkCandidate(
        loaded=SimpleNamespace(state=state),
        selected_joints=("joint_a",),
        control_freq=100,
        artifact_hashes={"sysid_state": SHA_A},
    )


def _resolved(
    candidate: Any,
    motions: tuple[BenchmarkMotionSource, ...],
    *,
    profile: BenchmarkProfile | None = None,
) -> Any:
    """Build the resolved fields consumed by the replay backend."""

    dataset_path = motions[0].path.parent
    return SimpleNamespace(
        robot="demo",
        candidate=candidate,
        dataset_path=dataset_path,
        motions=motions,
        resolved_profile=profile or _profile(),
        fingerprints={
            "asset": {"asset_sha256": SHA_A},
            "dataset": {"path": "inputs", **tree_fingerprint(dataset_path)},
        },
    )


class FakeReplayAdapter:
    """Deterministic host simulator that records reset and control behavior."""

    def __init__(
        self,
        *,
        next_states: tuple[ReplayState, ...] = (),
        bias: tuple[float, ...] = (10.0, 20.0),
    ) -> None:
        """Initialize the fake with optional states returned by successive steps."""

        self.joint_names = ("joint_a", "joint_b")
        self.next_states = list(next_states)
        self.bias = bias
        self.reset_calls: list[dict[str, Any]] = []
        self.step_calls: list[dict[str, Any]] = []
        self.bias_calls = 0
        self.closed = False
        self._state = ReplayState(self.joint_names, [0.0, 0.0], [0.0, 0.0])

    def reset(
        self,
        *,
        initial_position: dict[str, float],
        initial_velocity: dict[str, float],
        initial_command: dict[str, float],
        buffer_time_s: float,
    ) -> ReplayState:
        """Record and return the exact requested initial state."""

        self.reset_calls.append(
            {
                "position": dict(initial_position),
                "velocity": dict(initial_velocity),
                "command": dict(initial_command),
                "buffer_time_s": buffer_time_s,
            }
        )
        self._state = ReplayState(
            self.joint_names,
            [initial_position[name] for name in self.joint_names],
            [initial_velocity[name] for name in self.joint_names],
        )
        return self._state

    def step(
        self,
        *,
        command: dict[str, float],
        effort: dict[str, float],
    ) -> ReplayState:
        """Record one control and return a queued or command-following state."""

        self.step_calls.append({"command": dict(command), "effort": dict(effort)})
        if self.next_states:
            self._state = self.next_states.pop(0)
        else:
            self._state = ReplayState(
                self.joint_names,
                [command[name] for name in self.joint_names],
                [0.0, 0.0],
            )
        return self._state

    def eval_dynamic_bias(self) -> ReplayState:
        """Return deterministic bias values through the position payload."""

        self.bias_calls += 1
        return ReplayState(self.joint_names, self.bias, [0.0, 0.0])

    def close(self) -> None:
        """Record that orchestration closed the simulator."""

        self.closed = True


class FakeNeuralRunner:
    """Recurrent seam fake that records feature rows and emits raw outputs."""

    def __init__(self, outputs: tuple[np.ndarray, ...]) -> None:
        """Store normalized outputs returned by successive calls."""

        self.outputs = list(outputs)
        self.features: list[np.ndarray] = []
        self.reset_sizes: list[int] = []
        self.closed = False

    def reset(self, batch_size: int) -> None:
        """Record one motion-level hidden-state reset."""

        self.reset_sizes.append(batch_size)

    def predict(self, normalized_features: np.ndarray) -> np.ndarray:
        """Capture one feature batch and return the next configured output."""

        self.features.append(np.array(normalized_features, copy=True))
        return self.outputs.pop(0)

    def close(self) -> None:
        """Record recurrent runner cleanup."""

        self.closed = True


def _residual_metadata() -> RuntimeGRUMetadata:
    """Build two shared SISO mappings with every supported runtime feature."""

    mappings = []
    for index, joint in enumerate(("joint_a", "joint_b"), start=1):
        mappings.append(
            {
                "input_joints": [joint],
                "output_joints": [joint],
                "pd_baseline": {
                    "joint_names": [joint],
                    "kp": float(index * 10),
                    "kd": float(index),
                    "velocity_target": "zero",
                },
                "friction_baseline": {
                    "joint_names": [joint],
                    "dry": 0.1 * index,
                    "viscous": 0.2 * index,
                },
                "armature": {"joint_names": [joint], "values": 0.01 * index},
                "effort_limits": 2.0 if index == 1 else 2.5,
                "clamping": "residual_output",
            }
        )
    features = [
        {
            "name": name,
            "domain": "output" if name == "previous_torque" else "input",
            "components": ["value"],
            "width": 1,
        }
        for name in (
            "position",
            "position_error",
            "velocity",
            "solver_pd",
            "dynamic_bias",
            "previous_torque",
        )
    ]
    return RuntimeGRUMetadata.from_dict(
        {
            "schema_version": 3,
            "model_type": "gru",
            "runtime_dtype": "float32",
            "input_size": 6,
            "output_size": 1,
            "recurrent": {"hidden_size": 4, "num_layers": 1, "dropout": 0.0},
            "input_features": features,
            "flattening_order": ["feature", "joint", "component"],
            "mappings": mappings,
            "target_columns": ["torque_residual"],
            "sample_dt_s": 0.01,
            "normalization": {
                "inputs": {
                    "mean": {feature["name"]: 0.0 for feature in features},
                    "std": {feature["name"]: 1.0 for feature in features},
                },
                "targets": {
                    "mean": {"torque_residual": 0.0},
                    "std": {"torque_residual": 1.0},
                },
            },
            "delay": {"handling": "learned", "external_delay_s": 0.0},
            "previous_torque_derivation": {
                "source": "previous_target",
                "runtime_source": "previous_raw_network_output_pre_clamp",
                "initialization": "physical_zero",
                "reset_boundaries": ["recorded_episode", "joint_mapping"],
                "value_space": "residual_torque",
            },
            "dynamic_bias_derivation": {
                "source": "newton.solvers.SolverMuJoCo.eval_dynamic_bias",
                "provenance": {"provider": "test"},
            },
            "hashes": {
                "profile_sha256": SHA_A,
                "resolved_config_sha256": SHA_A,
                "sysid_state_sha256": SHA_A,
                "asset_sha256": SHA_A,
                "source_sha256": SHA_B,
            },
        },
        "test residual metadata",
    )


def _training_candidate(metadata: RuntimeGRUMetadata) -> TrainingBenchmarkCandidate:
    """Wrap runtime metadata in one authenticated training candidate."""

    return TrainingBenchmarkCandidate(
        network_path=Path("network.pt"),
        metadata=metadata,
        metadata_sha256=SHA_A,
        input_joints=("joint_a", "joint_b"),
        output_joints=("joint_a", "joint_b"),
        artifact_hashes={"network": SHA_A, "metadata": SHA_A},
    )


def _total_metadata() -> RuntimeGRUMetadata:
    """Convert the shared mapping fixture to full-torque output semantics."""

    document = _residual_metadata().model_dump(mode="json", exclude_none=True)
    document["target_columns"] = ["torque"]
    for mapping in document["mappings"]:
        mapping["clamping"] = "total_output"
    document["normalization"]["targets"] = {
        "mean": {"torque": 0.0},
        "std": {"torque": 1.0},
    }
    document["previous_torque_derivation"]["value_space"] = "physical_torque"
    return RuntimeGRUMetadata.from_dict(document, "test total metadata")


def _csv_header(payload: str) -> list[str]:
    """Read one trajectory CSV header."""

    return next(csv.reader(StringIO(payload)))


def test_sysid_replay_resamples_linearly_syncs_state_and_limits_metrics(tmp_path: Path) -> None:
    """SysID replay uses interpolated control and reports only selected outputs."""

    motion_path = tmp_path / "reach"
    _write_motion(motion_path)
    adapter = FakeReplayAdapter()
    resolved = _resolved(_sysid_candidate(), (_motion_source(motion_path),))

    result = run_benchmark_replay(resolved, adapter_factory=lambda _resolved: adapter)

    assert adapter.reset_calls[0]["position"] == {"joint_a": 0.0, "joint_b": 10.0}
    assert adapter.reset_calls[0]["velocity"] == {"joint_a": 0.0, "joint_b": 1.0}
    assert adapter.step_calls[0]["command"] == {"joint_a": 0.0, "joint_b": 10.0}
    assert adapter.step_calls[1]["command"] == {"joint_a": 1.0, "joint_b": 12.0}
    assert all(call["effort"] == {} for call in adapter.step_calls)
    assert result.metrics["candidate_outputs"] == ["joint_a"]
    header = _csv_header(result.trajectories["reach"].simulated_csv)
    assert header == ["timestamp_s", "position/joint_a", "velocity/joint_a"]
    assert all("torque" not in column for column in header)
    assert adapter.closed


def test_replay_rejects_trajectory_changes_after_resolution(tmp_path: Path) -> None:
    """Do not install results under an identity for older trajectory bytes."""

    motion_path = tmp_path / "reach"
    _write_motion(motion_path)
    resolved = _resolved(_sysid_candidate(), (_motion_source(motion_path),))
    with (motion_path / "control.csv").open("a", encoding="utf-8") as stream:
        stream.write("CONTROL,0.03,\"[3.0, 15.0]\"\n")

    with pytest.raises(RuntimeError, match="dataset changed after resolution"):
        run_benchmark_replay(
            resolved,
            adapter_factory=lambda _resolved: (_ for _ in ()).throw(
                AssertionError("runtime adapter must not start for stale input")
            ),
        )


def test_candidate_overrides_freeze_sysid_and_residual_values() -> None:
    """Candidate configuration touches only selected or neural-output joints."""

    sysid = candidate_joint_overrides(_sysid_candidate())
    assert set(sysid) == {"joint_a"}
    assert sysid["joint_a"].kp == 20.0
    assert sysid["joint_a"].viscous_friction == 0.4
    assert not sysid["joint_a"].force_only

    neural = candidate_joint_overrides(_training_candidate(_residual_metadata()))
    assert set(neural) == {"joint_a", "joint_b"}
    assert neural["joint_a"].kp == 10.0
    assert neural["joint_b"].kd == 2.0
    assert neural["joint_b"].dry_friction == 0.2
    assert not neural["joint_a"].force_only

    total = candidate_joint_overrides(_training_candidate(_total_metadata()))
    assert total["joint_a"].force_only
    assert total["joint_a"].kp == 0.0
    assert total["joint_a"].kd == 0.0
    assert total["joint_a"].dry_friction is None


def test_residual_replay_builds_features_and_feedback_before_clamp(tmp_path: Path) -> None:
    """Shared mappings have independent rows and feed back raw previous residual."""

    motion_path = tmp_path / "reach"
    _write_motion(
        motion_path,
        command=((2.0, 4.0), (4.0, 6.0)),
        position=((1.0, 2.0), (2.0, 3.0)),
        velocity=((0.1, 0.2), (0.3, 0.4)),
    )
    adapter = FakeReplayAdapter(
        next_states=(
            ReplayState(("joint_a", "joint_b"), [1.5, 2.5], [0.15, 0.25]),
            ReplayState(("joint_a", "joint_b"), [2.0, 3.0], [0.3, 0.4]),
        )
    )
    runner = FakeNeuralRunner(
        (
            np.asarray([[3.0], [4.0]], dtype=np.float32),
            np.asarray([[5.0], [6.0]], dtype=np.float32),
        )
    )
    candidate = _training_candidate(_residual_metadata())
    resolved = _resolved(candidate, (_motion_source(motion_path),))

    result = run_benchmark_replay(
        resolved,
        adapter_factory=lambda _resolved: adapter,
        neural_runner_factory=lambda _candidate: runner,
    )

    np.testing.assert_allclose(
        runner.features[0],
        np.asarray(
            [[1.0, 1.0, 0.1, 9.9, 10.0, 0.0], [2.0, 2.0, 0.2, 39.6, 20.0, 0.0]]
        ),
        rtol=0.0,
        atol=1.0e-5,
    )
    np.testing.assert_allclose(runner.features[1][:, -1], [3.0, 4.0])
    assert adapter.step_calls[0]["effort"] == {"joint_a": 2.0, "joint_b": 2.5}
    assert adapter.step_calls[1]["effort"] == {"joint_a": 2.0, "joint_b": 2.5}
    assert adapter.bias_calls == 2
    assert runner.reset_sizes == [2]
    assert runner.closed and adapter.closed
    assert np.isfinite(result.metrics["aggregate"]["position"]["rmse"])


def test_neural_hidden_and_previous_torque_reset_between_motions(tmp_path: Path) -> None:
    """Every recorded motion starts a fresh recurrent mapping batch."""

    first_path = tmp_path / "first"
    second_path = tmp_path / "second"
    _write_motion(first_path)
    _write_motion(second_path)
    adapter = FakeReplayAdapter()
    runner = FakeNeuralRunner(
        tuple(np.zeros((2, 1), dtype=np.float32) for _ in range(4))
    )
    candidate = _training_candidate(_residual_metadata())
    resolved = _resolved(
        candidate,
        (
            _motion_source(first_path, "first"),
            _motion_source(second_path, "second"),
        ),
    )

    run_benchmark_replay(
        resolved,
        adapter_factory=lambda _resolved: adapter,
        neural_runner_factory=lambda _candidate: runner,
    )

    assert runner.reset_sizes == [2, 2]
    np.testing.assert_array_equal(runner.features[0][:, -1], [0.0, 0.0])
    np.testing.assert_array_equal(runner.features[2][:, -1], [0.0, 0.0])
