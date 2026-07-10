from __future__ import annotations

import csv
import fnmatch
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, cast

import numpy as np

from anchor_bench.core.run_config import resolve_simulation_timing
from anchor_bench.data.motion import MotionData
from anchor_bench.data.sage import load_sage_motion

@dataclass(frozen=True)
class SignalComparison:
    sim_time: np.ndarray
    sim_values: np.ndarray
    real_time: np.ndarray
    real_values: np.ndarray
    common_time: np.ndarray
    sim_common: np.ndarray
    real_common: np.ndarray
    rmse: float
    correlation: float
    cosine_similarity: float


@dataclass(frozen=True)
class SimRun:
    seed: int | None
    dataset: str
    motion: str
    motion_rel: Path
    actuator_model: str
    folder: Path


@dataclass(frozen=True)
class JointOverlay:
    actuator_model: str
    trace: MotionData
    sim_joint_idx: int
    position: SignalComparison
    velocity: SignalComparison
    torque: SignalComparison


@dataclass(frozen=True)
class EndEffectorComparison:
    actuator_model: str
    sim_time: np.ndarray
    sim_xyz: np.ndarray
    real_time: np.ndarray
    real_xyz: np.ndarray
    common_time: np.ndarray
    sim_common: np.ndarray
    real_common: np.ndarray
    rmse_xyz: np.ndarray
    rmse_dist: float
    max_err: float
    duration: float


EndEffectorFk = Callable[[list[str], np.ndarray], np.ndarray]
_DEFAULT_EE_FK = object()


class EndEffectorFkUnavailable(RuntimeError):
    """Raised when forward kinematics solver is unavailable in the current runtime."""


def _read_trace(folder: Path, sample_dt: float | None = None) -> MotionData:
    motion = load_sage_motion(
        folder,
        target_dt=sample_dt,
        require_velocity=True,
        require_effort=True,
    )
    assert motion.velocity is not None
    assert motion.effort is not None
    return motion


def _rmse(sim_values: np.ndarray, real_values: np.ndarray) -> float:
    if sim_values.size == 0 or real_values.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((sim_values - real_values) ** 2)))


def _correlation(sim_values: np.ndarray, real_values: np.ndarray) -> float:
    if sim_values.size < 2 or real_values.size < 2:
        return float("nan")
    if np.allclose(sim_values, sim_values[0]) or np.allclose(real_values, real_values[0]):
        return float("nan")
    return float(np.corrcoef(sim_values, real_values)[0, 1])


def _cosine_similarity(sim_values: np.ndarray, real_values: np.ndarray) -> float:
    if sim_values.size == 0 or real_values.size == 0:
        return float("nan")
    denominator = float(np.linalg.norm(sim_values) * np.linalg.norm(real_values))
    if denominator == 0.0:
        return float("nan")
    return float(np.dot(sim_values, real_values) / denominator)


def _compare_signal(
    sim_time: np.ndarray,
    sim_values: np.ndarray,
    real_time: np.ndarray,
    real_values: np.ndarray,
) -> SignalComparison:
    if sim_time.size == 0 or real_time.size == 0:
        empty = np.asarray([], dtype=float)
        return SignalComparison(
            sim_time,
            sim_values,
            real_time,
            real_values,
            empty,
            empty,
            empty,
            float("nan"),
            float("nan"),
            float("nan"),
        )

    t_start = max(float(sim_time[0]), float(real_time[0]))
    t_end = min(float(sim_time[-1]), float(real_time[-1]))
    if t_end <= t_start:
        empty = np.asarray([], dtype=float)
        return SignalComparison(
            sim_time,
            sim_values,
            real_time,
            real_values,
            empty,
            empty,
            empty,
            float("nan"),
            float("nan"),
            float("nan"),
        )

    count = min(sim_values.size, real_values.size)
    common_time = np.linspace(t_start, t_end, count)
    sim_common = np.interp(common_time, sim_time, sim_values)
    real_common = np.interp(common_time, real_time, real_values)
    return SignalComparison(
        sim_time,
        sim_values,
        real_time,
        real_values,
        common_time,
        sim_common,
        real_common,
        _rmse(sim_common, real_common),
        _correlation(sim_common, real_common),
        _cosine_similarity(sim_common, real_common),
    )


def _format_metric(value: float) -> str:
    if math.isnan(value):
        return "nan"
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return "0.0" if text == "0" else text


def _motion_folders(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("state_motor.csv") if (path.parent / "joint_list.txt").exists())


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unnamed"


def _load_pyplot():
    try:
        cache_dir = Path(os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "anchor-bench-matplotlib")))
        cache_dir.mkdir(parents=True, exist_ok=True)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    return plt


def _seed_roots(result_root: Path, robot_name: str) -> list[tuple[int | None, Path]]:
    seeded = []
    for path in sorted(result_root.glob("seed_*")):
        if not (path / "sim" / robot_name).exists():
            continue
        seed_text = path.name.removeprefix("seed_")
        try:
            seed = int(seed_text)
        except ValueError:
            seed = None
        seeded.append((seed, path))
    if seeded:
        return seeded
    return [(None, result_root)]


def _scan_sim_runs(
    result_root: Path,
    robot_name: str,
    datasets: Iterable[str] | None,
    motions: Iterable[str] | None,
    actuator_models: Iterable[str] | None,
    seeds: Iterable[int] | None,
) -> list[SimRun]:
    dataset_filter = set(_selector_values(datasets))
    motion_filter = _selector_values(motions)
    actuator_filter = set(_selector_values(actuator_models))
    seed_filter = set(seeds or [])
    seed_roots = _seed_roots(result_root, robot_name)
    has_seed_roots = any(seed is not None for seed, _ in seed_roots)
    runs: list[SimRun] = []
    for seed, seed_root in seed_roots:
        if has_seed_roots and seed_filter and seed not in seed_filter:
            continue
        sim_root = seed_root / "sim" / robot_name
        for state_csv in _motion_folders(sim_root):
            folder = state_csv.parent
            relative = folder.relative_to(sim_root)
            if len(relative.parts) < 3:
                continue
            dataset = relative.parts[0]
            actuator_model = relative.parts[1]
            motion_rel = Path(*relative.parts[2:])
            motion = motion_rel.as_posix()
            if dataset_filter and dataset not in dataset_filter:
                continue
            if actuator_filter and actuator_model not in actuator_filter:
                continue
            if motion_filter and not _matches_motion_filter(motion_filter, motion, motion_rel.name):
                continue
            runs.append(
                SimRun(
                    seed=seed,
                    dataset=dataset,
                    motion=motion,
                    motion_rel=motion_rel,
                    actuator_model=actuator_model,
                    folder=folder,
                )
            )
    return runs


def _selector_values(values: Iterable[str] | str | None) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        return [] if values == "*" else [values]
    return [str(item) for item in values]


def _matches_motion_filter(patterns: list[str], motion: str, motion_name: str) -> bool:
    for pattern in patterns:
        if pattern in {motion, motion_name}:
            return True
        if fnmatch.fnmatch(motion, pattern) or fnmatch.fnmatch(motion_name, pattern):
            return True
    return False


def _real_folder(result_root: Path, robot_name: str, seed: int | None, dataset: str, motion_rel: Path) -> Path:
    seed_root = result_root / f"seed_{seed}" if seed is not None and (result_root / f"seed_{seed}").exists() else result_root
    return seed_root / "real" / robot_name / dataset / motion_rel


def _group_plot_dir(output_root: Path, seed: int | None, dataset: str, motion: str) -> Path:
    base = output_root / "plots"
    if seed is not None:
        base = base / f"seed_{seed}"
    base = base / _safe_name(dataset)
    for part in Path(motion).parts:
        if part not in {"", "."}:
            base = base / _safe_name(part)
    return base


def _plot_joint_overlay(
    plt,
    output_path: Path,
    dataset: str,
    motion: str,
    joint: str,
    real_trace: MotionData,
    real_joint_idx: int,
    overlays: list[JointOverlay],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(
        real_trace.command_position.time,
        real_trace.command_position.values[:, real_joint_idx],
        color="#f2a6a6",
        label="Real command",
        linewidth=1.0,
    )
    axes[0].plot(
        real_trace.position.time,
        real_trace.position.values[:, real_joint_idx],
        color="#b00020",
        linestyle="--",
        label="Real position",
        linewidth=1.2,
    )
    for overlay in overlays:
        axes[0].plot(
            overlay.position.sim_time,
            overlay.position.sim_values,
            linewidth=1.2,
            label=f"{overlay.actuator_model} position ({overlay.position.rmse:.4f} rad)",
        )
    axes[0].set_ylabel("position [rad]")
    axes[0].set_title(f"{dataset} / {motion} / {joint} position")
    axes[0].legend(loc="best")
    axes[0].grid(True)

    axes[1].plot(real_trace.velocity.time, real_trace.velocity.values[:, real_joint_idx], color="#b00020", linestyle="--", label="Real velocity", linewidth=1.2)
    for overlay in overlays:
        axes[1].plot(
            overlay.velocity.sim_time,
            overlay.velocity.sim_values,
            linewidth=1.2,
            label=f"{overlay.actuator_model} velocity ({overlay.velocity.rmse:.4f} rad/s)",
        )
    axes[1].set_ylabel("velocity [rad/s]")
    axes[1].set_title("Velocity")
    axes[1].legend(loc="best")
    axes[1].grid(True)

    axes[2].plot(real_trace.effort.time, real_trace.effort.values[:, real_joint_idx], color="#b00020", linestyle="--", label="Real torque", linewidth=1.2)
    for overlay in overlays:
        axes[2].plot(
            overlay.torque.sim_time,
            overlay.torque.sim_values,
            linewidth=1.2,
            label=f"{overlay.actuator_model} torque ({overlay.torque.rmse:.4f} Nm)",
        )
    axes[2].set_xlabel("time [s]")
    axes[2].set_ylabel("torque [Nm]")
    axes[2].set_title("Torque")
    axes[2].legend(loc="best")
    axes[2].grid(True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _compare_ee(
    actuator_model: str,
    sim_time: np.ndarray,
    sim_xyz: np.ndarray,
    real_time: np.ndarray,
    real_xyz: np.ndarray,
) -> EndEffectorComparison:
    if sim_time.size == 0 or real_time.size == 0:
        empty_xyz = np.empty((0, 3), dtype=float)
        nan_xyz = np.asarray([float("nan"), float("nan"), float("nan")], dtype=float)
        return EndEffectorComparison(actuator_model, sim_time, sim_xyz, real_time, real_xyz, np.asarray([]), empty_xyz, empty_xyz, nan_xyz, float("nan"), float("nan"), float("nan"))
    t_start = max(float(sim_time[0]), float(real_time[0]))
    t_end = min(float(sim_time[-1]), float(real_time[-1]))
    if t_end <= t_start:
        empty_xyz = np.empty((0, 3), dtype=float)
        nan_xyz = np.asarray([float("nan"), float("nan"), float("nan")], dtype=float)
        return EndEffectorComparison(actuator_model, sim_time, sim_xyz, real_time, real_xyz, np.asarray([]), empty_xyz, empty_xyz, nan_xyz, float("nan"), float("nan"), float("nan"))
    count = min(sim_xyz.shape[0], real_xyz.shape[0])
    common_time = np.linspace(t_start, t_end, count)
    sim_common = np.column_stack([np.interp(common_time, sim_time, sim_xyz[:, axis]) for axis in range(3)])
    real_common = np.column_stack([np.interp(common_time, real_time, real_xyz[:, axis]) for axis in range(3)])
    errors = sim_common - real_common
    dists = np.linalg.norm(errors, axis=1)
    return EndEffectorComparison(
        actuator_model=actuator_model,
        sim_time=sim_time,
        sim_xyz=sim_xyz,
        real_time=real_time,
        real_xyz=real_xyz,
        common_time=common_time,
        sim_common=sim_common,
        real_common=real_common,
        rmse_xyz=np.sqrt(np.mean(errors**2, axis=0)),
        rmse_dist=float(np.sqrt(np.mean(dists**2))),
        max_err=float(np.max(dists)),
        duration=float(t_end - t_start),
    )


def _plot_ee_overlay(
    plt,
    output_path: Path,
    dataset: str,
    motion: str,
    real_xyz: np.ndarray,
    comparisons: list[EndEffectorComparison],
) -> None:
    if not comparisons:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(real_xyz[:, 0], real_xyz[:, 1], real_xyz[:, 2], color="#1f77b4", linewidth=2.0, label="Real EE")
    for comparison in comparisons:
        ax.plot(
            comparison.sim_xyz[:, 0],
            comparison.sim_xyz[:, 1],
            comparison.sim_xyz[:, 2],
            linewidth=1.8,
            label=f"{comparison.actuator_model} ({comparison.rmse_dist * 1000.0:.2f} mm)",
        )
    ax.scatter([real_xyz[0, 0]], [real_xyz[0, 1]], [real_xyz[0, 2]], color="black", marker="o", label="start")
    ax.scatter([real_xyz[-1, 0]], [real_xyz[-1, 1]], [real_xyz[-1, 2]], color="black", marker="x", label="end")
    ax.set_title(f"SO-101 end-effector trajectory - {dataset}/{motion}")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _batched_fk_positions(
    positions: np.ndarray,
    batch_size: int,
    evaluate_chunk: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    if batch_size < 1:
        raise ValueError("FK batch_size must be >= 1")
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2:
        raise ValueError(f"FK positions must be a 2D array, got shape {positions.shape}")
    if positions.shape[0] == 0:
        return np.empty((0, 3), dtype=float)

    chunks: list[np.ndarray] = []
    for start in range(0, positions.shape[0], batch_size):
        chunk = positions[start : start + batch_size]
        xyz = np.asarray(evaluate_chunk(chunk), dtype=float)
        expected_shape = (chunk.shape[0], 3)
        if xyz.shape != expected_shape:
            raise ValueError(f"FK chunk returned shape {xyz.shape}, expected {expected_shape}")
        chunks.append(xyz)
    return np.concatenate(chunks, axis=0)


class IsaacLabNewtonRobotFk:
    def __init__(self, robot_name: str, ee_frame: str | None = None, batch_size: int = 512) -> None:
        self._simulation_app = None
        self._batch_size = int(batch_size)
        if self._batch_size < 1:
            raise ValueError("FK batch_size must be >= 1")
        try:
            from anchor_bench.runtime.app_launcher import launch_isaac_sim
        except Exception as exc:  # pragma: no cover - host-safe fallback
            raise EndEffectorFkUnavailable(f"FK runtime is unavailable: {exc}") from exc

        try:  # pragma: no cover - exercised only in full runtime
            self._app_launcher, self._simulation_app = launch_isaac_sim(headless=True)
        except Exception as exc:  # pragma: no cover - full runtime startup only
            raise EndEffectorFkUnavailable(f"FK app startup failed: {exc}") from exc

        try:  # pragma: no cover - exercised only in full runtime
            import torch
            from isaaclab.scene import InteractiveScene
            from isaaclab.sim import SimulationCfg, SimulationContext
            from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

            from anchor_bench.runtime.robot_scene import build_scene_class, resolve_fix_root_override
            from anchor_bench.robots.registry import get_robot
            from anchor_bench.runtime.assets import resolve_runtime_asset
            from anchor_bench.runtime.robot_config import load_implicit_actuator_params
        except Exception as exc:  # pragma: no cover - host-safe fallback
            self.close()
            raise EndEffectorFkUnavailable(f"FK dependencies are unavailable: {exc}") from exc

        self._torch = torch

        try:
            bundle = get_robot(robot_name)
            ee_frame = ee_frame or bundle.spec.ee_frame
            run_cfg = bundle.run_config
            sim_cfg_data = run_cfg.simulation
            timing = resolve_simulation_timing(sim_cfg_data, f"{bundle.spec.display_name} analysis FK simulation")
            fix_root = (run_cfg.benchmark.fix_root if run_cfg.benchmark.fix_root is not None else bundle.spec.fixed_base)
            actuator_values = load_implicit_actuator_params(bundle, None)
            usd_path = resolve_runtime_asset(bundle)
            fix_root_override = resolve_fix_root_override(fix_root, bundle.spec.fixed_base)
            SceneCfg = build_scene_class(
                usd_path, actuator_values, list(bundle.spec.joint_names), fix_root=fix_root_override
            )
            sim_cfg = SimulationCfg(dt=timing.physics_dt, render_interval=timing.render_interval, gravity=(0.0, 0.0, -9.81))
            sim_cfg.physics = NewtonCfg(
                solver_cfg=MJWarpSolverCfg(
                    integrator=sim_cfg_data.integrator or "implicitfast",
                    ls_iterations=sim_cfg_data.line_search_iterations or 10,
                    iterations=sim_cfg_data.solver_iterations or 10,
                ),
            )
            self._sim = SimulationContext(sim_cfg)
            self._scene = InteractiveScene(SceneCfg(num_envs=self._batch_size, env_spacing=0.0))
            self._sim.reset()
            self._scene.reset()
            self._robot = self._scene["robot"]
            body_names = list(getattr(self._robot, "body_names", []))
            if ee_frame not in body_names:
                raise EndEffectorFkUnavailable(f"{bundle.spec.display_name} ee_frame {ee_frame!r} not found in body names: {body_names}")
            self._ee_body_idx = body_names.index(ee_frame)
            self._ee_frame = ee_frame
            self._joint_name_to_idx = {name: i for i, name in enumerate(self._robot.joint_names)}
        except Exception:
            self.close()
            raise

    def __call__(self, joint_names: list[str], positions: np.ndarray) -> np.ndarray:  # pragma: no cover - full runtime only
        torch = self._torch
        joint_indices = [self._joint_name_to_idx[name] for name in joint_names]
        joint_ids = torch.tensor(joint_indices, dtype=torch.int64, device=self._robot.device)
        joint_pos = torch.zeros((self._batch_size, len(self._robot.joint_names)), dtype=torch.float32, device=self._robot.device)
        joint_vel = torch.zeros_like(joint_pos)

        def evaluate_chunk(chunk: np.ndarray) -> np.ndarray:
            active_count = chunk.shape[0]
            joint_pos.zero_()
            joint_vel.zero_()
            joint_pos[:active_count].index_copy_(
                1,
                joint_ids,
                torch.as_tensor(chunk, dtype=torch.float32, device=self._robot.device),
            )
            self._robot.write_joint_state_to_sim(joint_pos, joint_vel)
            self._scene.write_data_to_sim()
            if hasattr(self._sim, "forward"):
                self._sim.forward()
            else:
                self._sim.step(render=False)
            self._scene.update(0.0)
            pose = self._robot.data.body_link_pose_w[:active_count, self._ee_body_idx, :3]
            return pose.detach().cpu().numpy().astype(float)

        return _batched_fk_positions(positions, self._batch_size, evaluate_chunk)

    def close(self) -> None:  # pragma: no cover - full runtime only
        if getattr(self, "_simulation_app", None) is not None:
            self._simulation_app.close()
            self._simulation_app = None


def _default_ee_fk(robot_name: str) -> EndEffectorFk:
    try:
        return IsaacLabNewtonRobotFk(robot_name)
    except EndEffectorFkUnavailable:
        raise
    except Exception as exc:  # pragma: no cover - depends on full runtime startup
        raise EndEffectorFkUnavailable(f"FK startup failed: {exc}") from exc


def _warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr)


def _metric_rows() -> tuple[list[list[str]], list[list[str]]]:
    joint_rows = [[
        "dataset", "motion", "actuator_model", "seed", "joint",
        "position_rmse", "velocity_rmse", "torque_rmse",
        "position_correlation", "velocity_correlation", "torque_correlation",
        "position_cosine_similarity", "velocity_cosine_similarity", "torque_cosine_similarity",
    ]]
    end_effector_rows = [[
        "dataset", "motion", "actuator_model", "seed",
        "rmse_x_m", "rmse_y_m", "rmse_z_m", "rmse_dist_m", "max_err_m", "duration_s",
    ]]
    return joint_rows, end_effector_rows


def _group_sim_runs(sim_runs: list[SimRun]) -> dict[tuple[int | None, str, str], list[SimRun]]:
    groups: dict[tuple[int | None, str, str], list[SimRun]] = {}
    for run in sim_runs:
        groups.setdefault((run.seed, run.dataset, run.motion), []).append(run)
    return groups


def _group_sort_key(item: tuple[tuple[int | None, str, str], list[SimRun]]) -> tuple[bool, int, str, str]:
    seed, dataset, motion = item[0]
    return seed is not None, seed or -1, dataset, motion


def _load_analysis_group(
    result_root: Path,
    robot_name: str,
    key: tuple[int | None, str, str],
    runs: list[SimRun],
    sample_dt: float | None,
) -> tuple[MotionData, list[tuple[SimRun, MotionData]], list[str]] | None:
    seed, dataset, _motion = key
    real_folder = _real_folder(result_root, robot_name, seed, dataset, runs[0].motion_rel)
    if not (real_folder / "state_motor.csv").exists():
        return None
    real_trace = _read_trace(real_folder, sample_dt=sample_dt)
    sim_traces = [
        (run, _read_trace(run.folder, sample_dt=sample_dt))
        for run in sorted(runs, key=lambda item: item.actuator_model)
    ]
    joints = [
        joint
        for joint in real_trace.joint_names
        if any(joint in trace.joint_names for _, trace in sim_traces)
    ]
    return real_trace, sim_traces, joints


def _compare_joint_group(
    rows: list[list[str]],
    key: tuple[int | None, str, str],
    real_trace: MotionData,
    sim_traces: list[tuple[SimRun, MotionData]],
    joints: list[str],
) -> dict[str, list[JointOverlay]]:
    seed, dataset, motion = key
    seed_text = "" if seed is None else str(seed)
    overlays: dict[str, list[JointOverlay]] = {joint: [] for joint in joints}
    for run, sim_trace in sim_traces:
        for joint in joints:
            if joint not in sim_trace.joint_names:
                continue
            sim_idx = sim_trace.joint_names.index(joint)
            real_idx = real_trace.joint_names.index(joint)
            assert sim_trace.velocity is not None
            assert sim_trace.effort is not None
            assert real_trace.velocity is not None
            assert real_trace.effort is not None
            comparisons = [
                _compare_signal(
                    sim_signal.time,
                    sim_signal.values[:, sim_idx],
                    real_signal.time,
                    real_signal.values[:, real_idx],
                )
                for sim_signal, real_signal in (
                    (sim_trace.position, real_trace.position),
                    (sim_trace.velocity, real_trace.velocity),
                    (sim_trace.effort, real_trace.effort),
                )
            ]
            position, velocity, torque = comparisons
            rows.append([
                dataset, motion, run.actuator_model, seed_text, joint,
                _format_metric(position.rmse), _format_metric(velocity.rmse), _format_metric(torque.rmse),
                _format_metric(position.correlation), _format_metric(velocity.correlation), _format_metric(torque.correlation),
                _format_metric(position.cosine_similarity),
                _format_metric(velocity.cosine_similarity),
                _format_metric(torque.cosine_similarity),
            ])
            overlays[joint].append(
                JointOverlay(
                    actuator_model=run.actuator_model,
                    trace=sim_trace,
                    sim_joint_idx=sim_idx,
                    position=position,
                    velocity=velocity,
                    torque=torque,
                )
            )
    return overlays


def _plot_joint_group(
    plt,
    output_root: Path,
    key: tuple[int | None, str, str],
    real_trace: MotionData,
    overlays: dict[str, list[JointOverlay]],
) -> None:
    seed, dataset, motion = key
    group_dir = _group_plot_dir(output_root, seed, dataset, motion)
    for joint, joint_overlays in overlays.items():
        if joint_overlays:
            _plot_joint_overlay(
                plt,
                group_dir / "joints" / f"{_safe_name(joint)}.png",
                dataset,
                motion,
                joint,
                real_trace,
                real_trace.joint_names.index(joint),
                joint_overlays,
            )


def _compare_end_effector_group(
    rows: list[list[str]],
    plt,
    output_root: Path,
    key: tuple[int | None, str, str],
    real_trace: MotionData,
    sim_traces: list[tuple[SimRun, MotionData]],
    fk: EndEffectorFk,
) -> None:
    seed, dataset, motion = key
    real_xyz = np.asarray(fk(real_trace.joint_names, real_trace.position.values), dtype=float)
    comparisons: list[EndEffectorComparison] = []
    for run, sim_trace in sim_traces:
        sim_xyz = np.asarray(fk(sim_trace.joint_names, sim_trace.position.values), dtype=float)
        comparison = _compare_ee(
            run.actuator_model,
            sim_trace.position.time,
            sim_xyz,
            real_trace.position.time,
            real_xyz,
        )
        comparisons.append(comparison)
        rows.append([
            dataset, motion, run.actuator_model, "" if seed is None else str(seed),
            _format_metric(float(comparison.rmse_xyz[0])),
            _format_metric(float(comparison.rmse_xyz[1])),
            _format_metric(float(comparison.rmse_xyz[2])),
            _format_metric(comparison.rmse_dist),
            _format_metric(comparison.max_err),
            _format_metric(comparison.duration),
        ])
    if plt is not None:
        _plot_ee_overlay(
            plt,
            _group_plot_dir(output_root, seed, dataset, motion) / "end_effector.png",
            dataset,
            motion,
            real_xyz,
            comparisons,
        )


def _resolve_end_effector_fk(
    robot_name: str,
    enabled: bool,
    required: bool,
    ee_fk: EndEffectorFk | object,
) -> tuple[EndEffectorFk | None, Callable[[], None] | None]:
    if not enabled:
        return None, None
    try:
        fk = cast(EndEffectorFk, _default_ee_fk(robot_name) if ee_fk is _DEFAULT_EE_FK else ee_fk)
        return fk, getattr(fk, "close", None)
    except EndEffectorFkUnavailable as exc:
        if required:
            raise
        _warn(f"skipping {robot_name} end-effector analysis: {exc}")
        return None, None


def _write_analysis_metrics(
    robot_name: str,
    metrics_dir: Path,
    joint_rows: list[list[str]],
    end_effector_rows: list[list[str]],
) -> Path:
    metrics_path = metrics_dir / f"{robot_name}_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerows(joint_rows)
    if len(end_effector_rows) > 1:
        with (metrics_dir / f"{robot_name}_end_effector_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerows(end_effector_rows)
    return metrics_path


def run_robot_analysis(
    robot_name: str,
    result_folder: str | Path,
    output: str | Path,
    datasets: Iterable[str] | None = None,
    motions: Iterable[str] | None = None,
    actuator_models: Iterable[str] | None = None,
    make_joint_plots: bool = True,
    make_end_effector_plots: bool = True,
    require_end_effector: bool = False,
    ee_fk: EndEffectorFk | object = _DEFAULT_EE_FK,
    seeds: Iterable[int] | None = None,
    sample_dt: float | None = None,
) -> Path:
    result_root = Path(result_folder)
    output_root = Path(output)
    metrics_dir = output_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plt = _load_pyplot() if make_joint_plots or make_end_effector_plots else None
    joint_rows, end_effector_rows = _metric_rows()
    groups = _group_sim_runs(
        _scan_sim_runs(result_root, robot_name, datasets, motions, actuator_models, seeds)
    )
    fk, close_fk = _resolve_end_effector_fk(
        robot_name,
        make_end_effector_plots,
        require_end_effector,
        ee_fk,
    )

    try:
        for key, runs in sorted(groups.items(), key=_group_sort_key):
            loaded = _load_analysis_group(result_root, robot_name, key, runs, sample_dt)
            if loaded is None:
                continue
            real_trace, sim_traces, joints = loaded
            overlays = _compare_joint_group(joint_rows, key, real_trace, sim_traces, joints)
            if plt is not None and make_joint_plots:
                _plot_joint_group(plt, output_root, key, real_trace, overlays)
            if fk is not None:
                try:
                    _compare_end_effector_group(
                        end_effector_rows,
                        plt,
                        output_root,
                        key,
                        real_trace,
                        sim_traces,
                        fk,
                    )
                except EndEffectorFkUnavailable as exc:
                    if require_end_effector:
                        raise
                    _, dataset, motion = key
                    _warn(f"skipping {robot_name} end-effector analysis for {dataset}/{motion}: {exc}")
                    fk = None
        return _write_analysis_metrics(robot_name, metrics_dir, joint_rows, end_effector_rows)
    finally:
        if close_fk is not None:
            close_fk()
