from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from anchor_bench.core.run_config import resolve_simulation_timing
from anchor_bench.benchmark.init_sync import (
    BenchmarkInitialState,
    apply_benchmark_initial_states,
    load_benchmark_initial_state,
)
from anchor_bench.data.sage import load_real_data
from anchor_bench.robots.registry import get_robot
from anchor_bench.runtime.assets import resolve_runtime_asset
from anchor_bench.runtime.so101_config import load_implicit_actuator_params
from anchor_bench.runtime.so101_scene import (
    apply_static_actuator_params,
    build_scene_class,
    so101_joint_names,
    to_torch,
)


def _log(message: str) -> None:
    print(f"[Benchmark] {message}", flush=True)


def _write_sage(folder: Path, joint_names: list[str], control_rows, state_rows) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "joint_list.txt").write_text("\n".join(joint_names) + "\n", encoding="utf-8")
    with (folder / "control.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["type", "timestamp", "positions"])
        writer.writerows(control_rows)
    with (folder / "state_motor.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["type", "timestamp", "positions", "velocities", "torques"])
        writer.writerows(state_rows)


def _copy_real_sage(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("control.csv", "state_motor.csv", "joint_list.txt", "metadata.yaml"):
        src = source / name
        if src.exists():
            shutil.copy2(src, destination / name)


def _timestamp_us_text(timestamp_s: float) -> str:
    return f"{timestamp_s * 1e6:.1f}"


def _append_benchmark_sample(
    control_rows: list[list[Any]],
    state_rows: list[list[Any]],
    *,
    timestamp_s: float,
    command: list[float],
    position: list[float],
    velocity: list[float],
) -> None:
    position_values = list(position)
    velocity_values = list(velocity)
    timestamp_text = _timestamp_us_text(timestamp_s)
    effort = [0.0] * len(position_values)
    control_rows.append(["CONTROL", timestamp_text, list(command)])
    state_rows.append(["STATE_MOTOR", timestamp_text, position_values, velocity_values, effort])


def _buffer_substeps(buffer_time: float, control_dt: float, control_divisor: int) -> int:
    if buffer_time < 0.0:
        raise ValueError("benchmark.buffer_time must be >= 0")
    return int(round(buffer_time / control_dt)) * int(control_divisor)


@dataclass(frozen=True)
class So101BenchmarkMotion:
    motion: str
    motion_name: str | None = None


@dataclass(frozen=True)
class _LoadedBenchmarkMotion:
    spec: So101BenchmarkMotion
    path: Path
    motion_name: str
    commands: list[list[float]]
    timestamps: list[float]
    initial_state: BenchmarkInitialState | None


def _build_run_summary(
    motion_path: Path,
    motion_name: str,
    dataset_name: str,
    actuator_model_name: str,
    seed: int | None,
    actual_num_envs: int,
    configured_batch_size: int,
    batch_size: int,
    batch_index: int,
    env_index: int,
    actuator_values: dict[str, Any],
    actuator_params: str | None,
    real_init_pose_sync: bool = False,
) -> dict[str, Any]:
    return {
        "robot_name": "so101",
        "dataset": dataset_name,
        "motion": str(motion_path),
        "motion_name": motion_name,
        "actuator_model": actuator_model_name,
        "actuator_type": "implicit",
        "seed": seed,
        "num_envs": actual_num_envs,
        "configured_batch_size": configured_batch_size,
        "batch_size": batch_size,
        "batch_index": batch_index,
        "env_index": env_index,
        "actuator_parameters": {
            "parameters": actuator_values,
            "source": actuator_params or "robot_config",
        },
        "runtime_options": {
            "real_init_pose_sync": real_init_pose_sync,
        },
    }


def run_so101_benchmark(
    motion: str,
    output: str = "outputs/sim2real_benchmark",
    headless: bool = False,
    motion_name: str | None = None,
    actuator_params: str | None = None,
    dataset: str | None = None,
    actuator_model: str | None = None,
    seed: int | None = None,
) -> Path:
    return run_so101_benchmark_batch(
        motions=[So101BenchmarkMotion(motion=motion, motion_name=motion_name)],
        output=output,
        headless=headless,
        actuator_params=actuator_params,
        dataset=dataset,
        actuator_model=actuator_model,
        seed=seed,
        num_envs=1,
        batch_index=0,
    )


def run_so101_benchmark_batch(
    motions: list[So101BenchmarkMotion],
    output: str = "outputs/sim2real_benchmark",
    headless: bool = False,
    actuator_params: str | None = None,
    dataset: str | None = None,
    actuator_model: str | None = None,
    seed: int | None = None,
    num_envs: int | None = None,
    batch_index: int = 0,
) -> Path:
    if not motions:
        raise ValueError("SO-101 benchmark batch requires at least one motion")
    from anchor_bench.runtime.app_launcher import launch_isaac_sim

    simulation_app = None
    try:
        _app_launcher, simulation_app = launch_isaac_sim(headless=headless)

        import torch
        import warp as wp
        from isaaclab.scene import InteractiveScene
        from isaaclab.sim import SimulationCfg, SimulationContext
        from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

        bundle = get_robot("so101")
        joint_names = so101_joint_names(bundle)
        run_cfg = bundle.run_config
        sim_cfg_data = run_cfg.get("simulation", {})
        benchmark_cfg = run_cfg.get("benchmark", {})
        real_init_pose_sync = bool(benchmark_cfg.get("real_init_pose_sync", False))
        timing = resolve_simulation_timing(sim_cfg_data, "SO-101 benchmark simulation")
        physics_dt = timing.physics_dt
        control_freq = timing.control_freq
        divisor = timing.control_divisor
        buffer_time = float(benchmark_cfg.get("buffer_time", 0.0))
        buffer_substeps = _buffer_substeps(buffer_time, timing.control_dt, timing.control_divisor)
        fix_root = bool(benchmark_cfg.get("fix_root", bundle.spec.fixed_base))

        loaded_motions: list[_LoadedBenchmarkMotion] = []
        for spec in motions:
            motion_path = Path(spec.motion)
            real_data = load_real_data(motion_path, target_dt=1.0 / control_freq, joint_names=joint_names)
            initial_state = load_benchmark_initial_state(motion_path, joint_names) if real_init_pose_sync else None
            commands = real_data.control_positions
            timestamps = real_data.control_timestamps
            if not commands:
                raise ValueError(f"No SAGE rows found in {spec.motion}")
            if len(timestamps) != len(commands):
                raise ValueError(f"SAGE timestamps and command rows differ for {spec.motion}")
            loaded_motions.append(
                _LoadedBenchmarkMotion(
                    spec=spec,
                    path=motion_path,
                    motion_name=spec.motion_name or motion_path.name,
                    commands=commands,
                    timestamps=timestamps,
                    initial_state=initial_state,
                )
            )
        batch_size = len(loaded_motions)
        actual_num_envs = batch_size
        configured_batch_size = int(num_envs or batch_size)

        dataset_name = dataset or benchmark_cfg.get("dataset") or benchmark_cfg.get("motion_name", "example")
        actuator_model_name = actuator_model or "so101_implicit"
        output_root = Path(output)

        actuator_values = load_implicit_actuator_params(bundle, actuator_params)
        usd_path = resolve_runtime_asset(bundle)
        SceneCfg = build_scene_class(usd_path, actuator_values, joint_names, fix_root=fix_root)
        sim_cfg = SimulationCfg(dt=physics_dt, render_interval=timing.render_interval, gravity=(0.0, 0.0, -9.81))
        sim_cfg.physics = NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                integrator=sim_cfg_data.get("integrator", "implicitfast"),
                ls_iterations=sim_cfg_data.get("ls_iterations", sim_cfg_data.get("line_search_iterations", 10)),
                iterations=sim_cfg_data.get("solver_iterations", 10),
            ),
        )
        sim = SimulationContext(sim_cfg)
        scene = InteractiveScene(SceneCfg(num_envs=batch_size, env_spacing=4.0))
        sim.reset()
        scene.reset()
        robot = scene["robot"]
        joint_name_to_idx = {name: i for i, name in enumerate(robot.joint_names)}
        joint_indices = [joint_name_to_idx[name] for name in joint_names]
        joint_ids = torch.tensor(joint_indices, dtype=torch.int32, device=robot.device)
        env_ids = torch.arange(batch_size, dtype=torch.int32, device=robot.device)
        apply_static_actuator_params(robot, actuator_values, joint_ids, env_ids, torch, joint_names)

        control_rows = [[] for _ in loaded_motions]
        state_rows = [[] for _ in loaded_motions]
        starts = torch.tensor([motion.commands[0] for motion in loaded_motions], dtype=torch.float32, device=robot.device)
        joint_template = torch.zeros((batch_size, len(robot.joint_names)), dtype=torch.float32, device=robot.device)
        joint_pos = joint_template.clone()
        joint_vel = torch.zeros_like(joint_pos)
        joint_pos[:, joint_ids.long()] = starts
        joint_ids_long = joint_ids.long()
        target_buffer = torch.zeros_like(joint_template)
        target_buffer[:, joint_ids_long] = starts
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        robot.set_joint_position_target(target_buffer)
        for _ in range(max(1, buffer_substeps)):
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(physics_dt)

        if real_init_pose_sync:
            sync_count = apply_benchmark_initial_states(
                robot,
                to_torch_fn=to_torch,
                torch_module=torch,
                warp_module=wp,
                joint_ids_long=joint_ids_long,
                joint_names=joint_names,
                initial_states=[motion.initial_state for motion in loaded_motions],
            )
            _log(f"Init-pose sync: teleported {sync_count} envs to real row-0 state")

        sim_pos = to_torch(robot.data.joint_pos, wp)[:, joint_ids_long].detach().cpu().numpy()
        sim_vel = to_torch(robot.data.joint_vel, wp)[:, joint_ids_long].detach().cpu().numpy()
        for env_index, motion in enumerate(loaded_motions):
            _append_benchmark_sample(
                control_rows[env_index],
                state_rows[env_index],
                timestamp_s=motion.timestamps[0],
                command=motion.commands[0],
                position=sim_pos[env_index].tolist(),
                velocity=sim_vel[env_index].tolist(),
            )

        max_steps = max(len(motion.commands) for motion in loaded_motions)
        for index in range(1, max_steps):
            target_buffer.zero_()
            command_batch = []
            active = []
            for motion in loaded_motions:
                is_active = index < len(motion.commands)
                active.append(is_active)
                command_batch.append(motion.commands[index] if is_active else motion.commands[-1])
            command_t = torch.tensor(command_batch, dtype=torch.float32, device=robot.device)
            target_buffer[:, joint_ids_long] = command_t
            robot.set_joint_position_target(target_buffer)
            for _ in range(divisor):
                scene.write_data_to_sim()
                sim.step(render=False)
                scene.update(physics_dt)
            sim_pos = to_torch(robot.data.joint_pos, wp)[:, joint_ids_long].detach().cpu().numpy()
            sim_vel = to_torch(robot.data.joint_vel, wp)[:, joint_ids_long].detach().cpu().numpy()
            for env_index, motion in enumerate(loaded_motions):
                if not active[env_index]:
                    continue
                _append_benchmark_sample(
                    control_rows[env_index],
                    state_rows[env_index],
                    timestamp_s=motion.timestamps[index],
                    command=command_batch[env_index],
                    position=sim_pos[env_index].tolist(),
                    velocity=sim_vel[env_index].tolist(),
                )

        first_sim_output: Path | None = None
        for env_index, motion in enumerate(loaded_motions):
            sim_output = output_root / "sim" / "so101" / dataset_name / actuator_model_name / motion.motion_name
            real_output = output_root / "real" / "so101" / dataset_name / motion.motion_name
            if first_sim_output is None:
                first_sim_output = sim_output
            _write_sage(sim_output, joint_names, control_rows[env_index], state_rows[env_index])
            _copy_real_sage(motion.path, real_output)
            summary = _build_run_summary(
                motion_path=motion.path,
                motion_name=motion.motion_name,
                dataset_name=dataset_name,
                actuator_model_name=actuator_model_name,
                seed=seed,
                actual_num_envs=actual_num_envs,
                configured_batch_size=configured_batch_size,
                batch_size=batch_size,
                batch_index=batch_index,
                env_index=env_index,
                actuator_values=actuator_values,
                actuator_params=actuator_params,
                real_init_pose_sync=real_init_pose_sync,
            )
            (sim_output / "run_summary.yaml").write_text(
                yaml.safe_dump(summary, sort_keys=False),
                encoding="utf-8",
            )
            _log(f"{motion.motion_name} saved to {sim_output}")
        return first_sim_output or output_root
    finally:
        if simulation_app is not None:
            simulation_app.close()
