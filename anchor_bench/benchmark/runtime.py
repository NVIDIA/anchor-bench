from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from anchor_bench.core.run_config import resolve_simulation_timing
from anchor_bench.benchmark.init_sync import (
    BenchmarkInitialState,
    apply_benchmark_initial_states,
)
from anchor_bench.data.sage import load_sage_motion
from anchor_bench.data.sage import copy_sage_motion, write_sage_motion
from anchor_bench.robots.registry import get_robot
from anchor_bench.runtime.assets import resolve_runtime_asset
from anchor_bench.runtime.simulation import create_newton_scene, joint_ids_for
from anchor_bench.runtime.robot_config import load_implicit_actuator_params
from anchor_bench.runtime.robot_scene import (
    apply_static_actuator_params,
    build_scene_class,
    resolve_fix_root_override,
    robot_joint_names,
    to_torch,
)


def _log(message: str) -> None:
    print(f"[Benchmark] {message}", flush=True)


def _write_sage(folder: Path, joint_names: list[str], control_rows, state_rows) -> None:
    write_sage_motion(
        folder,
        joint_names,
        control_rows,
        state_rows,
        {
            "schema": "sage",
            "time_unit": "microseconds",
            "position_unit": "rad",
            "velocity_unit": "rad/s",
            "torque_unit": "N*m",
            "source": "anchor-bench simulation",
        },
    )


def _copy_real_sage(source: Path, destination: Path) -> None:
    copy_sage_motion(source, destination)


def _timestamp_us_text(timestamp_s: float) -> str:
    return f"{timestamp_s * 1e6:.1f}"


def _read_benchmark_state(robot: Any, wp: Any, joint_ids_long: Any) -> tuple[Any, Any, Any]:
    position = to_torch(robot.data.joint_pos, wp)[:, joint_ids_long].detach().cpu().numpy()
    velocity = to_torch(robot.data.joint_vel, wp)[:, joint_ids_long].detach().cpu().numpy()
    # Note: Implicit model's applied_torque is a clipped actuator torque est and not Newton's internal torque 
    effort = to_torch(robot.data.applied_torque, wp)[:, joint_ids_long].detach().cpu().numpy()
    return position, velocity, effort


def _append_benchmark_sample(
    control_rows: list[list[Any]],
    state_rows: list[list[Any]],
    *,
    timestamp_s: float,
    command: list[float],
    position: list[float],
    velocity: list[float],
    effort: list[float],
) -> None:
    position_values = list(position)
    velocity_values = list(velocity)
    effort_values = list(effort)
    timestamp_text = _timestamp_us_text(timestamp_s)
    control_rows.append(["CONTROL", timestamp_text, list(command)])
    state_rows.append(["STATE_MOTOR", timestamp_text, position_values, velocity_values, effort_values])


def _buffer_substeps(buffer_time: float, control_dt: float, control_divisor: int) -> int:
    if buffer_time < 0.0:
        raise ValueError("benchmark.buffer_time must be >= 0")
    return int(round(buffer_time / control_dt)) * int(control_divisor)


@dataclass(frozen=True)
class RobotBenchmarkMotion:
    motion: str
    motion_name: str | None = None


@dataclass(frozen=True)
class _LoadedBenchmarkMotion:
    spec: RobotBenchmarkMotion
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
    robot_name: str,
    real_init_pose_sync: bool = False,
) -> dict[str, Any]:
    return {
        "robot_name": robot_name,
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


def run_robot_benchmark_batch(  # noqa: C901
    robot_name: str,
    motions: list[RobotBenchmarkMotion],
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
        raise ValueError(f"{robot_name} benchmark batch requires at least one motion")
    from anchor_bench.runtime.app_launcher import launch_isaac_sim

    simulation_app = None
    try:
        _app_launcher, simulation_app = launch_isaac_sim(headless=headless)

        import torch
        import warp as wp

        bundle = get_robot(robot_name)
        joint_names = robot_joint_names(bundle)
        run_cfg = bundle.run_config
        sim_cfg_data = run_cfg.simulation
        benchmark_cfg = run_cfg.benchmark
        real_init_pose_sync = bool(benchmark_cfg.real_init_pose_sync or False)
        timing = resolve_simulation_timing(sim_cfg_data, f"{bundle.spec.display_name} benchmark simulation")
        physics_dt = timing.physics_dt
        control_freq = timing.control_freq
        divisor = timing.control_divisor
        buffer_time = float(benchmark_cfg.buffer_time or 0.0)
        buffer_substeps = _buffer_substeps(buffer_time, timing.control_dt, timing.control_divisor)
        fix_root = bool(benchmark_cfg.fix_root if benchmark_cfg.fix_root is not None else bundle.spec.fixed_base)

        loaded_motions: list[_LoadedBenchmarkMotion] = []
        for spec in motions:
            motion_path = Path(spec.motion)
            source_motion = load_sage_motion(motion_path, joint_names=joint_names)
            source_initial = source_motion.initial_state() if real_init_pose_sync else None
            initial_state = (
                BenchmarkInitialState(
                    positions=source_initial.positions.tolist(),
                    velocities=source_initial.velocities.tolist() if source_initial.velocities is not None else None,
                )
                if source_initial is not None
                else None
            )
            real_data = source_motion.resample(1.0 / control_freq)
            commands = real_data.command_position.values.tolist()
            timestamps = real_data.command_position.time.tolist()
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
        _log(f"Starting {bundle.spec.display_name} benchmark: envs={batch_size}, motions={batch_size}")
        actual_num_envs = batch_size
        configured_batch_size = int(num_envs or batch_size)

        dataset_name = dataset or benchmark_cfg.dataset or benchmark_cfg.motion_name or "example"
        actuator_model_name = actuator_model or f"{robot_name}_implicit"
        output_root = Path(output)

        actuator_values = load_implicit_actuator_params(bundle, actuator_params)
        usd_path = resolve_runtime_asset(bundle)
        fix_root_override = resolve_fix_root_override(fix_root, bundle.spec.fixed_base)
        SceneCfg = build_scene_class(usd_path, actuator_values, joint_names, fix_root=fix_root_override)
        sim, scene, robot = create_newton_scene(
            SceneCfg, sim_cfg_data, timing, num_envs=batch_size
        )
        joint_ids = joint_ids_for(robot, joint_names, torch)
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

        sim_pos, sim_vel, sim_effort = _read_benchmark_state(robot, wp, joint_ids_long)
        for env_index, motion in enumerate(loaded_motions):
            _append_benchmark_sample(
                control_rows[env_index],
                state_rows[env_index],
                timestamp_s=motion.timestamps[0],
                command=motion.commands[0],
                position=sim_pos[env_index].tolist(),
                velocity=sim_vel[env_index].tolist(),
                effort=sim_effort[env_index].tolist(),
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
            sim_pos, sim_vel, sim_effort = _read_benchmark_state(robot, wp, joint_ids_long)
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
                    effort=sim_effort[env_index].tolist(),
                )

        first_sim_output: Path | None = None
        for env_index, motion in enumerate(loaded_motions):
            sim_output = output_root / "sim" / robot_name / dataset_name / actuator_model_name / motion.motion_name
            real_output = output_root / "real" / robot_name / dataset_name / motion.motion_name
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
                robot_name=robot_name,
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
