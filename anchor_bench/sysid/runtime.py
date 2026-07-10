from __future__ import annotations

from collections import deque
import csv
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from anchor_bench.core.run_config import resolve_simulation_timing
from anchor_bench.core.schemas import CMAESConfig, RobotRunFile, SysIdBoundsFile
from anchor_bench.data.sage import load_sage_motion
from anchor_bench.robots.registry import get_robot
from anchor_bench.runtime.assets import resolve_runtime_asset
from anchor_bench.runtime.robot_config import build_fitted_actuator_document, build_implicit_actuator_params, load_sysid_bounds
from anchor_bench.runtime.simulation import create_newton_scene, joint_ids_for
from anchor_bench.runtime.robot_scene import (
    _write_viscous_friction,
    build_scene_class,
    robot_joint_names,
    to_torch,
)
from anchor_bench.sysid.validation import (
    resolve_optional_positive_int as _resolve_optional_positive_int,
    resolve_positive_int as _resolve_positive_int,
)


def _log(message: str) -> None:
    print(f"[SYSID] {message}", flush=True)


def _write_summary(output: Path, run_cfg: RobotRunFile, bounds: SysIdBoundsFile, actuator_params: dict[str, Any], args: dict[str, Any], robot_name: str) -> None:
    summary = {
        "run_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "robot_name": robot_name,
        "simulation": run_cfg.simulation.to_dict(),
        "sysid": args,
        "bounds": {
            "parameters": {name: {"lower": r.lower, "upper": r.upper} for name, r in bounds.parameters.items()},
            "fixed": bounds.fixed,
        },
        "actuator_model": {"type": "implicit", "parameters": actuator_params},
    }
    (output / "run_summary.yaml").write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")


def _apply_population(robot, population: list[dict[str, dict[str, float]]], joint_ids, env_ids, torch, joint_names: list[str]) -> None:
    writers = {
        "armature": lambda vals: robot.write_joint_armature_to_sim_index(armature=vals, joint_ids=joint_ids, env_ids=env_ids),
        "dry_friction": lambda vals: robot.write_joint_friction_coefficient_to_sim_index(joint_friction_coeff=vals, joint_ids=joint_ids, env_ids=env_ids),
        "viscous_friction": lambda vals: _write_viscous_friction(robot, vals, joint_ids),
        "stiffness": lambda vals: robot.write_joint_stiffness_to_sim_index(stiffness=vals, joint_ids=joint_ids, env_ids=env_ids),
        "damping": lambda vals: robot.write_joint_damping_to_sim_index(damping=vals, joint_ids=joint_ids, env_ids=env_ids),
    }
    for prop, writer in writers.items():
        if prop not in population[0]:
            continue
        missing = [joint for joint in joint_names if joint not in population[0][prop]]
        if missing:
            raise ValueError(f"missing {prop} values for joints: {', '.join(missing)}")
        values = [[candidate[prop][joint] for joint in joint_names] for candidate in population]
        writer(torch.tensor(values, dtype=torch.float32, device=robot.device))


def _resolve_sysid_joint_names(bundle, configured: Any) -> list[str]:
    all_joints = robot_joint_names(bundle)
    if configured is None or configured == "full":
        return all_joints
    if isinstance(configured, str):
        requested = [configured]
    elif isinstance(configured, list):
        requested = configured
    else:
        raise ValueError("sysid.joints must be 'full' or a list of joint names")
    if not requested:
        raise ValueError("sysid.joints must select at least one joint")
    aliases = dict(getattr(bundle.spec, "joint_aliases", {}) or {})
    selected: list[str] = []
    for item in requested:
        joint = aliases.get(str(item), str(item))
        if joint not in all_joints:
            raise ValueError(f"unknown sysid joint '{item}'")
        if joint not in selected:
            selected.append(joint)
    return selected


def run_robot_sysid(
    robot_name: str,
    data: str,
    output: str,
    headless: bool = False,
    num_envs: int | None = None,
    max_iter: int | None = None,
    max_trajectory_len: int | None = None,
    seed: int | None = None,
    config_sources: dict[str, str] | None = None,
) -> Path:
    from anchor_bench.runtime.app_launcher import launch_isaac_sim

    bundle = get_robot(robot_name)
    run_cfg = bundle.run_config
    sim_cfg_data = run_cfg.simulation
    sysid_cfg = run_cfg.sysid
    joint_names = _resolve_sysid_joint_names(bundle, sysid_cfg.joints or "full")
    all_joint_names = robot_joint_names(bundle)
    bounds = load_sysid_bounds(bundle).model_copy(update={"joint_types": joint_names})
    cmaes_cfg = sysid_cfg.cmaes or CMAESConfig()

    num_envs = _resolve_positive_int("num_envs", num_envs, sysid_cfg.num_envs or 64)
    max_iter = _resolve_positive_int("max_iter", max_iter, cmaes_cfg.max_iterations)
    max_trajectory_len = _resolve_optional_positive_int("max_trajectory_len", max_trajectory_len)
    seed = int(seed if seed is not None else 0)
    timing = resolve_simulation_timing(sim_cfg_data, f"{bundle.spec.display_name} sysid simulation")
    physics_freq = timing.physics_freq
    control_freq = timing.control_freq
    physics_dt = timing.physics_dt
    control_dt = timing.control_dt
    divisor = timing.control_divisor
    buffer_time = float(sysid_cfg.buffer_time or 0.0)
    if buffer_time < 0.0:
        raise ValueError("sysid.buffer_time must be >= 0")

    simulation_app = None
    try:
        _app_launcher, simulation_app = launch_isaac_sim(headless=headless)

        import torch
        import warp as wp

        output_path = Path(output)
        output_path.mkdir(parents=True, exist_ok=True)

        dataset = load_sage_motion(data, target_dt=control_dt, joint_names=joint_names)
        commands = dataset.command_position.values.tolist()
        measured = dataset.position.values.tolist()
        if max_trajectory_len is not None:
            commands = commands[:max_trajectory_len]
            measured = measured[:max_trajectory_len]
        if not commands:
            raise ValueError(f"No SAGE rows found in {data}")

        _log(f"Starting {bundle.spec.display_name} SysID: envs={num_envs}, rows={len(commands)}, max_iter={max_iter}")

        actuator_params = build_implicit_actuator_params(bundle)
        motor_lag_ms = float(actuator_params.get("motor_lag_ms", bounds.fixed.get("motor_lag_ms", 0.0)))
        motor_lag_steps = int(round((motor_lag_ms / 1000.0) / control_dt))
        usd_path = resolve_runtime_asset(bundle)
        SceneCfg = build_scene_class(usd_path, actuator_params, all_joint_names)
        _log(f"Using USD asset: {usd_path}")

        sim, scene, robot = create_newton_scene(
            SceneCfg, sim_cfg_data, timing, num_envs=num_envs
        )
        joint_pos_template = torch.zeros((num_envs, len(robot.joint_names)), dtype=torch.float32, device=robot.device)
        joint_vel_template = torch.zeros_like(joint_pos_template)
        joint_ids = joint_ids_for(robot, joint_names, torch)
        joint_ids_long = joint_ids.long()
        env_ids = torch.arange(num_envs, dtype=torch.int32, device=robot.device)
        commands_t = torch.tensor(commands, dtype=torch.float32, device=robot.device)
        start = commands_t[0]
        measured_t = torch.tensor(measured, dtype=torch.float32, device=robot.device)
        target_buffer = torch.zeros_like(joint_pos_template)

        from anchor_bench.sysid.optimizer import CMAESOptimizer

        optimizer = CMAESOptimizer(bounds, num_envs=num_envs, cmaes=cmaes_cfg, seed=seed)
        _write_summary(
            output_path,
            run_cfg,
            bounds,
            actuator_params,
            {
                "real_data_dir": data,
                "output_dir": str(output_path),
                "num_envs": num_envs,
                "max_iter": max_iter,
                "max_trajectory_len": max_trajectory_len,
                "seed": seed,
                "physics_freq": physics_freq,
                "control_freq": control_freq,
                "render_freq": timing.render_freq,
                "render_interval": timing.render_interval,
                "buffer_time": buffer_time,
                "motor_lag_ms": motor_lag_ms,
                "joints": joint_names,
                "config_sources": dict(config_sources or {}),
                "cmaes": {
                    "sigma": cmaes_cfg.sigma,
                    "max_iterations": cmaes_cfg.max_iterations,
                    "epsilon": cmaes_cfg.epsilon,
                },
            },
            robot_name=robot_name,
        )
        log_path = output_path / "optimization_log.csv"
        with log_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["generation", "best_mse", "mean_mse"])

        buffer_steps = int(round(buffer_time / control_dt))
        best_score = float("inf")
        best_params: dict[str, Any] = {}
        start_broadcast = start.unsqueeze(0).expand(num_envs, -1)
        joint_pos = torch.zeros_like(joint_pos_template)
        joint_vel = torch.zeros_like(joint_vel_template)
        for generation in range(max_iter):
            population = optimizer.sample_population()
            _apply_population(robot, population, joint_ids, env_ids, torch, joint_names)

            joint_pos.zero_()
            joint_vel.zero_()
            joint_pos[:, joint_ids_long] = start_broadcast
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(physics_dt)

            target_buffer.zero_()
            target_buffer[:, joint_ids_long] = start_broadcast
            robot.set_joint_position_target(target_buffer)
            for _ in range(buffer_steps * divisor):
                scene.write_data_to_sim()
                sim.step(render=False)
                scene.update(physics_dt)

            if motor_lag_steps > 0:
                cmd_buffer = deque(maxlen=motor_lag_steps + 1)
                for _ in range(motor_lag_steps):
                    cmd_buffer.append(start)
            else:
                cmd_buffer = None

            scores = torch.zeros(num_envs, dtype=torch.float32, device=robot.device)
            for index in range(commands_t.shape[0]):
                command_t = commands_t[index]
                if cmd_buffer is not None:
                    cmd_buffer.append(command_t)
                    delayed_command = cmd_buffer[0]
                else:
                    delayed_command = command_t
                target_buffer[:, joint_ids_long] = delayed_command.unsqueeze(0).expand(num_envs, -1)
                robot.set_joint_position_target(target_buffer)
                for _ in range(divisor):
                    scene.write_data_to_sim()
                    sim.step(render=False)
                    scene.update(physics_dt)
                sim_pos = to_torch(robot.data.joint_pos, wp)[:, joint_ids_long]
                diff = sim_pos - measured_t[index].unsqueeze(0)
                scores += (diff * diff).sum(dim=1)
            mean_scores = scores / max(1, len(commands))
            converged = optimizer.tell(mean_scores.detach().cpu().numpy())
            if optimizer.best_params is not None:
                best_score = optimizer.best_score
                best_params = optimizer.best_params
            with log_path.open("a", newline="", encoding="utf-8") as stream:
                csv.writer(stream).writerow([generation, best_score, float(mean_scores.mean().item())])
            _log(f"generation {generation}: best_mse={best_score:.6f}")
            if converged:
                _log(f"converged after generation {generation}")
                break

        if optimizer.best_params is None:
            raise RuntimeError(
                "SysID optimizer never produced a finite best score "
                f"({max_iter} generations exhausted). Check for NaN MSE / sim divergence "
                "before consuming best_params.yaml."
            )
        document = build_fitted_actuator_document(
            bundle,
            best_params,
            metadata={
                "source": "sysid",
                "best_mse": best_score,
                "robot_name": robot_name,
                "mode": "all_joints" if joint_names == all_joint_names else "selected_joints",
                "joints": joint_names,
                "data_source": data,
            },
        )
        (output_path / "best_params.yaml").write_text(
            yaml.safe_dump(document, sort_keys=False),
            encoding="utf-8",
        )
        _log(f"Best parameters saved to {output_path / 'best_params.yaml'}")
        return output_path
    finally:
        if simulation_app is not None:
            simulation_app.close()
