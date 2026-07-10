"""Lazy Isaac Lab/Newton simulation construction for authoritative phases."""

from __future__ import annotations

from typing import Any, Protocol


class NewtonSimulationConfig(Protocol):
    """Newton solver settings required to construct an Isaac Lab scene."""

    integrator: str
    solver_iterations: int
    line_search_iterations: int


def create_newton_scene(
    scene_config: Any,
    simulation_config: NewtonSimulationConfig,
    timing: Any,
    *,
    num_envs: int,
) -> tuple[Any, Any, Any]:
    """Create a Newton-backed Isaac Lab scene after AppLauncher has started."""

    from isaaclab.scene import InteractiveScene
    from isaaclab.sim import SimulationCfg, SimulationContext
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

    sim_cfg = SimulationCfg(
        dt=timing.physics_dt,
        render_interval=timing.render_interval,
        gravity=(0.0, 0.0, -9.81),
    )
    sim_cfg.physics = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            integrator=simulation_config.integrator,
            ls_iterations=simulation_config.line_search_iterations,
            iterations=simulation_config.solver_iterations,
        )
    )
    simulation = SimulationContext(sim_cfg)
    scene = InteractiveScene(scene_config(num_envs=num_envs, env_spacing=4.0))
    simulation.reset()
    scene.reset()
    return simulation, scene, scene["robot"]


def joint_ids_for(robot: Any, joint_names: list[str], torch: Any) -> Any:
    """Return articulation indices in requested order or report missing joints."""

    indices_by_name = {name: index for index, name in enumerate(robot.joint_names)}
    missing = [name for name in joint_names if name not in indices_by_name]
    if missing:
        raise ValueError(f"runtime articulation is missing joints: {', '.join(missing)}")
    indices = [indices_by_name[name] for name in joint_names]
    return torch.tensor(indices, dtype=torch.int32, device=robot.device)
