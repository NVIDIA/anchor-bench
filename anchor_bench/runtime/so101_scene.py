from __future__ import annotations

from pathlib import Path
from typing import Any


def so101_joint_names(bundle) -> list[str]:
    joint_names = list(bundle.spec.joint_names)
    if not joint_names:
        raise ValueError("SO-101 robot config must define joint_names")
    return joint_names


def to_torch(value, wp):
    return wp.to_torch(value) if isinstance(value, wp.array) else value


def _joint_ids_to_list(joint_ids) -> list[int]:
    if hasattr(joint_ids, "detach"):
        return [int(value) for value in joint_ids.detach().cpu().tolist()]
    return [int(value) for value in joint_ids]


def _get_newton_model_for_viscous_friction():
    try:
        from isaaclab_newton.physics import NewtonManager
    except Exception as exc:
        raise RuntimeError("NewtonManager is required to write viscous_friction") from exc

    try:
        newton_model = NewtonManager.get_model()
    except Exception as exc:
        raise RuntimeError("NewtonManager.get_model() failed while writing viscous_friction") from exc
    if newton_model is None:
        raise RuntimeError("NewtonManager.get_model() returned no model while writing viscous_friction")

    mujoco_ns = getattr(newton_model, "mujoco", None)
    if mujoco_ns is None or not hasattr(mujoco_ns, "dof_passive_damping"):
        raise RuntimeError("Newton model has no mujoco.dof_passive_damping for viscous_friction")
    return NewtonManager, newton_model


def _write_viscous_friction(robot, values, joint_ids) -> None:
    NewtonManager, newton_model = _get_newton_model_for_viscous_friction()
    try:
        from newton.solvers import SolverNotifyFlags
    except Exception as exc:
        raise RuntimeError("Newton SolverNotifyFlags is required to notify viscous_friction changes") from exc

    ids = _joint_ids_to_list(joint_ids)
    vals_np = values.detach().cpu().numpy()
    damping = newton_model.mujoco.dof_passive_damping
    damping_np = damping.numpy()
    num_dofs = len(robot.joint_names)
    if damping_np.ndim == 2:
        for env_i in range(vals_np.shape[0]):
            for local_j, joint_id in enumerate(ids):
                damping_np[env_i, joint_id] = vals_np[env_i, local_j]
    else:
        for env_i in range(vals_np.shape[0]):
            for local_j, joint_id in enumerate(ids):
                damping_np[env_i * num_dofs + joint_id] = vals_np[env_i, local_j]
    damping.assign(damping_np)
    NewtonManager.add_model_change(SolverNotifyFlags.JOINT_DOF_PROPERTIES)


def _matrix_from_param_map(actuator_params: dict[str, Any], prop: str, joint_names: list[str], num_envs: int, torch, device):
    mapping = actuator_params.get(prop)
    if mapping is None:
        return None
    missing = [joint for joint in joint_names if joint not in mapping]
    if missing:
        raise ValueError(f"missing {prop} values for joints: {', '.join(missing)}")
    values = [[float(mapping[joint]) for joint in joint_names] for _ in range(num_envs)]
    return torch.tensor(values, dtype=torch.float32, device=device)


def apply_static_actuator_params(robot, actuator_params: dict[str, Any], joint_ids, env_ids, torch, joint_names: list[str] | None = None) -> None:
    num_envs = int(env_ids.numel())
    if joint_names is None:
        ids = _joint_ids_to_list(joint_ids)
        joint_names = [robot.joint_names[joint_id] for joint_id in ids]
    static_writers = {
        "armature": lambda vals: robot.write_joint_armature_to_sim_index(armature=vals, joint_ids=joint_ids, env_ids=env_ids),
        "dry_friction": lambda vals: robot.write_joint_friction_coefficient_to_sim_index(joint_friction_coeff=vals, joint_ids=joint_ids, env_ids=env_ids),
        "viscous_friction": lambda vals: _write_viscous_friction(robot, vals, joint_ids),
        "stiffness": lambda vals: robot.write_joint_stiffness_to_sim_index(stiffness=vals, joint_ids=joint_ids, env_ids=env_ids),
        "damping": lambda vals: robot.write_joint_damping_to_sim_index(damping=vals, joint_ids=joint_ids, env_ids=env_ids),
    }
    for prop, writer in static_writers.items():
        values = _matrix_from_param_map(actuator_params, prop, joint_names, num_envs, torch, robot.device)
        if values is None:
            continue
        writer(values)


def _isaac_implicit_actuator_kwargs(actuator_params: dict[str, Any]) -> dict[str, Any]:
    """Map Anchor Bench actuator params to IsaacLab ImplicitActuatorCfg kwargs."""
    return {
        "joint_names_expr": [".*"],
        "effort_limit_sim": actuator_params.get("effort_limit_sim"),
        "velocity_limit_sim": actuator_params.get("velocity_limit_sim"),
        "stiffness": actuator_params["stiffness"],
        "damping": actuator_params["damping"],
        "armature": actuator_params.get("armature"),
        "dynamic_friction": actuator_params.get("dry_friction"),
        "viscous_friction": actuator_params.get("viscous_friction"),
    }


def build_scene_class(
    usd_path: Path,
    actuator_params: dict[str, Any],
    joint_names: list[str],
    fix_root: bool | None = None,
):
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg, AssetBaseCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.utils import configclass as _configclass

    from anchor_bench.runtime.actuator_net_gru import _coerce_configclass

    configclass = _coerce_configclass(_configclass)
    implicit_actuator_kwargs = _isaac_implicit_actuator_kwargs(actuator_params)

    @configclass
    class So101SysidSceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(
            prim_path="/World/ground",
            spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
        )

        robot: ArticulationCfg = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(usd_path),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(fix_root_link=fix_root)
                if fix_root is not None
                else None,
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.0),
                joint_pos={name: 0.0 for name in joint_names},
                joint_vel={".*": 0.0},
            ),
            actuators={
                "all": ImplicitActuatorCfg(**implicit_actuator_kwargs)
            },
        )

        dome_light = AssetBaseCfg(
            prim_path="/World/DomeLight",
            spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
        )

    return So101SysidSceneCfg
