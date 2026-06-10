from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from anchor_bench.data.sage import load_initial_state


@dataclass(frozen=True)
class BenchmarkInitialState:
    positions: list[float]
    velocities: list[float] | None = None


def load_benchmark_initial_state(motion_path: str | Path, joint_names: list[str]) -> BenchmarkInitialState:
    state = load_initial_state(motion_path, joint_names)
    return BenchmarkInitialState(positions=state.positions, velocities=state.velocities)


def apply_benchmark_initial_states(
    robot: Any,
    *,
    to_torch_fn: Callable[[Any, Any], Any],
    torch_module: Any,
    warp_module: Any,
    joint_ids_long: Any,
    joint_names: list[str],
    initial_states: list[BenchmarkInitialState | None],
) -> int:
    """Teleport scored joints to row-0 real state for each benchmark env."""
    if not initial_states:
        return 0
    missing = [idx for idx, state in enumerate(initial_states) if state is None]
    if missing:
        raise ValueError(f"Missing benchmark init states for env indices: {missing}")
    for idx, state in enumerate(initial_states):
        assert state is not None
        if len(state.positions) != len(joint_names):
            raise ValueError(
                f"benchmark init state {idx} has {len(state.positions)} positions "
                f"but runtime tracks {len(joint_names)} joints"
            )
        if state.velocities is not None and len(state.velocities) != len(joint_names):
            raise ValueError(
                f"benchmark init state {idx} has {len(state.velocities)} velocities "
                f"but runtime tracks {len(joint_names)} joints"
            )

    full_pos = to_torch_fn(robot.data.joint_pos, warp_module).clone()
    full_vel = to_torch_fn(robot.data.joint_vel, warp_module).clone()
    if full_pos.shape[0] != len(initial_states):
        raise ValueError(f"benchmark init state count {len(initial_states)} does not match env count {full_pos.shape[0]}")

    states = [state for state in initial_states if state is not None]
    init_pos_t = torch_module.tensor([state.positions for state in states], dtype=full_pos.dtype, device=robot.device)
    full_pos[:, joint_ids_long] = init_pos_t
    if any(state.velocities is not None for state in states):
        init_vel_t = torch_module.tensor(
            [state.velocities or [0.0] * len(joint_names) for state in states],
            dtype=full_vel.dtype,
            device=robot.device,
        )
        full_vel[:, joint_ids_long] = init_vel_t
    robot.write_joint_state_to_sim(full_pos, full_vel)
    return len(states)
