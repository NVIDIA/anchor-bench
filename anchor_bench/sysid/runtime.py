"""Lazy SysID optimization backend for an already resolved immutable run.

This module consumes only :class:`ResolvedSysIdRun`.  It never rereads legacy
run, bounds, or actuator files and never writes final run artifacts.  The
optimization loop is host-testable through an injected simulation adapter;
Isaac Sim, Isaac Lab, Newton, Warp, and Torch are imported only by the default
adapter when a real run starts.
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from anchor_bench.core.paths import repo_root, robot_config_dir
from anchor_bench.core.profiles import load_yaml_mapping
from anchor_bench.core.timing import resolve_simulation_timing
from anchor_bench.core.hashing import sha256_file, sha256_tree, validate_sha256
from anchor_bench.robots.schema import RobotIdentity
from anchor_bench.sysid.runner import ResolvedSysIdRun, SysIdExecutionResult
from anchor_bench.sysid.schema import SUPPORTED_SYSID_PARAMETERS, parameter_layout


ParameterState = Mapping[str, Mapping[str, float]]


@dataclass(frozen=True)
class PopulationEvaluation:
    """Unweighted per-environment trajectory errors from one population."""

    position_mse: Sequence[float]
    velocity_mse: Sequence[float]

    def validated(self, num_envs: int) -> tuple[np.ndarray, np.ndarray]:
        """Return finite, non-negative component arrays of the expected width."""

        return (
            _validated_error_component(
                self.position_mse,
                name="position_mse",
                num_envs=num_envs,
            ),
            _validated_error_component(
                self.velocity_mse,
                name="velocity_mse",
                num_envs=num_envs,
            ),
        )


class SysIdRuntimeAdapter(Protocol):
    """Simulator boundary required by the deterministic optimization loop."""

    def evaluate_population(
        self,
        population: Sequence[ParameterState],
    ) -> PopulationEvaluation:
        """Evaluate complete parameter states and return unweighted errors."""

    def close(self) -> None:
        """Release simulator resources, including a launched application."""


class SysIdOptimizer(Protocol):
    """Small optimizer surface consumed by :func:`optimize_resolved_sysid`."""

    best_params: dict[str, dict[str, float]] | None
    best_score: float
    generation: int

    @property
    def normalized_mean(self) -> list[float]:
        """Return the optimizer's normalized starting point."""

    def sample_population(self) -> list[dict[str, dict[str, float]]]:
        """Sample one sparse fitted patch per environment."""

    def tell(self, scores: Sequence[float]) -> bool:
        """Consume weighted losses and report convergence."""


RuntimeAdapterFactory = Callable[[ResolvedSysIdRun], SysIdRuntimeAdapter]
OptimizerFactory = Callable[[ResolvedSysIdRun], SysIdOptimizer]


def _validated_error_component(
    values: Sequence[float],
    *,
    name: str,
    num_envs: int,
) -> np.ndarray:
    """Validate one population error component returned by a simulator."""

    array = np.asarray(values, dtype=np.float64)
    if array.shape != (num_envs,):
        raise ValueError(
            f"{name} must have shape ({num_envs},), got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(array < 0.0):
        raise ValueError(f"{name} must contain only non-negative MSE values")
    return array


def _complete_population(
    resolved: ResolvedSysIdRun,
    fitted_population: Sequence[Mapping[str, Mapping[str, float]]],
) -> list[dict[str, dict[str, float]]]:
    """Overlay each fitted patch on the exact selected parent parameter state."""

    complete_population: list[dict[str, dict[str, float]]] = []
    for fitted in fitted_population:
        patch = resolved.validate_fitted_parameters(fitted)
        complete = {
            parameter: {joint: float(value) for joint, value in joint_map.items()}
            for parameter, joint_map in resolved.parent_state.parameters.items()
        }
        for parameter, joint_map in patch.items():
            complete.setdefault(parameter, {}).update(joint_map)
        complete_population.append(complete)
    return complete_population


def _default_optimizer_factory(resolved: ResolvedSysIdRun) -> SysIdOptimizer:
    """Construct CMA-ES from the resolved profile and exact parent state."""

    from anchor_bench.sysid.optimizer import CMAESOptimizer

    config = resolved.resolved_config
    return CMAESOptimizer(
        config,
        num_envs=config.num_envs,
        cmaes=config.optimizer,
        seed=config.seed,
        parent_parameters=resolved.parent_state.parameters,
    )


def _history_csv(history: Sequence[Mapping[str, float | int]]) -> str:
    """Serialize deterministic per-generation optimizer evidence as CSV."""

    stream = StringIO(newline="")
    fieldnames = [
        "generation",
        "best_loss",
        "mean_loss",
        "best_position_mse",
        "best_velocity_mse",
    ]
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in history:
        writer.writerow(
            {
                key: value if key == "generation" else format(float(value), ".17g")
                for key, value in row.items()
            }
        )
    return stream.getvalue()


def _optimizer_json(
    resolved: ResolvedSysIdRun,
    optimizer: SysIdOptimizer,
    *,
    converged: bool,
) -> str:
    """Serialize the optimizer layout and termination state as canonical JSON."""

    config = resolved.resolved_config
    document = {
        "schema_version": 1,
        "type": config.optimizer.type,
        "seed": config.seed,
        "num_envs": config.num_envs,
        "sigma": config.optimizer.sigma,
        "max_iterations": config.optimizer.max_iterations,
        "epsilon": config.optimizer.epsilon,
        "parameter_layout": [entry.name for entry in parameter_layout(config)],
        "normalized_parent_mean": optimizer.normalized_mean,
        "generations_completed": optimizer.generation,
        "converged": converged,
    }
    return json.dumps(
        document,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ) + "\n"


def optimize_resolved_sysid(
    resolved: ResolvedSysIdRun,
    *,
    adapter_factory: RuntimeAdapterFactory | None = None,
    optimizer_factory: OptimizerFactory | None = None,
) -> SysIdExecutionResult:
    """Optimize one resolved SysID recipe and return unwritten run artifacts."""

    config = resolved.resolved_config
    optimizer = (optimizer_factory or _default_optimizer_factory)(resolved)
    adapter = (adapter_factory or _default_adapter_factory)(resolved)
    history: list[dict[str, float | int]] = []
    best_loss = math.inf
    best_position_mse = math.inf
    best_velocity_mse = math.inf
    converged = False
    try:
        for generation in range(config.optimizer.max_iterations):
            fitted_population = optimizer.sample_population()
            if len(fitted_population) != config.num_envs:
                raise ValueError(
                    "optimizer population width does not match resolved num_envs: "
                    f"{len(fitted_population)} != {config.num_envs}"
                )
            complete_population = _complete_population(resolved, fitted_population)
            evaluation = adapter.evaluate_population(complete_population)
            if not isinstance(evaluation, PopulationEvaluation):
                raise TypeError(
                    "SysID runtime adapter must return PopulationEvaluation"
                )
            position_mse, velocity_mse = evaluation.validated(config.num_envs)
            losses = (
                float(config.loss.position) * position_mse
                + float(config.loss.velocity) * velocity_mse
            )
            generation_best = int(np.argmin(losses))
            generation_best_loss = float(losses[generation_best])
            if generation_best_loss < best_loss:
                best_loss = generation_best_loss
                best_position_mse = float(position_mse[generation_best])
                best_velocity_mse = float(velocity_mse[generation_best])
            converged = bool(optimizer.tell(losses))
            history.append(
                {
                    "generation": generation,
                    "best_loss": best_loss,
                    "mean_loss": float(np.mean(losses)),
                    "best_position_mse": best_position_mse,
                    "best_velocity_mse": best_velocity_mse,
                }
            )
            if converged:
                break
    finally:
        adapter.close()

    if optimizer.best_params is None or not math.isfinite(best_loss):
        raise RuntimeError("SysID optimizer did not produce a finite fitted result")
    if not math.isfinite(float(optimizer.best_score)) or not math.isclose(
        float(optimizer.best_score),
        best_loss,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            "SysID optimizer best score disagrees with evaluated trajectory loss"
        )
    fitted = resolved.validate_fitted_parameters(optimizer.best_params)
    metrics = {
        "best_loss": best_loss,
        "best_position_mse": best_position_mse,
        "best_velocity_mse": best_velocity_mse,
        "generations_completed": optimizer.generation,
        "converged": converged,
        "num_envs": config.num_envs,
        "loss_weights": {
            "position": float(config.loss.position),
            "velocity": float(config.loss.velocity),
        },
    }
    return SysIdExecutionResult(
        fitted_parameters=fitted,
        metrics=metrics,
        optimizer_artifacts={
            "history.csv": _history_csv(history),
            "optimizer.json": _optimizer_json(
                resolved,
                optimizer,
                converged=converged,
            ),
        },
    )


def _load_robot_identity(robot: str) -> RobotIdentity:
    """Load robot identity without validating legacy phase configuration files."""

    path = robot_config_dir(robot) / "robot.yaml"
    if not path.is_file():
        raise KeyError(f"unknown robot {robot!r}")
    spec = RobotIdentity.from_dict(load_yaml_mapping(path), path)
    if spec.name != robot:
        raise ValueError(
            f"robot identity mismatch: requested {robot!r}, robot.yaml names {spec.name!r}"
        )
    return spec


def _resolve_runtime_usd(spec: RobotIdentity) -> Path:
    """Resolve the declarative runtime USD without reading actuator or bounds files."""

    if not spec.runtime_usd:
        raise FileNotFoundError(
            f"robot {spec.name!r} has no runtime_usd configured in robot.yaml"
        )
    manifest = repo_root() / spec.asset_path
    roots = [manifest.parent / "files"]
    configured_root = os.environ.get("ANCHOR_BENCH_ASSET_DIR")
    if configured_root:
        external = Path(configured_root).expanduser()
        roots = [external, external / spec.name, *roots]
    for root in roots:
        candidate = root / spec.runtime_usd
        if candidate.is_file():
            return candidate
    searched = ", ".join(str(root / spec.runtime_usd) for root in roots)
    raise FileNotFoundError(
        f"missing runtime USD {spec.runtime_usd!r} for {spec.name}; searched: {searched}"
    )


def _fingerprint_digest(
    fingerprint: Mapping[str, Any],
    *,
    label: str,
) -> str:
    """Read one required raw or tree digest from a resolved fingerprint."""

    for key in ("sha256", f"{label}_sha256", "tree_sha256"):
        value = fingerprint.get(key)
        if isinstance(value, str):
            return validate_sha256(value, label=f"{label} fingerprint {key}")
    raise ValueError(f"resolved {label} fingerprint is missing its content digest")


def _verify_runtime_inputs(
    resolved: ResolvedSysIdRun,
    usd_path: Path,
) -> None:
    """Fail if asset or dataset bytes changed after semantic resolution."""

    expected_asset = _fingerprint_digest(
        resolved.fingerprints["asset"],
        label="asset",
    )
    actual_asset = sha256_file(usd_path)
    if actual_asset != expected_asset:
        raise RuntimeError(
            "SysID runtime asset changed after resolution: "
            f"expected {expected_asset}, got {actual_asset}"
        )
    dataset = repo_root() / resolved.resolved_config.data
    expected_dataset = _fingerprint_digest(
        resolved.fingerprints["dataset"],
        label="dataset",
    )
    actual_dataset = sha256_tree(dataset)
    if actual_dataset != expected_dataset:
        raise RuntimeError(
            "SysID dataset changed after resolution: "
            f"expected {expected_dataset}, got {actual_dataset}"
        )


def _base_actuator_parameters(
    resolved: ResolvedSysIdRun,
) -> dict[str, dict[str, float]]:
    """Build the simulator's initial PD state only from the selected parent."""

    parent = resolved.parent_state.parameters
    selected = resolved.resolved_config.joints
    missing_parameters = [name for name in ("stiffness", "damping") if name not in parent]
    if missing_parameters:
        raise ValueError(
            "parent SysID state is missing simulator PD parameter(s): "
            + ", ".join(missing_parameters)
        )
    for parameter in ("stiffness", "damping"):
        missing = [joint for joint in selected if joint not in parent[parameter]]
        if missing:
            raise ValueError(
                f"parent SysID state is missing {parameter} cells: {', '.join(missing)}"
            )
    return {
        parameter: {joint: float(value) for joint, value in joint_map.items()}
        for parameter, joint_map in parent.items()
        if parameter in SUPPORTED_SYSID_PARAMETERS
    }


def _apply_population(
    robot: Any,
    population: Sequence[ParameterState],
    *,
    parameter_names: Sequence[str],
    joint_names: Sequence[str],
    joint_ids: Any,
    env_ids: Any,
    torch: Any,
) -> None:
    """Write one candidate's configured physical cells to each environment."""

    from anchor_bench.runtime.robot_scene import _write_viscous_friction

    writers = {
        "armature": lambda values: robot.write_joint_armature_to_sim_index(
            armature=values,
            joint_ids=joint_ids,
            env_ids=env_ids,
        ),
        "dry_friction": lambda values: (
            robot.write_joint_friction_coefficient_to_sim_index(
                joint_friction_coeff=values,
                joint_ids=joint_ids,
                env_ids=env_ids,
            )
        ),
        "viscous_friction": lambda values: _write_viscous_friction(
            robot,
            values,
            joint_ids,
        ),
        "stiffness": lambda values: robot.write_joint_stiffness_to_sim_index(
            stiffness=values,
            joint_ids=joint_ids,
            env_ids=env_ids,
        ),
        "damping": lambda values: robot.write_joint_damping_to_sim_index(
            damping=values,
            joint_ids=joint_ids,
            env_ids=env_ids,
        ),
    }
    for parameter in parameter_names:
        writer = writers[parameter]
        rows: list[list[float]] = []
        for candidate in population:
            joint_map = candidate.get(parameter)
            if joint_map is None:
                raise ValueError(f"candidate is missing parameter {parameter!r}")
            missing = [joint for joint in joint_names if joint not in joint_map]
            if missing:
                raise ValueError(
                    f"candidate is missing {parameter} cells: {', '.join(missing)}"
                )
            rows.append([float(joint_map[joint]) for joint in joint_names])
        writer(torch.tensor(rows, dtype=torch.float32, device=robot.device))


class IsaacNewtonRuntimeAdapter:
    """Real batched Isaac Lab/Newton trajectory evaluator for one resolved run."""

    def __init__(self, resolved: ResolvedSysIdRun) -> None:
        """Launch the runtime lazily and allocate one reusable batched scene."""

        self._application: Any | None = None
        try:
            self._initialize(resolved)
        except BaseException:
            self.close()
            raise

    def _initialize(self, resolved: ResolvedSysIdRun) -> None:
        """Load immutable inputs and create the simulator after AppLauncher starts."""

        from anchor_bench.data.sage import load_sage_motion
        from anchor_bench.runtime.app_launcher import launch_isaac_sim

        config = resolved.resolved_config
        spec = _load_robot_identity(resolved.robot)
        usd_path = _resolve_runtime_usd(spec)
        _verify_runtime_inputs(resolved, usd_path)
        timing = resolve_simulation_timing(
            config.simulation,
            f"{resolved.robot} resolved SysID simulation",
        )
        motion = load_sage_motion(
            repo_root() / config.data,
            joint_names=config.joints,
            target_dt=timing.control_dt,
            require_velocity=config.loss.velocity > 0.0,
        )
        if not len(motion.position.time):
            raise ValueError(f"SysID dataset {config.data!r} contains no state rows")
        actuator_parameters = _base_actuator_parameters(resolved)

        _launcher, self._application = launch_isaac_sim(headless=True)

        import torch
        import warp as wp

        from anchor_bench.runtime.robot_scene import build_scene_class, to_torch
        from anchor_bench.runtime.simulation import create_newton_scene, joint_ids_for

        scene_class = build_scene_class(
            usd_path,
            actuator_parameters,
            list(spec.joint_names),
            fix_root=spec.fixed_base,
        )
        simulation, scene, robot = create_newton_scene(
            scene_class,
            config.simulation,
            timing,
            num_envs=config.num_envs,
        )

        self._resolved = resolved
        self._timing = timing
        self._torch = torch
        self._wp = wp
        self._to_torch = to_torch
        self._simulation = simulation
        self._scene = scene
        self._robot = robot
        self._joint_names = tuple(config.joints)
        self._joint_ids = joint_ids_for(robot, list(config.joints), torch)
        self._joint_ids_long = self._joint_ids.long()
        self._env_ids = torch.arange(
            config.num_envs,
            dtype=torch.int32,
            device=robot.device,
        )
        self._commands = torch.tensor(
            motion.command_position.values,
            dtype=torch.float32,
            device=robot.device,
        )
        self._measured_position = torch.tensor(
            motion.position.values,
            dtype=torch.float32,
            device=robot.device,
        )
        self._measured_velocity = (
            torch.tensor(
                motion.velocity.values,
                dtype=torch.float32,
                device=robot.device,
            )
            if motion.velocity is not None
            else None
        )
        self._joint_position = torch.zeros(
            (config.num_envs, len(robot.joint_names)),
            dtype=torch.float32,
            device=robot.device,
        )
        self._joint_velocity = torch.zeros_like(self._joint_position)
        self._target = torch.zeros_like(self._joint_position)

    def _reset_trajectory(self) -> None:
        """Reset every environment to the measured initial state and settle it."""

        config = self._resolved.resolved_config
        torch = self._torch
        start_position = self._measured_position[0].unsqueeze(0).expand(
            config.num_envs,
            -1,
        )
        self._joint_position.zero_()
        self._joint_velocity.zero_()
        self._joint_position[:, self._joint_ids_long] = start_position
        if self._measured_velocity is not None:
            start_velocity = self._measured_velocity[0].unsqueeze(0).expand(
                config.num_envs,
                -1,
            )
            self._joint_velocity[:, self._joint_ids_long] = start_velocity
        self._robot.write_joint_state_to_sim(
            self._joint_position,
            self._joint_velocity,
        )
        self._scene.write_data_to_sim()
        self._simulation.step(render=False)
        self._scene.update(self._timing.physics_dt)

        first_command = self._commands[0].unsqueeze(0).expand(config.num_envs, -1)
        self._target.zero_()
        self._target[:, self._joint_ids_long] = first_command
        self._robot.set_joint_position_target(self._target)
        buffer_steps = int(round(config.buffer_time / self._timing.control_dt))
        for _ in range(buffer_steps * self._timing.control_divisor):
            self._scene.write_data_to_sim()
            self._simulation.step(render=False)
            self._scene.update(self._timing.physics_dt)

    def evaluate_population(
        self,
        population: Sequence[ParameterState],
    ) -> PopulationEvaluation:
        """Simulate direct commands with no additional motor-delay mechanism."""

        config = self._resolved.resolved_config
        if len(population) != config.num_envs:
            raise ValueError(
                f"population must contain {config.num_envs} candidates, got {len(population)}"
            )
        _apply_population(
            self._robot,
            population,
            parameter_names=tuple(config.parameters),
            joint_names=self._joint_names,
            joint_ids=self._joint_ids,
            env_ids=self._env_ids,
            torch=self._torch,
        )
        self._reset_trajectory()

        position_error = self._torch.zeros(
            config.num_envs,
            dtype=self._torch.float64,
            device=self._robot.device,
        )
        velocity_error = self._torch.zeros_like(position_error)
        for row in range(self._commands.shape[0]):
            command = self._commands[row].unsqueeze(0).expand(config.num_envs, -1)
            self._target[:, self._joint_ids_long] = command
            self._robot.set_joint_position_target(self._target)
            for _ in range(self._timing.control_divisor):
                self._scene.write_data_to_sim()
                self._simulation.step(render=False)
                self._scene.update(self._timing.physics_dt)
            simulated_position = self._to_torch(
                self._robot.data.joint_pos,
                self._wp,
            )[:, self._joint_ids_long]
            position_delta = simulated_position - self._measured_position[row]
            position_error += (position_delta.double() ** 2).sum(dim=1)
            if self._measured_velocity is not None:
                simulated_velocity = self._to_torch(
                    self._robot.data.joint_vel,
                    self._wp,
                )[:, self._joint_ids_long]
                velocity_delta = simulated_velocity - self._measured_velocity[row]
                velocity_error += (velocity_delta.double() ** 2).sum(dim=1)

        samples = int(self._commands.shape[0]) * len(self._joint_names)
        position_mse = (position_error / samples).detach().cpu().numpy()
        velocity_mse = (velocity_error / samples).detach().cpu().numpy()
        return PopulationEvaluation(position_mse, velocity_mse)

    def close(self) -> None:
        """Close the launched Isaac application exactly once."""

        application = self._application
        self._application = None
        if application is not None:
            application.close()


def _default_adapter_factory(resolved: ResolvedSysIdRun) -> SysIdRuntimeAdapter:
    """Construct the production Isaac Lab/Newton adapter lazily."""

    return IsaacNewtonRuntimeAdapter(resolved)


__all__ = [
    "IsaacNewtonRuntimeAdapter",
    "OptimizerFactory",
    "PopulationEvaluation",
    "RuntimeAdapterFactory",
    "SysIdRuntimeAdapter",
    "optimize_resolved_sysid",
]
