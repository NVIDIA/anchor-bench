# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Physical characterization of the native MuJoCo sysID runtime."""

from __future__ import annotations

from pathlib import Path

import newton
import numpy as np
import pytest
import warp as wp

from anchor_bench.core.simulation import SimulationContext
from anchor_bench.core.simulation.newton_stepper import NewtonStepExecutor
from anchor_bench.core.simulation.registry import create_simulation_context
from anchor_bench.schema.actuator_sys_id import ActuatorSysIdConfig
from anchor_bench.tasks.actuator_sys_id.model import Candidate, Trajectory
from anchor_bench.tasks.actuator_sys_id.parameter_space import create_parameter_space
from anchor_bench.tasks.actuator_sys_id.simulation import NewtonPopulationEvaluator

pytestmark = pytest.mark.integration


@pytest.fixture
def scalar_asset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Author a unit-mass horizontal slider, with no contacts or downloads."""

    monkeypatch.setattr(SimulationContext, "_instance", None)
    asset = tmp_path / "slider.usda"
    asset.write_text(
        """#usda 1.0
(
    defaultPrim = "Slider"
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "Slider" (
    prepend apiSchemas = ["PhysicsArticulationRootAPI"]
) {
    def Xform "base" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    ) {
        float physics:mass = 1
        float3 physics:diagonalInertia = (0.1, 0.1, 0.1)
    }
    def Xform "carriage" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    ) {
        float physics:mass = 1
        float3 physics:diagonalInertia = (0.1, 0.1, 0.1)
    }
    def PhysicsFixedJoint "root" {
        rel physics:body1 = </Slider/base>
    }
    def PhysicsPrismaticJoint "slide" {
        rel physics:body0 = </Slider/base>
        rel physics:body1 = </Slider/carriage>
        uniform token physics:axis = "X"
        float physics:lowerLimit = -10
        float physics:upperLimit = 10
    }
}
""",
        encoding="utf-8",
    )
    return asset


def _evaluator(config: ActuatorSysIdConfig) -> NewtonPopulationEvaluator:
    context = create_simulation_context(
        config.simulation.type, config.simulation.usd_path, fixed_base=True
    )
    evaluator = context.create_sysid_evaluator(config, create_parameter_space(config))
    assert isinstance(evaluator, NewtonPopulationEvaluator)
    return evaluator


def _config(
    asset: Path, *, delay_ms: int = 0, buffer_time: float = 0.0
) -> ActuatorSysIdConfig:
    return ActuatorSysIdConfig.model_validate(
        {
            "type": "actuator_sys_id",
            "joint_parameters": [
                {
                    "names": ["slide"],
                    "controller": {
                        "type": "pd",
                        "delay_duration_ms": delay_ms,
                        "clamps": {"effort_limit": 1000.0, "velocity_limit": 100.0},
                        "stiffness": {"lower": 1.0, "upper": 100.0, "initial": 40.0},
                        "damping": {"lower": 0.0, "upper": 10.0, "initial": 2.0},
                    },
                    "parameters": {
                        "armature": {"lower": 0.0, "upper": 1.0, "initial": 0.1},
                        "dry_friction": {"lower": 0.0, "upper": 1.0, "initial": 0.0},
                        "viscous_friction": {
                            "lower": 0.0,
                            "upper": 2.0,
                            "initial": 0.0,
                        },
                    },
                }
            ],
            "optimizer": {
                "type": "python-cmaes",
                "sigma": 0.2,
                "max_iterations": 1,
                "epsilon": 0.001,
            },
            "simulation": {
                "type": "newton",
                "usd_path": str(asset),
                "physics_freq": 200,
                "control_freq": 100,
                "render_freq": 50,
                "solver": {
                    "type": "warp_mujoco",
                    "solver_iterations": 10,
                    "line_search_iterations": 4,
                },
            },
            "num_envs": 2,
            "buffer_time": buffer_time,
            "output_dir": str(asset.parent / "output"),
            "dataset": {
                "type": "parquet",
                "source": "unused.parquet",
                "resampling": {"type": "uniform", "dt": 0.01},
            },
        }
    )


def _candidate(*, stiffness: float = 40.0, armature: float = 0.1) -> Candidate:
    return {
        "slide": {
            "stiffness": stiffness,
            "damping": 2.0,
            "armature": armature,
            "dry_friction": 0.0,
            "viscous_friction": 0.0,
        }
    }


def _trajectory(command: float, initial: float, *, samples: int = 25) -> Trajectory:
    return Trajectory(
        experiment=f"command={command},initial={initial}",
        joint_names=["slide"],
        commanded_position=np.full((samples, 1), command),
        measured_position=np.full((samples, 1), initial),
    )


@pytest.mark.parametrize(
    "alternative",
    [_candidate(stiffness=80.0), _candidate(armature=0.8)],
    ids=["gain", "armature"],
)
def test_native_mujoco_candidate_updates_change_each_world_response(
    scalar_asset: Path, alternative: Candidate
) -> None:
    # Dropping notify_model_changed, broadcasting world zero, or caching the
    # first population must change these physical scores and final positions.
    config = _config(scalar_asset)
    with wp.ScopedDevice("cpu"):
        evaluator = _evaluator(config)
        episode = _trajectory(0.6, 0.0)
        # CPU Warp arrays expose borrowed NumPy storage; retain a snapshot
        # before the next evaluation reuses the score buffer.
        first = evaluator.evaluate([_candidate(), alternative], [episode]).copy()
        first_q = evaluator._simulation_context.state.joint_q
        assert first_q is not None
        positions = first_q.numpy().copy()
        swapped = evaluator.evaluate([alternative, _candidate()], [episode]).copy()
        swapped_q = evaluator._simulation_context.state.joint_q
        assert swapped_q is not None

    assert np.isfinite(first).all() and np.isfinite(swapped).all()
    assert abs(first[0] - first[1]) > 1.0e-4
    assert abs(positions[0] - positions[1]) > 1.0e-3
    np.testing.assert_allclose(swapped, first[::-1], rtol=1.0e-5, atol=1.0e-8)
    np.testing.assert_allclose(
        swapped_q.numpy(), positions[::-1], rtol=1.0e-5, atol=1.0e-7
    )


def test_native_mujoco_repeated_trajectory_reset_is_deterministic(
    scalar_asset: Path,
) -> None:
    # Resetting after restoring the measured pose, retaining velocity, or
    # leaking solver state across episodes breaks A/B/A repeatability.
    config = _config(scalar_asset, delay_ms=20, buffer_time=0.02)
    with wp.ScopedDevice("cpu"):
        evaluator = _evaluator(config)
        population = [_candidate(), _candidate(stiffness=80.0, armature=0.4)]
        a = _trajectory(0.6, 0.23)
        b = _trajectory(-0.7, -0.47, samples=13)
        first = evaluator.evaluate(population, [a]).copy()
        first_q = evaluator._simulation_context.state.joint_q
        assert first_q is not None
        positions = first_q.numpy().copy()
        middle = evaluator.evaluate(population, [b]).copy()
        repeated = evaluator.evaluate(population, [a]).copy()
        repeated_q = evaluator._simulation_context.state.joint_q
        assert repeated_q is not None

    assert np.isfinite(middle).all()
    assert not np.allclose(middle, first)
    np.testing.assert_allclose(repeated, first, rtol=1.0e-6, atol=1.0e-9)
    np.testing.assert_allclose(repeated_q.numpy(), positions, rtol=1.0e-6, atol=1.0e-7)


def _native_runtime(
    asset: Path, device: str
) -> tuple[
    newton.Model, newton.solvers.SolverMuJoCo, newton.Control, NewtonStepExecutor
]:
    """Build a separate native runtime for explicit command scheduling."""

    with wp.ScopedDevice(device):
        template = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(template)
        template.add_usd(str(asset), floating=False)
        index = next(
            i
            for i, label in enumerate(template.joint_label)
            if label.endswith("/slide")
        )
        dof = template.joint_qd_start[index]
        template.joint_target_mode[dof] = int(newton.JointTargetMode.POSITION)
        template.joint_target_ke[dof] = 40.0
        template.joint_target_kd[dof] = 2.0
        template.joint_armature[dof] = 0.1
        template.joint_effort_limit[dof] = 1000.0
        template.joint_velocity_limit[dof] = 100.0
        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.replicate(template, copy_count=2)
        model = builder.finalize()
        solver = newton.solvers.SolverMuJoCo(
            model,
            solver="newton",
            integrator="implicitfast",
            iterations=10,
            ls_iterations=4,
        )
        control = model.control()
        executor = NewtonStepExecutor(
            model=model,
            solver=solver,
            state=model.state(),
            state_next=model.state(),
            control=control,
            contacts=None,
        )
    return model, solver, control, executor


def _reset_native(
    model: newton.Model,
    solver: newton.solvers.SolverMuJoCo,
    executor: NewtonStepExecutor,
    *,
    q: float,
    qd: float = 0.0,
) -> None:
    solver.reset(executor.state)
    for state in (executor.state, executor.state_next):
        assert state.joint_q is not None and state.joint_qd is not None
        state.joint_q.assign(np.full(2, q, dtype=np.float32))
        state.joint_qd.assign(np.full(2, qd, dtype=np.float32))
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)


def test_native_mujoco_delay_and_buffer_match_explicit_physical_rollout(
    scalar_asset: Path,
) -> None:
    # An off-by-one delay, scoring the warmup prefix, or scoring the final
    # unused command fails against this literal six-transition schedule.
    config = _config(scalar_asset, delay_ms=20, buffer_time=0.02)
    episode = Trajectory(
        experiment="two-step-delay",
        joint_names=["slide"],
        commanded_position=np.array(
            [[0.3], [-0.2], [0.5], [0.1], [-0.4], [0.2], [9.0]]
        ),
        measured_position=np.array(
            [[0.15], [19.0], [-13.0], [0.3], [-0.15], [0.05], [-0.2]]
        ),
    )
    with wp.ScopedDevice("cpu"):
        evaluator = _evaluator(config)
        scores = evaluator.evaluate([_candidate(), _candidate()], [episode])
        model, solver, control, executor = _native_runtime(scalar_asset, "cpu")
        _reset_native(model, solver, executor, q=0.15)
        assert control.joint_target_q is not None
        positions = []
        for target in [0.3, 0.3, 0.3, -0.2, 0.5, 0.1]:
            control.joint_target_q.assign(np.full(2, target, dtype=np.float32))
            executor.step(dt=0.01, substeps=2)
            assert executor.state.joint_q is not None
            positions.append(executor.state.joint_q.numpy().copy())

    # The first two transitions are deliberately huge measurement outliers.
    expected = np.mean(
        np.square(
            np.asarray(positions)[2:] - np.array([0.3, -0.15, 0.05, -0.2])[:, None]
        ),
        axis=0,
    )
    assert np.isfinite(expected).all() and np.all(expected > 0.01)
    np.testing.assert_allclose(scores, expected, rtol=1.0e-5, atol=1.0e-8)


@pytest.mark.parametrize("substeps", [3, 4], ids=["odd", "even"])
def test_native_mujoco_cuda_graph_matches_eager_with_live_candidates(
    scalar_asset: Path, substeps: int
) -> None:
    # Frozen capture-time properties/targets, swapped state identities, or an
    # omitted odd-step copy fail against an independent eager native solver.
    devices = wp.get_cuda_devices()
    if not devices:
        pytest.skip(
            "Warp detected no CUDA device; native CUDA graph coverage requires CUDA"
        )
    device = devices[0].alias
    eager = _native_runtime(scalar_asset, device)
    captured = _native_runtime(scalar_asset, device)
    for model, solver, control, executor in (eager, captured):
        assert model.device.is_cuda and not solver.use_mujoco_cpu
        assert control.joint_target_q is not None
        control.joint_target_q.assign(np.array([0.6, -0.2], dtype=np.float32))
        # Compile and allocate outside capture, then restore physical state.
        for _ in range(2):
            executor.step(dt=0.01, substeps=substeps)
        _reset_native(model, solver, executor, q=0.15, qd=0.2)

    graph_executor = captured[3]
    canonical = graph_executor.state
    graph_executor.capture_graph(dt=0.01, substeps=substeps)
    assert graph_executor._graph is not None
    previous = None
    for gains, armatures, targets in [
        ([40.0, 40.0], [0.1, 0.1], [0.6, -0.2]),
        ([80.0, 10.0], [0.1, 0.1], [0.6, -0.2]),
        ([80.0, 10.0], [0.8, 1.0], [0.6, -0.2]),
        ([80.0, 10.0], [0.8, 1.0], [-0.5, 0.4]),
    ]:
        results = []
        for model, solver, control, executor in (eager, captured):
            assert model.joint_target_ke is not None
            assert model.joint_armature is not None
            assert control.joint_target_q is not None
            model.joint_target_ke.assign(np.asarray(gains, dtype=np.float32))
            model.joint_armature.assign(np.asarray(armatures, dtype=np.float32))
            control.joint_target_q.assign(np.asarray(targets, dtype=np.float32))
            solver.notify_model_changed(
                newton.ModelFlags.JOINT_DOF_PROPERTIES
                | newton.ModelFlags.ACTUATOR_PROPERTIES
            )
            _reset_native(model, solver, executor, q=0.15, qd=0.2)
            states = []
            for _ in range(12):
                if executor is graph_executor:
                    executor.step(dt=0.01, substeps=substeps)
                else:
                    # Independent eager one-substep intervals expose an odd
                    # interval that forgets to restore the canonical buffer.
                    for _ in range(substeps):
                        executor.step(dt=0.01 / substeps, substeps=1)
                assert executor.state.joint_q is not None
                assert executor.state.joint_qd is not None
                states.append(
                    np.stack(
                        (
                            executor.state.joint_q.numpy(),
                            executor.state.joint_qd.numpy(),
                        )
                    )
                )
                if executor is graph_executor:
                    assert executor.state is canonical
            results.append(np.asarray(states))
        assert np.isfinite(results).all()
        np.testing.assert_allclose(results[1], results[0], rtol=2.0e-5, atol=2.0e-6)
        if previous is not None:
            assert np.all(np.abs(results[1][-1, 0] - previous) > 1.0e-4)
        previous = results[1][-1, 0].copy()
