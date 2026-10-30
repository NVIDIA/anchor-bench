# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Physical regressions for native training replay and bias features."""

from __future__ import annotations

from importlib import import_module
from types import SimpleNamespace
from typing import Any

import newton
import numpy as np
import pandas as pd
import pytest
import warp as wp
from newton.solvers import SolverMuJoCo

from anchor_bench.core.simulation import SimulationContext
from anchor_bench.core.simulation.newton_solvers import MuJoCoAdapter
from anchor_bench.tasks.train.data_processors.gru import DynamicBiasDataProcessor
from anchor_bench.tasks.train.validators.newton_replay_adapter import (
    _NewtonReplayAdapter,
)

pytestmark = pytest.mark.integration


class _CPUMuJoCo(SolverMuJoCo):
    """Select real CPU MuJoCo at replay's solver-construction boundary."""

    def __init__(self, model: newton.Model, **options: Any) -> None:
        options["use_mujoco_cpu"] = True
        super().__init__(model, **options)


def _context(*, device: str, coupled: bool = False) -> SimpleNamespace:
    """Build articulated scalar joints without imports or downloaded assets."""

    gravity = (0.0, -9.81, 0.0) if coupled else (0.0, 0.0, 0.0)
    builder = newton.ModelBuilder(gravity=gravity)
    SolverMuJoCo.register_custom_attributes(builder)
    first = builder.add_link(
        com=wp.vec3(0.3, 0.0, 0.0),
        mass=2.0,
        inertia=wp.mat33(0.02, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.02),
    )
    shoulder = builder.add_joint_revolute(
        parent=-1,
        child=first,
        axis=newton.Axis.Z,
        label="shoulder",
        target_ke=12.0,
        target_kd=0.8,
        armature=0.1,
        friction=0.0,
        damping=0.0,
        actuator_mode=newton.JointTargetMode.POSITION,
    )
    joints = [shoulder]
    if coupled:
        second = builder.add_link(
            com=wp.vec3(0.25, 0.0, 0.0),
            mass=1.0,
            inertia=wp.mat33(0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01),
        )
        joints.append(
            builder.add_joint_revolute(
                parent=first,
                child=second,
                parent_xform=wp.transform(wp.vec3(0.6, 0.0, 0.0), wp.quat_identity()),
                axis=newton.Axis.Z,
                label="elbow",
                armature=0.1,
            )
        )
    builder.add_articulation(joints)
    model = builder.finalize(device=device)
    if coupled:
        mujoco_attributes = getattr(model, "mujoco", None)
        assert mujoco_attributes is not None
        mujoco_attributes.dof_ref.assign(np.array([0.37, -0.28], dtype=np.float32))
    return SimpleNamespace(model=model, state=model.state())


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
@pytest.mark.parametrize("substeps", [1, 2])
@pytest.mark.parametrize("residual", [False, True], ids=["direct", "residual"])
def test_native_replay_direct_and_residual_reset(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    substeps: int,
    residual: bool,
) -> None:
    """Catch duplicated native PD, lost residual PD, and stale reset state."""

    if backend == "cuda" and not wp.is_cuda_available():
        pytest.skip("Native MuJoCo Warp replay requires an available CUDA device.")
    if backend == "cpu":
        monkeypatch.setattr(MuJoCoAdapter, "solver_class", _CPUMuJoCo)
    context = _context(device="cpu" if backend == "cpu" else "cuda:0")
    original_mode = context.model.joint_target_mode.numpy().copy()
    original_ke = context.model.joint_target_ke.numpy().copy()
    original_kd = context.model.joint_target_kd.numpy().copy()
    adapter = _NewtonReplayAdapter(
        context,
        output_joints=("shoulder",),
        residual=residual,
        substeps=substeps,
        disable_contacts=True,
    )

    def rollout(command: float, effort: float) -> np.ndarray:
        initial = adapter.reset(
            position={"shoulder": 0.25},
            velocity={"shoulder": -0.1},
            command={"shoulder": command},
        )
        assert initial.position["shoulder"] == pytest.approx(0.25)
        assert initial.velocity["shoulder"] == pytest.approx(-0.1)
        values = []
        for _ in range(20):
            state = adapter.step(
                command={"shoulder": command},
                effort={"shoulder": effort},
                dt=0.005,
            )
            values.append([state.position["shoulder"], state.velocity["shoulder"]])
        result = np.asarray(values)
        assert np.isfinite(result).all()
        return result

    try:
        forward = rollout(0.8, 0.3)
        backward = rollout(-0.4, 0.3)
        repeated = rollout(0.8, 0.3)
        np.testing.assert_allclose(repeated, forward, rtol=1e-6, atol=1e-7)
        if residual:
            assert forward[-1, 0] > backward[-1, 0] + 0.05
            np.testing.assert_array_equal(
                context.model.joint_target_mode.numpy(), original_mode
            )
        else:
            np.testing.assert_allclose(forward, backward, rtol=1e-6, atol=1e-7)
            unforced = rollout(0.8, 0.0)
            assert forward[-1, 1] > unforced[-1, 1] + 0.05
            np.testing.assert_array_equal(context.model.joint_target_ke.numpy(), [0.0])
            np.testing.assert_array_equal(context.model.joint_target_kd.numpy(), [0.0])
    finally:
        adapter.close()
    np.testing.assert_array_equal(
        context.model.joint_target_mode.numpy(), original_mode
    )
    np.testing.assert_array_equal(context.model.joint_target_ke.numpy(), original_ke)
    np.testing.assert_array_equal(context.model.joint_target_kd.numpy(), original_kd)


def test_offline_bias_matches_live_replay_bias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch missing dof_ref, velocity, joint mapping, or stale bias caching."""

    mujoco = import_module("mujoco")
    monkeypatch.setattr(MuJoCoAdapter, "solver_class", _CPUMuJoCo)
    context = _context(device="cpu", coupled=True)
    monkeypatch.setattr(SimulationContext, "current", classmethod(lambda cls: context))
    # Reverse dataset order relative to model order to expose mapping mistakes.
    dataset = pd.DataFrame(
        {
            "elbow/actual_q": [-0.45, 0.6, -0.45],
            "elbow/actual_dq": [1.3, -0.9, 1.3],
            "shoulder/actual_q": [0.4, -0.7, 0.4],
            "shoulder/actual_dq": [0.8, -1.1, 0.8],
        },
        index=pd.Index(["first", "different", "repeat"], name="sample"),
    )
    offline = DynamicBiasDataProcessor(dataset).process()
    assert offline.index.equals(dataset.index)
    assert offline.columns.tolist() == ["elbow/qfrc_bias", "shoulder/qfrc_bias"]
    reference_solver = SolverMuJoCo(
        context.model, use_mujoco_cpu=True, disable_contacts=True
    )
    reference_model = reference_solver.mj_model
    reference_data = mujoco.MjData(reference_model)
    names = ("shoulder", "elbow")
    references = dict(zip(names, [0.37, -0.28], strict=True))
    addresses = {}
    for name in names:
        joint_id = mujoco.mj_name2id(reference_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert joint_id >= 0
        addresses[name] = (
            int(reference_model.jnt_qposadr[joint_id]),
            int(reference_model.jnt_dofadr[joint_id]),
        )
    adapter = _NewtonReplayAdapter(
        context,
        output_joints=names,
        residual=True,
        substeps=1,
        disable_contacts=True,
    )
    try:
        for row_label, row in dataset.iterrows():
            positions = {name: float(row[f"{name}/actual_q"]) for name in names}
            velocities = {name: float(row[f"{name}/actual_dq"]) for name in names}
            state = adapter.reset(position=positions, velocity=velocities, command={})
            mujoco.mj_resetData(reference_model, reference_data)
            for name, (q_address, dof_address) in addresses.items():
                reference_data.qpos[q_address] = positions[name] + references[name]
                reference_data.qvel[dof_address] = velocities[name]
            mujoco.mj_forward(reference_model, reference_data)
            expected = reference_data.qfrc_bias.copy()
            assert np.isfinite(expected).all()
            for name, (_, dof_address) in addresses.items():
                assert offline.loc[row_label, f"{name}/qfrc_bias"] == pytest.approx(
                    expected[dof_address], rel=1e-6, abs=1e-6
                )
                assert adapter.dynamic_bias(name, state) == pytest.approx(
                    expected[dof_address], rel=1e-6, abs=1e-6
                )
            # This coupled fixture must exercise velocity-dependent bias as well.
            reference_data.qvel[:] = 0.0
            mujoco.mj_forward(reference_model, reference_data)
            assert np.max(np.abs(reference_data.qfrc_bias - expected)) > 0.01
            for name, (q_address, _) in addresses.items():
                reference_data.qpos[q_address] = positions[name]
                reference_data.qvel[addresses[name][1]] = velocities[name]
            mujoco.mj_forward(reference_model, reference_data)
            assert np.max(np.abs(reference_data.qfrc_bias - expected)) > 0.1
    finally:
        adapter.close()
