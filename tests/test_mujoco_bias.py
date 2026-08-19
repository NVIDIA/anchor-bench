# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from anchor_bench.train.utils import mujoco_bias
from anchor_bench.train.utils.mujoco_bias import MuJoCoBiasEvaluator


class FakeArray:
    def __init__(self, values: list[float]) -> None:
        self.values = np.asarray(values)

    def numpy(self) -> np.ndarray:
        return self.values.copy()


class FakeModel:
    def __init__(self, dof_ref: list[float] | None) -> None:
        self.joint_label = ["joint_a", "joint_b"]
        self.joint_q_start = FakeArray([0.0, 1.0, 2.0])
        self.joint_qd_start = FakeArray([0.0, 1.0, 2.0])
        self.joint_dof_count = 2
        self.mujoco = (
            None if dof_ref is None else SimpleNamespace(dof_ref=FakeArray(dof_ref))
        )


class FakeSolver:
    def __init__(self, model: FakeModel, **options: object) -> None:
        assert options == {"use_mujoco_cpu": True, "disable_contacts": True}
        self.mj_model = object()
        self.mj_data = SimpleNamespace(
            qpos=np.asarray([10.0, 20.0, 30.0]),
            qvel=np.asarray([0.0, 0.0]),
            qfrc_bias=np.asarray([0.0, 0.0]),
        )
        self.mj_q_start = FakeArray([0.0, 2.0])
        self.mj_qd_start = FakeArray([0.0, 1.0])


@pytest.fixture
def fake_bias_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    def forward(model: object, data: SimpleNamespace) -> None:
        data.qfrc_bias[:] = [
            data.qpos[0] + 2.0 * data.qvel[0],
            data.qpos[2] + 2.0 * data.qvel[1],
        ]

    monkeypatch.setattr(mujoco_bias.newton.solvers, "SolverMuJoCo", FakeSolver)
    monkeypatch.setattr(mujoco_bias.mujoco, "mj_forward", forward)


@pytest.mark.parametrize(
    ("dof_ref", "expected"),
    [
        ([0.5, -0.25], [[6.75, 10.5], [2.75, 4.5]]),
        (None, [[7.0, 10.0], [3.0, 4.0]]),
    ],
)
def test_evaluator_generates_bias_rows_with_dof_reference_fallback(
    fake_bias_runtime: None,
    dof_ref: list[float] | None,
    expected: list[list[float]],
) -> None:
    evaluator = MuJoCoBiasEvaluator(
        FakeModel(dof_ref),
        ("joint_b", "joint_a"),
    )

    values = evaluator.evaluate(
        np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        np.asarray([[3.0, 4.0], [0.0, 0.0]]),
    )

    np.testing.assert_allclose(values, expected)


def test_evaluator_generates_named_single_state_bias(
    fake_bias_runtime: None,
) -> None:
    evaluator = MuJoCoBiasEvaluator(
        FakeModel([0.5, -0.25]),
        ("joint_b", "joint_a"),
    )

    values = evaluator.evaluate_state(
        {"joint_a": 2.0, "joint_b": 1.0},
        {"joint_a": 4.0, "joint_b": 3.0},
    )

    assert values == {"joint_b": pytest.approx(6.75), "joint_a": pytest.approx(10.5)}


def test_evaluator_uses_complete_state_for_selected_coupled_output(
    fake_bias_runtime: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = MuJoCoBiasEvaluator(
        FakeModel(None),
        ("joint_a", "joint_b"),
        output_entities=("joint_a",),
    )

    def forward(model: object, data: SimpleNamespace) -> None:
        data.qfrc_bias[0] = data.qpos[0] + data.qpos[2] + data.qvel[1]

    monkeypatch.setattr(mujoco_bias.mujoco, "mj_forward", forward)

    values = evaluator.evaluate_state(
        {"joint_a": 2.0, "joint_b": 3.0},
        {"joint_a": 4.0, "joint_b": 5.0},
    )

    assert values == {"joint_a": pytest.approx(10.0)}


def test_evaluator_rejects_non_scalar_state_joint(
    fake_bias_runtime: None,
) -> None:
    model = FakeModel(None)
    model.joint_q_start = FakeArray([0.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="scalar joints: joint_a"):
        MuJoCoBiasEvaluator(model, ("joint_a",))


def test_evaluator_rejects_missing_mujoco_coordinates(
    fake_bias_runtime: None,
) -> None:
    class MissingCoordinateSolver(FakeSolver):
        def __init__(self, model: FakeModel, **options: object) -> None:
            super().__init__(model, **options)
            self.mj_q_start = FakeArray([0.0, -1.0])

    with pytest.raises(ValueError, match="joint_b"):
        MuJoCoBiasEvaluator(
            FakeModel(None),
            ("joint_b",),
            solver_type=MissingCoordinateSolver,
        )


def test_evaluator_ignores_unmapped_non_output_state_joint(
    fake_bias_runtime: None,
) -> None:
    class PartiallyMappedSolver(FakeSolver):
        def __init__(self, model: FakeModel, **options: object) -> None:
            super().__init__(model, **options)
            self.mj_q_start = FakeArray([0.0, -1.0])
            self.mj_qd_start = FakeArray([0.0, -1.0])

    evaluator = MuJoCoBiasEvaluator(
        FakeModel(None),
        ("joint_a", "joint_b"),
        output_entities=("joint_a",),
        solver_type=PartiallyMappedSolver,
    )

    values = evaluator.evaluate_state(
        {"joint_a": 2.0, "joint_b": 100.0},
        {"joint_a": 4.0, "joint_b": 100.0},
    )

    assert values == {"joint_a": pytest.approx(10.0)}
