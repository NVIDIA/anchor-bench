# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar

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
        self.joint_qd_start = FakeArray([0.0, 1.0, 2.0])
        self.joint_dof_count = 2
        self.mujoco = (
            None if dof_ref is None else SimpleNamespace(dof_ref=FakeArray(dof_ref))
        )


class FakeSolver:
    instances: ClassVar[list[FakeSolver]] = []

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
        self.instances.append(self)


@pytest.fixture
def fake_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeSolver.instances.clear()

    def forward(model: object, data: SimpleNamespace) -> None:
        data.qfrc_bias[:] = [
            data.qpos[0] + 2.0 * data.qvel[0],
            data.qpos[2] + 2.0 * data.qvel[1],
        ]

    modules = {
        "newton": SimpleNamespace(
            solvers=SimpleNamespace(SolverMuJoCo=FakeSolver),
        ),
        "mujoco": SimpleNamespace(mj_forward=forward),
    }
    monkeypatch.setattr(mujoco_bias, "import_module", modules.__getitem__)


@pytest.mark.parametrize(
    ("dof_ref", "expected"),
    [
        ([0.5, -0.25], [[6.75, 10.5], [2.75, 4.5]]),
        (None, [[7.0, 10.0], [3.0, 4.0]]),
    ],
)
def test_evaluator_generates_batch_bias_with_dof_reference_fallback(
    fake_modules: None,
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


def test_evaluator_generates_named_single_state_bias(fake_modules: None) -> None:
    evaluator = MuJoCoBiasEvaluator(
        FakeModel([0.5, -0.25]),
        ("joint_b", "joint_a"),
    )

    values = evaluator.evaluate_state(
        {"joint_a": 2.0, "joint_b": 1.0},
        {"joint_a": 4.0, "joint_b": 3.0},
    )

    assert values == {"joint_b": pytest.approx(6.75), "joint_a": pytest.approx(10.5)}


def test_evaluator_rejects_missing_mujoco_coordinates(fake_modules: None) -> None:
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
