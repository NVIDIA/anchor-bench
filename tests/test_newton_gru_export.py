# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
import pandas as pd
import pytest
import torch
import warp as wp
from newton.actuators import Actuator, DriveNeuralGRU
from onnx.reference import ReferenceEvaluator

from anchor_bench.schema.training import GruModelConfig
from anchor_bench.tasks.train.models import GruModel
from anchor_bench.tasks.train.newton_export import export_newton_gru
from anchor_bench.tasks.train.normalization import fit_normalization
from anchor_bench.tasks.train.task import MappedTrainingSequence


def sequences(features=("actual_q", "position_error", "actual_dq", "previous_torque")):
    index = pd.MultiIndex.from_product(
        [["episode"], [0, 2_000_000, 4_000_000]], names=["experiment", "time_ns"]
    )
    return [
        MappedTrainingSequence(
            name=joint,
            experiment="episode",
            inputs=pd.DataFrame(
                {
                    f"{joint}/{feature}": np.array([1.0, 2.0, 4.0]) * (column + 1)
                    for column, feature in enumerate(features)
                },
                index=index,
            ),
            targets=pd.DataFrame(
                {f"{joint}/actual_effort": [3.0, 5.0, 9.0]}, index=index
            ),
        )
        for joint in ("a", "b")
    ]


@pytest.mark.parametrize("layers,activation", [(1, "identity"), (2, "tanh")])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_export_matches_pytorch_across_recurrent_steps(
    tmp_path: Path, layers: int, activation: str, device: str
):
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(17)
    config = GruModelConfig.model_validate(
        {
            "type": "gru",
            "dropout": 0.0,
            "hidden_size": 8,
            "num_layers": layers,
            "output": {"activation": activation, "scale": 2.5},
        }
    )
    model = GruModel(config, input_size=4, output_size=1).eval()
    mapped = sequences()
    stats = fit_normalization(mapped, inputs="standard", outputs="standard")
    path = export_newton_gru(model, stats, mapped, tmp_path / "model.onnx")
    onnx.checker.check_model(onnx.load(path))
    reference = ReferenceEvaluator(onnx.load(path))
    actuator = Actuator(
        indices=wp.array([1, 0], dtype=wp.uint32, device=device),
        drive=DriveNeuralGRU(str(path)),
    )
    current, following = actuator.state(), actuator.state()
    assert current is not None and following is not None
    state = SimpleNamespace(
        joint_q=wp.zeros(2, device=device), joint_qd=wp.zeros(2, device=device)
    )
    control = SimpleNamespace(
        joint_target_q=wp.zeros(2, device=device),
        joint_target_qd=wp.zeros(2, device=device),
        joint_f=wp.zeros(2, device=device),
        previous_torque=wp.zeros(2, device=device),
    )
    hidden = None
    for step in range(4):
        position = np.array([0.5, -0.4], dtype=np.float32) + step * 0.1
        target = np.array([1.2, 0.2], dtype=np.float32)
        velocity = np.array([-0.3, 0.4], dtype=np.float32)
        previous = np.array([step * 0.7, step * -0.2], dtype=np.float32)
        state.joint_q.assign(position)
        state.joint_qd.assign(velocity)
        control.joint_target_q.assign(target)
        control.previous_torque.assign(previous)
        control.joint_f.zero_()
        features = np.stack([position, target - position, velocity, previous], axis=1)[
            [1, 0]
        ][:, None, :]
        reference_inputs = {
            "inputs": stats.inputs.normalize_array(features[:, 0, :])[None].astype(
                np.float32
            )
        }
        for layer in range(layers):
            reference_inputs[f"hidden_{layer}"] = (
                np.zeros((1, 2, 8), dtype=np.float32)
                if hidden is None
                else hidden[layer : layer + 1].numpy()
            )
        reference_outputs = reference.run(None, reference_inputs)
        assert isinstance(reference_outputs, list)
        onnx_prediction = reference_outputs[0]
        with torch.inference_mode():
            expected = stats.wrap(model)(torch.tensor(features), hidden)
        np.testing.assert_allclose(
            onnx_prediction[:, 0] * stats.outputs.standard_deviation[0]
            + stats.outputs.mean[0],
            expected.predictions[:, 0, 0].numpy(),
            rtol=2e-5,
            atol=2e-5,
        )
        hidden = expected.state
        actuator.step(state, control, current, following, dt=0.002)
        np.testing.assert_allclose(
            control.joint_f.numpy()[[1, 0]],
            expected.predictions[:, 0, 0].numpy(),
            rtol=2e-5,
            atol=2e-5,
        )
        assert isinstance(following.drive_state, DriveNeuralGRU.State)
        assert following.drive_state.hidden is not None
        hidden_array = following.drive_state.hidden
        assert isinstance(hidden_array, wp.array)
        np.testing.assert_allclose(
            hidden_array.numpy(), hidden.numpy(), rtol=2e-5, atol=2e-5
        )
        current, following = following, current
    current.reset()
    following.reset()
    assert isinstance(current.drive_state, DriveNeuralGRU.State)
    assert current.drive_state.hidden is not None
    hidden_array = current.drive_state.hidden
    assert isinstance(hidden_array, wp.array)
    np.testing.assert_array_equal(hidden_array.numpy(), 0.0)
    with pytest.raises(ValueError, match="sample_dt_s"):
        actuator.step(state, control, current, following, dt=0.001)


def test_rejects_multiple_custom_inputs(tmp_path: Path):
    mapped = sequences(("solver_pd", "qfrc_bias"))
    model = GruModel(
        GruModelConfig.model_validate(
            {
                "type": "gru",
                "hidden_size": 8,
                "num_layers": 1,
                "dropout": 0.0,
                "output": {"activation": "identity", "scale": 1.0},
            }
        ),
        input_size=2,
        output_size=1,
    )
    with pytest.raises(ValueError, match="at most one custom input"):
        export_newton_gru(
            model,
            fit_normalization(mapped, inputs="none", outputs="none"),
            mapped,
            tmp_path / "invalid.onnx",
        )


def test_rejects_multi_joint_mapping(tmp_path: Path):
    mapped = sequences()
    mapped[0].targets["b/actual_effort"] = 0.0
    model = GruModel(
        GruModelConfig.model_validate(
            {
                "type": "gru",
                "hidden_size": 8,
                "num_layers": 1,
                "dropout": 0.0,
                "output": {"activation": "identity", "scale": 1.0},
            }
        ),
        input_size=4,
        output_size=2,
    )
    with pytest.raises(ValueError, match="one output joint"):
        export_newton_gru(
            model,
            fit_normalization(mapped[:1], inputs="none", outputs="none"),
            mapped[:1],
            tmp_path / "invalid.onnx",
        )
