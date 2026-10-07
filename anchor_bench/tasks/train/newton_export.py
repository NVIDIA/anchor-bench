# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export trained per-joint GRUs for Newton's DriveNeuralGRU."""

from __future__ import annotations

import json
import keyword
from pathlib import Path
from typing import Sequence

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from torch import nn

from anchor_bench.tasks.train.models import GruModel
from anchor_bench.tasks.train.normalization import (
    FeatureNormalization,
    NormalizationStatistics,
)
from anchor_bench.tasks.train.task import MappedTrainingSequence

FEATURE_NAMES = {
    "actual_q": "position",
    "command_q": "target_position",
    "position_error": "position_error",
    "actual_dq": "velocity",
    "command_dq": "target_velocity",
    "velocity_error": "velocity_error",
}
BUILTIN_FEATURES = set(FEATURE_NAMES.values())


class UnsupportedNewtonGRUModel(ValueError):
    """Indicate a valid training model outside Newton's GRU drive contract."""


def export_newton_gru(
    model: GruModel,
    normalization: NormalizationStatistics,
    sequences: Sequence[MappedTrainingSequence],
    path: Path,
) -> Path:
    """Embed weights, training statistics, and timestep in a portable ONNX file.

    Newton evaluates one scalar output per joint, sharing weights between joint
    lanes. Every mapping must contain the same ordered features of its output
    joint. Unsupported architectures fail instead of exporting a partial model.
    """
    if not sequences:
        raise ValueError("Newton GRU export requires mapped training sequences.")
    features: list[str] | None = None
    intervals: set[int] = set()
    target_features: set[str] = set()
    for sequence in sequences:
        if sequence.targets.shape[1] != 1:
            raise UnsupportedNewtonGRUModel(
                "Newton GRU requires one output joint per mapping."
            )
        joint, target_feature = str(sequence.targets.columns[0]).rsplit("/", 1)
        target_features.add(target_feature)
        current = []
        for column in sequence.inputs.columns:
            entity, feature = str(column).rsplit("/", 1)
            if entity != joint:
                raise UnsupportedNewtonGRUModel(
                    "Newton GRU inputs must belong to the output joint."
                )
            current.append(FEATURE_NAMES.get(feature, feature))
        if features is not None and features != current:
            raise UnsupportedNewtonGRUModel(
                "Newton GRU mappings must share ordered input features."
            )
        features = current
        timestamps = sequence.inputs.index.get_level_values("time_ns").to_numpy(
            dtype=np.int64
        )
        intervals.update(int(value) for value in np.diff(timestamps))
    if len(target_features) != 1:
        raise UnsupportedNewtonGRUModel(
            "Newton GRU mappings must share one output feature."
        )
    if len(intervals) != 1 or next(iter(intervals)) <= 0:
        raise ValueError(
            "Newton GRU export requires a uniform positive sample timestep."
        )
    if not features or len(set(features)) != len(features):
        raise UnsupportedNewtonGRUModel(
            "Newton GRU input features must be non-empty and unique."
        )
    custom = [feature for feature in features if feature not in BUILTIN_FEATURES]
    if len(custom) > 1 or any(
        not name.isidentifier() or keyword.iskeyword(name) for name in custom
    ):
        raise UnsupportedNewtonGRUModel(
            "Newton GRU supports at most one custom input with a valid identifier."
        )
    if model.output.out_features != 1 or model.recurrent.input_size != len(features):
        raise ValueError("Newton GRU model dimensions do not match per-joint mappings.")
    if not isinstance(model.activation, (nn.Identity, nn.Tanh)):
        raise UnsupportedNewtonGRUModel(
            "Newton GRU supports only identity or tanh output activation."
        )
    if (
        len(normalization.inputs.mean) != len(features)
        or len(normalization.inputs.standard_deviation) != len(features)
        or len(normalization.outputs.mean) != 1
        or len(normalization.outputs.standard_deviation) != 1
    ):
        raise ValueError("Newton GRU normalization dimensions do not match mappings.")

    def statistics(
        names: list[str], values: FeatureNormalization
    ) -> dict[str, dict[str, float]]:
        mean = dict(zip(names, values.mean))
        std = dict(zip(names, values.standard_deviation))
        if not all(np.isfinite(value) for value in mean.values()) or not all(
            np.isfinite(value) and value > 0 for value in std.values()
        ):
            raise ValueError(
                "Newton GRU normalization must be finite with positive standard deviations."
            )
        return {"mean": mean, "std": std}

    metadata = {
        "model_type": "gru",
        "input_columns": features,
        "custom_inputs": custom,
        "sample_dt_s": next(iter(intervals)) / 1_000_000_000,
        "normalization": {
            "inputs": statistics(features, normalization.inputs),
            "targets": statistics(["torque"], normalization.outputs),
        },
        "output_feature": str(sequences[0].targets.columns[0]).rsplit("/", 1)[1],
    }
    hidden = model.recurrent.hidden_size
    weights = model.state_dict()
    nodes = []
    initializers = []
    inputs = [
        helper.make_tensor_value_info(
            "inputs", TensorProto.FLOAT, [1, "batch", len(features)]
        )
    ]
    outputs = []

    def initializer(name: str, value: np.ndarray) -> str:
        initializers.append(numpy_helper.from_array(value, name))
        return name

    def gates(name: str) -> np.ndarray:
        array = weights[name].detach().cpu().numpy().astype(np.float32)
        # PyTorch uses (reset, update, new); ONNX uses (update, reset, new).
        return np.concatenate(
            (array[hidden : 2 * hidden], array[:hidden], array[2 * hidden :])
        )[None]

    source = "inputs"
    for layer in range(model.recurrent.num_layers):
        prefix = f"layer_{layer}"
        state_in, state_out = f"hidden_{layer}", f"next_hidden_{layer}"
        inputs.append(
            helper.make_tensor_value_info(
                state_in, TensorProto.FLOAT, [1, "batch", hidden]
            )
        )
        outputs.append(
            helper.make_tensor_value_info(
                state_out, TensorProto.FLOAT, [1, "batch", hidden]
            )
        )
        w = initializer(f"{prefix}_W", gates(f"recurrent.weight_ih_l{layer}"))
        r = initializer(f"{prefix}_R", gates(f"recurrent.weight_hh_l{layer}"))
        b = initializer(
            f"{prefix}_B",
            np.concatenate(
                (
                    gates(f"recurrent.bias_ih_l{layer}"),
                    gates(f"recurrent.bias_hh_l{layer}"),
                ),
                axis=1,
            ),
        )
        nodes.append(
            helper.make_node(
                "GRU",
                [source, w, r, b, "", state_in],
                [f"{prefix}_Y", state_out],
                hidden_size=hidden,
                linear_before_reset=1,
            )
        )
        shape = initializer(
            f"{prefix}_shape", np.array([1, -1, hidden], dtype=np.int64)
        )
        source = f"{prefix}_sequence"
        nodes.append(helper.make_node("Reshape", [f"{prefix}_Y", shape], [source]))
    shape = initializer("head_shape", np.array([-1, hidden], dtype=np.int64))
    nodes.append(helper.make_node("Reshape", [source, shape], ["head_input"]))
    w = initializer(
        "head_weight",
        weights["output.weight"].detach().cpu().numpy().astype(np.float32),
    )
    b = initializer(
        "head_bias", weights["output.bias"].detach().cpu().numpy().astype(np.float32)
    )
    nodes.append(
        helper.make_node("Gemm", ["head_input", w, b], ["head_output"], transB=1)
    )
    source = "head_output"
    if isinstance(model.activation, nn.Tanh):
        nodes.append(helper.make_node("Tanh", [source], ["activated"]))
        source = "activated"
    scale = initializer(
        "output_scale",
        weights["output_scale"].detach().cpu().numpy().astype(np.float32),
    )
    nodes.append(helper.make_node("Mul", [source, scale], ["effort"]))
    outputs.insert(
        0, helper.make_tensor_value_info("effort", TensorProto.FLOAT, ["batch", 1])
    )
    graph = helper.make_graph(nodes, "joint_gru", inputs, outputs, initializers)
    artifact = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    helper.set_model_props(
        artifact, {"metadata": json.dumps(metadata, allow_nan=False)}
    )
    onnx.checker.check_model(artifact)
    onnx.save(artifact, path)
    return path
