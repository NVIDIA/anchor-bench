from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from anchor_bench.core.config_model import GRU_INPUT_COLUMNS

_SUPPORTED_FEATURE_DOMAINS = {"input_joints", "output_joints"}


def _import_torch():
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "PyTorch is required to export GRU actuator checkpoints."
        ) from exc
    return torch


def build_runtime_gru(
    *,
    input_size: int,
    output_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
):
    torch = _import_torch()

    class RuntimeGRU(torch.nn.Module):
        def __init__(self):
            super().__init__()
            recurrent_dropout = dropout if num_layers > 1 else 0.0
            self.gru = torch.nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=recurrent_dropout,
                batch_first=True,
            )
            self.head = torch.nn.Linear(hidden_size, output_size)

        def forward(self, x, hidden):
            out, hidden = self.gru(x, hidden)
            return self.head(out), hidden

    return RuntimeGRU()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if result <= 0 or result != value:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _name_list(value: Any, name: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"runtime_metadata.{name} must be a non-empty list")
    if any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"runtime_metadata.{name} entries must be non-empty strings")
    if len(set(value)) != len(value):
        raise ValueError(f"runtime_metadata.{name} must not contain duplicates")
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"runtime_metadata.{name} must be a mapping")
    return value


def _finite_values(value: Any, width: int, name: str, *, positive: bool = False) -> None:
    if width == 1 and not isinstance(value, list):
        values = [value]
    elif isinstance(value, list) and len(value) == width:
        values = value
    else:
        raise ValueError(f"runtime_metadata.{name} must contain {width} value(s)")
    for item in values:
        if isinstance(item, bool):
            raise ValueError(f"runtime_metadata.{name} must contain numeric values")
        try:
            number = float(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"runtime_metadata.{name} must contain numeric values") from exc
        if not math.isfinite(number) or (positive and number <= 0.0):
            qualifier = "finite positive" if positive else "finite"
            raise ValueError(f"runtime_metadata.{name} must contain {qualifier} values")


def _finite_number(value: Any, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"runtime_metadata.{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"runtime_metadata.{name} must be numeric") from exc
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        qualifier = "finite and nonnegative" if nonnegative else "finite"
        raise ValueError(f"runtime_metadata.{name} must be {qualifier}")
    return result


def _input_feature_widths(
    metadata: Mapping[str, Any],
    *,
    input_joints: Sequence[str],
    output_joints: Sequence[str],
    columns: Sequence[str],
) -> list[int]:
    specs = metadata.get("input_feature_specs")
    if not isinstance(specs, list) or len(specs) != len(columns):
        raise ValueError(
            "runtime_metadata.input_feature_specs must be an ordered list "
            "matching input_columns"
        )

    widths: list[int] = []
    axis_widths = {
        "input_joints": len(input_joints),
        "output_joints": len(output_joints),
    }
    required_keys = {"name", "domain", "channels"}
    for index, (spec, column) in enumerate(zip(specs, columns, strict=True)):
        label = f"runtime_metadata.input_feature_specs[{index}]"
        if not isinstance(spec, Mapping):
            raise ValueError(f"{label} must be a mapping")
        if set(spec) != required_keys:
            raise ValueError(
                f"{label} must contain exactly name, domain, and channels"
            )

        name = spec.get("name")
        if name != column:
            raise ValueError(
                "runtime_metadata.input_feature_specs names must match "
                "input_columns in order"
            )

        domain = spec.get("domain")
        if not isinstance(domain, str) or domain not in _SUPPORTED_FEATURE_DOMAINS:
            raise ValueError(
                f"{label}.domain must be one of "
                + ", ".join(sorted(_SUPPORTED_FEATURE_DOMAINS))
            )

        channels = spec.get("channels")
        if not isinstance(channels, list) or not channels:
            raise ValueError(f"{label}.channels must be a non-empty list")
        if any(not isinstance(channel, str) or not channel for channel in channels):
            raise ValueError(
                f"{label}.channels entries must be non-empty strings"
            )
        if len(set(channels)) != len(channels):
            raise ValueError(f"{label}.channels must not contain duplicates")
        if name == "previous_torque" and (
            domain != "output_joints" or channels != ["value"]
        ):
            raise ValueError(
                "runtime_metadata.input_feature_specs previous_torque must use "
                "domain='output_joints' and channels=['value']"
            )
        widths.append(axis_widths[domain] * len(channels))

    return widths


def _metadata_axes(
    metadata: Mapping[str, Any],
    *,
    input_size: int,
    output_size: int,
) -> tuple[list[str], list[str], list[str], list[int], str]:
    if metadata.get("schema_version") != 1 or metadata.get("model_type") != "gru":
        raise ValueError("runtime_metadata must describe a schema-version-1 GRU")
    input_joints = _name_list(metadata.get("joint_names"), "joint_names")
    output_joints = _name_list(metadata.get("output_joint_names"), "output_joint_names")
    if any(name not in input_joints for name in output_joints):
        raise ValueError("runtime_metadata.output_joint_names must be a subset of joint_names")
    columns = _name_list(metadata.get("input_columns"), "input_columns")
    unsupported = sorted(set(columns) - GRU_INPUT_COLUMNS)
    if unsupported:
        raise ValueError(
            "runtime_metadata contains unsupported input features: " + ", ".join(unsupported)
        )
    targets = _name_list(metadata.get("target_columns"), "target_columns")
    if len(targets) != 1 or targets[0] not in {"torque", "torque_residual"}:
        raise ValueError("runtime_metadata.target_columns must contain one torque semantic")
    input_widths = _input_feature_widths(
        metadata,
        input_joints=input_joints,
        output_joints=output_joints,
        columns=columns,
    )
    expected_input_size = sum(input_widths)
    if input_size != expected_input_size:
        raise ValueError(
            f"checkpoint input_size is {input_size}, metadata requires {expected_input_size}"
        )
    if output_size != len(output_joints):
        raise ValueError(
            f"checkpoint output_size is {output_size}, metadata requires {len(output_joints)}"
        )
    if _finite_number(metadata.get("sample_dt_s"), "sample_dt_s") <= 0.0:
        raise ValueError("runtime_metadata.sample_dt_s must be finite and positive")
    return input_joints, output_joints, columns, input_widths, targets[0]


def _validate_training_topology(
    metadata: Mapping[str, Any],
    *,
    input_joints: Sequence[str],
    output_joints: Sequence[str],
) -> None:
    expected = (
        "siso"
        if len(input_joints) == 1
        else "miso"
        if len(output_joints) == 1
        else "mimo"
    )
    if metadata.get("training_topology") != expected:
        raise ValueError(
            "runtime_metadata.training_topology must be "
            f"{expected!r} for {len(input_joints)} input joint(s) and "
            f"{len(output_joints)} output joint(s)"
        )


def _validate_normalization(
    metadata: Mapping[str, Any],
    *,
    output_joints: Sequence[str],
    columns: Sequence[str],
    input_widths: Sequence[int],
    target: str,
) -> None:
    normalization = _mapping(metadata.get("normalization"), "normalization")
    inputs = _mapping(normalization.get("inputs"), "normalization.inputs")
    input_mean = _mapping(inputs.get("mean"), "normalization.inputs.mean")
    input_std = _mapping(inputs.get("std"), "normalization.inputs.std")
    target_values = _mapping(normalization.get("targets"), "normalization.targets")
    target_mean = _mapping(target_values.get("mean"), "normalization.targets.mean")
    target_std = _mapping(target_values.get("std"), "normalization.targets.std")
    for column, width in zip(columns, input_widths, strict=True):
        _finite_values(input_mean.get(column), width, f"normalization.inputs.mean.{column}")
        _finite_values(
            input_std.get(column),
            width,
            f"normalization.inputs.std.{column}",
            positive=True,
        )
    _finite_values(
        target_mean.get(target),
        len(output_joints),
        f"normalization.targets.mean.{target}",
    )
    _finite_values(
        target_std.get(target),
        len(output_joints),
        f"normalization.targets.std.{target}",
        positive=True,
    )


def _validate_delay(metadata: Mapping[str, Any]) -> None:
    delay = _mapping(metadata.get("delay"), "delay")
    if delay.get("handling") != "learned":
        raise ValueError("runtime_metadata.delay.handling must be 'learned'")
    _finite_number(delay.get("training_delay_s", 0.0), "delay.training_delay_s", nonnegative=True)
    external_delay_s = _finite_number(
        delay.get("external_delay_s", 0.0),
        "delay.external_delay_s",
        nonnegative=True,
    )
    if external_delay_s != 0.0:
        raise ValueError("runtime_metadata.delay.external_delay_s must be zero")


def _validate_feature_derivations(
    metadata: Mapping[str, Any],
    *,
    columns: Sequence[str],
    input_joints: Sequence[str],
) -> None:
    if "dynamic_bias" in columns:
        derivation = _mapping(metadata.get("dynamic_bias_derivation"), "dynamic_bias_derivation")
        if not isinstance(derivation.get("source"), str) or not derivation["source"]:
            raise ValueError("runtime_metadata.dynamic_bias_derivation.source is required")
        requested = _name_list(
            derivation.get("requested_joint_names"),
            "dynamic_bias_derivation.requested_joint_names",
        )
        if set(requested) != set(input_joints):
            raise ValueError(
                "runtime_metadata.dynamic_bias_derivation.requested_joint_names "
                "must match joint_names"
            )
    if "previous_torque" in columns:
        derivation = _mapping(
            metadata.get("previous_torque_derivation"),
            "previous_torque_derivation",
        )
        expected = {
            "source": "previous_target",
            "runtime_source": "previous_raw_network_output_pre_clamp",
            "initialization": "physical_zero",
            "reset": "episode",
            "value_space": "physical_torque",
        }
        if any(derivation.get(key) != value for key, value in expected.items()):
            raise ValueError("runtime_metadata.previous_torque_derivation is inconsistent")


def _validate_baselines(
    metadata: Mapping[str, Any],
    *,
    columns: Sequence[str],
    target: str,
) -> None:
    if target == "torque_residual" or "solver_pd" in columns:
        pd_baseline = _mapping(metadata.get("pd_baseline"), "pd_baseline")
        _finite_number(pd_baseline.get("kp"), "pd_baseline.kp", nonnegative=True)
        _finite_number(pd_baseline.get("kd"), "pd_baseline.kd", nonnegative=True)
        if pd_baseline.get("velocity_target") != "zero":
            raise ValueError("runtime_metadata.pd_baseline.velocity_target must be 'zero'")
    if target == "torque_residual":
        friction = _mapping(metadata.get("friction_baseline"), "friction_baseline")
        _finite_number(friction.get("dry"), "friction_baseline.dry", nonnegative=True)
        _finite_number(
            friction.get("viscous"),
            "friction_baseline.viscous",
            nonnegative=True,
        )


def _validate_runtime_metadata(
    metadata: Mapping[str, Any],
    *,
    input_size: int,
    output_size: int,
) -> None:
    input_joints, output_joints, columns, input_widths, target = _metadata_axes(
        metadata,
        input_size=input_size,
        output_size=output_size,
    )

    _validate_training_topology(
        metadata,
        input_joints=input_joints,
        output_joints=output_joints,
    )

    _validate_normalization(
        metadata,
        output_joints=output_joints,
        columns=columns,
        input_widths=input_widths,
        target=target,
    )
    _validate_delay(metadata)

    _validate_feature_derivations(
        metadata,
        columns=columns,
        input_joints=input_joints,
    )

    _validate_baselines(metadata, columns=columns, target=target)


def export_gru_checkpoint(checkpoint_path: str | Path, output_path: str | Path | None = None) -> Path:
    torch = _import_torch()
    source = Path(checkpoint_path)
    checkpoint: dict[str, Any] = torch.load(source, map_location="cpu")
    runtime_metadata = checkpoint.get("runtime_metadata")
    hparams = checkpoint.get("hyperparameters") or {}
    if runtime_metadata is None:
        input_columns = checkpoint.get("input_columns") or []
        target_columns = checkpoint.get("target_columns") or []
        if not input_columns:
            raise ValueError(f"{source} checkpoint is missing input_columns")
        if not target_columns:
            raise ValueError(f"{source} checkpoint is missing target_columns")
        if "input_feature_specs" in checkpoint:
            raise ValueError(
                "checkpoint input_feature_specs requires non-null runtime_metadata"
            )
        input_size = _positive_int(
            checkpoint.get("input_size", len(input_columns)),
            "checkpoint input_size",
        )
        output_size = _positive_int(
            checkpoint.get("output_size", len(target_columns)),
            "checkpoint output_size",
        )
    else:
        input_size = _positive_int(
            checkpoint.get("input_size"),
            "checkpoint input_size",
        )
        output_size = _positive_int(
            checkpoint.get("output_size"),
            "checkpoint output_size",
        )
        metadata = _mapping(runtime_metadata, "runtime_metadata")
        _validate_runtime_metadata(
            metadata,
            input_size=input_size,
            output_size=output_size,
        )
    model = build_runtime_gru(
        input_size=input_size,
        output_size=output_size,
        hidden_size=int(hparams.get("hidden_size", 64)),
        num_layers=int(hparams.get("num_layers", 1)),
        dropout=float(hparams.get("dropout", 0.0)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    scripted = torch.jit.script(model)

    destination = Path(output_path) if output_path is not None else source.with_name(source.name.replace("_best.pt", "_network.pt"))
    if destination == source:
        destination = source.with_name(f"{source.stem}_network.pt")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if runtime_metadata is None:
        scripted.save(str(destination))
    else:
        torch.jit.save(
            scripted,
            str(destination),
            _extra_files={
                "metadata.json": json.dumps(runtime_metadata, sort_keys=True, allow_nan=False)
            },
        )
    return destination
