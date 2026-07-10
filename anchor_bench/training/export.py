from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


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


def _validate_statistic(value: Any, width: int, label: str, *, positive: bool) -> None:
    """Validate one scalar-or-vector normalization statistic."""

    if width == 1:
        values = [value] if not isinstance(value, list) else []
    else:
        values = value if isinstance(value, list) and len(value) == width else []
    if len(values) != width or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in values):
        raise ValueError(f"runtime_metadata normalization {label} must contain {width} numeric value(s)")
    if any(not math.isfinite(float(item)) or (positive and float(item) <= 0.0) for item in values):
        qualifier = "positive finite" if positive else "finite"
        raise ValueError(f"runtime_metadata normalization {label} must contain {qualifier} values")


def _validate_metadata_header(
    metadata: dict[str, Any], input_columns: list[str], target_columns: list[str]
) -> None:
    """Validate required metadata fields and checkpoint column identity."""

    required = {
        "delay",
        "input_columns",
        "input_feature_specs",
        "input_size",
        "joint_mappings",
        "model_type",
        "normalization",
        "output_size",
        "sample_dt_s",
        "schema_version",
        "target_columns",
        "training_topology",
    }
    missing = sorted(required - metadata.keys())
    if missing:
        raise ValueError("runtime_metadata is missing: " + ", ".join(missing))
    if metadata["schema_version"] != 3 or metadata["model_type"] != "gru":
        raise ValueError("runtime_metadata must describe a schema-version-3 GRU")
    if metadata["input_columns"] != input_columns or metadata["target_columns"] != target_columns:
        raise ValueError("checkpoint columns disagree with runtime_metadata")
    if len(set(input_columns)) != len(input_columns):
        raise ValueError("runtime_metadata input_columns must not contain duplicates")


def _metadata_mapping_widths(metadata: dict[str, Any]) -> tuple[int, int]:
    """Validate physical mappings and return one mapping axis size."""

    mappings = metadata["joint_mappings"]
    if not isinstance(mappings, list) or not mappings:
        raise ValueError("runtime_metadata.joint_mappings must be a non-empty list")
    first = mappings[0]
    if not isinstance(first, dict):
        raise ValueError("runtime_metadata joint mappings must be objects")
    input_width = len(first.get("input_joints") or ())
    output_width = len(first.get("output_joints") or ())
    if input_width < 1 or output_width < 1:
        raise ValueError("runtime_metadata joint mappings must have non-empty axes")
    for mapping in mappings:
        inputs = mapping.get("input_joints") if isinstance(mapping, dict) else None
        outputs = mapping.get("output_joints") if isinstance(mapping, dict) else None
        if (
            not isinstance(inputs, list)
            or not isinstance(outputs, list)
            or len(inputs) != input_width
            or len(outputs) != output_width
            or any(name not in inputs for name in outputs)
        ):
            raise ValueError("runtime_metadata contains an invalid or inconsistent joint mapping")
    return input_width, output_width


def _metadata_feature_widths(
    metadata: dict[str, Any],
    input_columns: list[str],
    input_width: int,
    output_width: int,
) -> tuple[dict[str, int], int]:
    """Validate ordered feature domains and return flattened widths."""

    specs = metadata["input_feature_specs"]
    if not isinstance(specs, list) or len(specs) != len(input_columns):
        raise ValueError("runtime_metadata.input_feature_specs must match input_columns")
    widths: dict[str, int] = {}
    for column, spec in zip(input_columns, specs, strict=True):
        if not isinstance(spec, dict) or spec.get("name") != column:
            raise ValueError("runtime_metadata feature specs must match input_columns in order")
        domain = spec.get("domain")
        channels = spec.get("channels")
        if domain not in {"input_joints", "output_joints"} or not isinstance(channels, list) or not channels:
            raise ValueError("runtime_metadata contains an invalid feature domain or channels")
        expected_domain = "output_joints" if column == "previous_torque" else "input_joints"
        if domain != expected_domain:
            raise ValueError(f"runtime_metadata feature {column} must use domain {expected_domain}")
        widths[column] = (input_width if domain == "input_joints" else output_width) * len(channels)
    return widths, sum(widths.values())


def _validate_metadata_normalization(
    metadata: dict[str, Any], feature_widths: dict[str, int], target: str, output_width: int
) -> None:
    """Validate every input and target normalization vector."""

    normalization = metadata["normalization"]
    if not isinstance(normalization, dict):
        raise ValueError("runtime_metadata normalization must be an object")
    inputs = normalization.get("inputs")
    targets = normalization.get("targets")
    if not isinstance(inputs, dict) or not isinstance(targets, dict):
        raise ValueError("runtime_metadata.normalization must contain inputs and targets")
    for section, widths in ((inputs, feature_widths), (targets, {target: output_width})):
        means = section.get("mean")
        stds = section.get("std")
        if (
            not isinstance(means, dict)
            or not isinstance(stds, dict)
            or any(column not in means or column not in stds for column in widths)
        ):
            raise ValueError("runtime_metadata normalization is missing feature statistics")
        for column, width in widths.items():
            _validate_statistic(means[column], width, f"mean.{column}", positive=False)
            _validate_statistic(stds[column], width, f"std.{column}", positive=True)


def _validate_metadata_time_and_delay(metadata: dict[str, Any]) -> None:
    """Validate the recorded timestep and learned-delay declaration."""

    try:
        sample_dt_s = float(metadata["sample_dt_s"])
    except (TypeError, ValueError) as exc:
        raise ValueError("runtime_metadata.sample_dt_s must be numeric") from exc
    if not math.isfinite(sample_dt_s) or sample_dt_s <= 0.0:
        raise ValueError("runtime_metadata.sample_dt_s must be positive and finite")
    delay = metadata["delay"]
    if not isinstance(delay, dict):
        raise ValueError("runtime_metadata delay must be an object")
    if delay.get("handling") != "learned" or delay.get("external_delay_s") != 0.0:
        raise ValueError("runtime_metadata delay must be learned with zero external delay")


def _validate_metadata_derivations(metadata: dict[str, Any], input_columns: list[str], target: str) -> None:
    """Validate metadata required by derived features and residual targets."""

    if "dynamic_bias" in input_columns and not isinstance(metadata.get("dynamic_bias_derivation"), dict):
        raise ValueError("runtime_metadata is missing dynamic_bias_derivation")
    if "previous_torque" in input_columns and not isinstance(metadata.get("previous_torque_derivation"), dict):
        raise ValueError("runtime_metadata is missing previous_torque_derivation")
    if (target == "torque_residual" or "solver_pd" in input_columns) and not isinstance(
        metadata.get("pd_baseline"), dict
    ):
        raise ValueError("runtime_metadata is missing pd_baseline")
    if target == "torque_residual" and not isinstance(metadata.get("friction_baseline"), dict):
        raise ValueError("runtime_metadata is missing friction_baseline")


def _validate_runtime_metadata(
    metadata: dict[str, Any],
    *,
    input_columns: list[str],
    target_columns: list[str],
    input_size: int,
    output_size: int,
) -> None:
    """Validate the axes and required fields of the embedded contract."""

    _validate_metadata_header(metadata, input_columns, target_columns)
    input_width, output_width = _metadata_mapping_widths(metadata)
    feature_widths, expected_input_size = _metadata_feature_widths(
        metadata, input_columns, input_width, output_width
    )
    if (input_size, output_size) != (expected_input_size, output_width):
        raise ValueError("checkpoint dimensions disagree with runtime_metadata axes")
    topology = "siso" if input_width == output_width == 1 else ("miso" if output_width == 1 else "mimo")
    if metadata["training_topology"] != topology:
        raise ValueError("runtime_metadata.training_topology disagrees with joint mappings")
    target = target_columns[0]
    _validate_metadata_normalization(metadata, feature_widths, target, output_width)
    _validate_metadata_time_and_delay(metadata)
    _validate_metadata_derivations(metadata, input_columns, target)


def export_gru_checkpoint(checkpoint_path: str | Path, output_path: str | Path | None = None) -> Path:
    """Script a GRU checkpoint and embed its runtime contract as metadata.json."""

    torch = _import_torch()
    source = Path(checkpoint_path)
    checkpoint: dict[str, Any] = torch.load(source, map_location="cpu")
    runtime_metadata = checkpoint.get("runtime_metadata")
    if not isinstance(runtime_metadata, dict):
        raise ValueError(f"{source} checkpoint is missing runtime_metadata")
    hparams = checkpoint.get("hyperparameters") or {}
    input_columns = checkpoint.get("input_columns") or []
    target_columns = checkpoint.get("target_columns") or []
    if not input_columns:
        raise ValueError(f"{source} checkpoint is missing input_columns")
    if len(target_columns) != 1:
        raise ValueError(f"{source} checkpoint must contain one target semantic")
    try:
        input_size = int(checkpoint["input_size"])
        output_size = int(checkpoint["output_size"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{source} checkpoint is missing model dimensions") from exc
    if input_size < 1 or output_size < 1:
        raise ValueError(f"{source} checkpoint dimensions must be positive")
    if runtime_metadata.get("input_size") != input_size or runtime_metadata.get("output_size") != output_size:
        raise ValueError(f"{source} checkpoint dimensions disagree with runtime_metadata")
    _validate_runtime_metadata(
        runtime_metadata,
        input_columns=input_columns,
        target_columns=target_columns,
        input_size=input_size,
        output_size=output_size,
    )
    state = checkpoint.get("model_state_dict") or {}
    if tuple(state["gru.weight_ih_l0"].shape)[1] != input_size or tuple(state["head.weight"].shape)[0] != output_size:
        raise ValueError(f"{source} checkpoint state dimensions are inconsistent")

    model = build_runtime_gru(
        input_size=input_size,
        output_size=output_size,
        hidden_size=int(hparams.get("hidden_size", 64)),
        num_layers=int(hparams.get("num_layers", 1)),
        dropout=float(hparams.get("dropout", 0.0)),
    )
    model.load_state_dict(state)
    model.eval()
    scripted = torch.jit.script(model)

    destination = (
        Path(output_path)
        if output_path is not None
        else source.with_name(source.name.replace("_best.pt", "_network.pt"))
    )
    if destination == source:
        destination = source.with_name(f"{source.stem}_network.pt")
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(
        scripted,
        str(destination),
        _extra_files={"metadata.json": json.dumps(runtime_metadata, sort_keys=True, allow_nan=False)},
    )
    return destination
