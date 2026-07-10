"""Strict schema-v3 TorchScript export for trained GRU checkpoints."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from anchor_bench.training.metadata import RuntimeGRUMetadata


def _import_torch():
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("PyTorch is required to export GRU actuator checkpoints.") from exc
    return torch


def build_runtime_gru(
    *,
    input_size: int,
    output_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
):
    """Construct the runtime-shaped GRU without importing Torch at module load."""

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


def serialize_runtime_metadata(metadata: RuntimeGRUMetadata | Mapping[str, Any]) -> str:
    """Return the deterministic JSON bytes embedded in a TorchScript archive."""

    validated = (
        metadata
        if isinstance(metadata, RuntimeGRUMetadata)
        else RuntimeGRUMetadata.model_validate(metadata)
    )
    return json.dumps(
        validated.model_dump(mode="json", exclude_none=True),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _validated_metadata(source: Path, checkpoint: Mapping[str, Any]) -> RuntimeGRUMetadata:
    raw = checkpoint.get("runtime_metadata")
    if raw is None:
        raise ValueError(f"{source} checkpoint is missing required schema-v3 runtime_metadata")
    try:
        metadata = (
            raw
            if isinstance(raw, RuntimeGRUMetadata)
            else RuntimeGRUMetadata.model_validate(raw)
        )
    except Exception as exc:
        raise ValueError(f"{source} checkpoint has invalid schema-v3 runtime_metadata") from exc

    input_size = _positive_int(checkpoint.get("input_size"), label="checkpoint input_size")
    output_size = _positive_int(checkpoint.get("output_size"), label="checkpoint output_size")
    if input_size != metadata.input_size:
        raise ValueError(
            f"checkpoint input_size is {input_size}, runtime_metadata requires {metadata.input_size}"
        )
    if output_size != metadata.output_size:
        raise ValueError(
            f"checkpoint output_size is {output_size}, runtime_metadata requires {metadata.output_size}"
        )

    hparams = checkpoint.get("hyperparameters")
    if hparams is not None:
        if not isinstance(hparams, Mapping):
            raise ValueError("checkpoint hyperparameters must be a mapping")
        expected: dict[str, int | float] = {
            "hidden_size": metadata.recurrent.hidden_size,
            "num_layers": metadata.recurrent.num_layers,
            "dropout": metadata.recurrent.dropout,
        }
        for name, expected_value in expected.items():
            if name not in hparams:
                continue
            value = hparams[name]
            if name == "dropout":
                try:
                    matches = math.isclose(float(value), float(expected_value), rel_tol=0.0, abs_tol=0.0)
                except (TypeError, ValueError):
                    matches = False
            else:
                matches = not isinstance(value, bool) and value == expected_value
            if not matches:
                raise ValueError(
                    f"checkpoint hyperparameters.{name} does not match runtime_metadata.recurrent.{name}"
                )
    return metadata


def _default_output_path(source: Path) -> Path:
    if source.name == "best.pt" and source.parent.name == "checkpoints":
        return source.parent.parent / "network.pt"
    return source.with_name("network.pt")


def export_gru_checkpoint(
    checkpoint_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Export only a complete schema-v3 checkpoint and embed its metadata."""

    torch = _import_torch()
    source = Path(checkpoint_path)
    loaded = torch.load(source, map_location="cpu")
    if not isinstance(loaded, Mapping):
        raise ValueError(f"{source} checkpoint must contain a mapping")
    checkpoint: Mapping[str, Any] = loaded
    metadata = _validated_metadata(source, checkpoint)
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError(f"{source} checkpoint is missing model_state_dict")

    model = build_runtime_gru(
        input_size=metadata.input_size,
        output_size=metadata.output_size,
        hidden_size=metadata.recurrent.hidden_size,
        num_layers=metadata.recurrent.num_layers,
        dropout=metadata.recurrent.dropout,
    )
    model.load_state_dict(state)
    model.eval()
    scripted = torch.jit.script(model)

    destination = Path(output_path) if output_path is not None else _default_output_path(source)
    if destination == source:
        raise ValueError("checkpoint and exported network paths must differ")
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(
        scripted,
        str(destination),
        _extra_files={"metadata.json": serialize_runtime_metadata(metadata)},
    )
    return destination
