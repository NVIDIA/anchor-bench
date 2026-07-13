from __future__ import annotations

import json
import math
from pathlib import Path


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
    output_activation: str = "linear",
    output_scale: float = 1.0,
):
    """Build an inference GRU with the same output head used for training."""

    if output_activation not in {"linear", "tanh"}:
        raise ValueError("output_activation must be 'linear' or 'tanh'")
    if not math.isfinite(output_scale) or output_scale <= 0.0:
        raise ValueError("output_scale must be a positive finite value")

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
            self.activation = (
                torch.nn.Tanh() if output_activation == "tanh" else torch.nn.Identity()
            )
            self.output_scale = output_scale

        def forward(self, x, hidden):
            out, hidden = self.gru(x, hidden)
            return self.activation(self.head(out)) * self.output_scale, hidden

    return RuntimeGRU()


def export_gru_checkpoint(checkpoint_path: str | Path, output_path: str | Path | None = None) -> Path:
    """Script a trained GRU and embed its self-contained runtime metadata."""

    torch = _import_torch()
    source = Path(checkpoint_path)
    checkpoint = torch.load(source, map_location="cpu")
    runtime_metadata = checkpoint.get("runtime_metadata")
    if not isinstance(runtime_metadata, dict):
        raise ValueError(f"{source} checkpoint is missing runtime_metadata")
    input_size = int(checkpoint["input_size"])
    output_size = int(checkpoint["output_size"])
    if (runtime_metadata.get("input_size"), runtime_metadata.get("output_size")) != (input_size, output_size):
        raise ValueError(f"{source} checkpoint dimensions disagree with runtime_metadata")
    hparams = checkpoint.get("hyperparameters") or {}
    checkpoint_head = checkpoint.get("output_head")
    metadata_head = runtime_metadata.get("output_head")
    if (checkpoint_head is None) != (metadata_head is None) or (
        checkpoint_head is not None and checkpoint_head != metadata_head
    ):
        raise ValueError(f"{source} checkpoint output_head disagrees with runtime_metadata")
    output_head = checkpoint_head or {}

    model = build_runtime_gru(
        input_size=input_size,
        output_size=output_size,
        hidden_size=int(hparams.get("hidden_size", 64)),
        num_layers=int(hparams.get("num_layers", 1)),
        dropout=float(hparams.get("dropout", 0.0)),
        output_activation=str(output_head.get("activation", "linear")),
        output_scale=float(output_head.get("scale", 1.0)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
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
