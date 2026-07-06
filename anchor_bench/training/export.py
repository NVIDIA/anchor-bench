from __future__ import annotations

import json
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


def export_gru_checkpoint(checkpoint_path: str | Path, output_path: str | Path | None = None) -> Path:
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
    if not target_columns:
        raise ValueError(f"{source} checkpoint is missing target_columns")

    model = build_runtime_gru(
        input_size=len(input_columns),
        output_size=len(target_columns),
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
    torch.jit.save(
        scripted,
        str(destination),
        _extra_files={"metadata.json": json.dumps(runtime_metadata, sort_keys=True)},
    )
    return destination
