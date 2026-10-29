# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Artifact writing for completed actuator system-identification runs."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

import yaml

from anchor_bench.schema.simulation import JsonValue
from anchor_bench.tasks.actuator_sys_id.model import (
    Candidate,
    GenerationRecord,
    IdentifiedModelArtifact,
    SysIdResult,
)


def write_artifacts(
    output_dir: Path,
    *,
    best_parameters: Candidate,
    best_mse: float,
    history: Sequence[GenerationRecord],
    model_artifact: IdentifiedModelArtifact | None = None,
    runtime_simulation: Mapping[str, JsonValue] | None = None,
) -> SysIdResult:
    """Write common results and record an optional backend-native model export."""

    best_parameters_path = output_dir / "best_params.yaml"
    optimization_log_path = output_dir / "optimization_log.csv"
    run_summary_path = output_dir / "run_summary.yaml"

    best_document = {
        "schema_version": 1,
        "type": "actuator_sys_id_result",
        "metric": "position_mse",
        "best_mse": best_mse,
        "parameters": best_parameters,
    }
    best_parameters_path.write_text(
        yaml.safe_dump(best_document, sort_keys=False),
        encoding="utf-8",
    )

    with optimization_log_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["generation", "best_mse", "population_mean_mse"])
        for record in history:
            writer.writerow(
                [
                    record.generation,
                    record.best_mse,
                    record.population_mean_mse,
                ]
            )

    summary = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "metric": "position_mse",
        "best_mse": best_mse,
        "generations": len(history),
    }
    if runtime_simulation is not None:
        summary["runtime_simulation"] = dict(runtime_simulation)
    run_summary_path.write_text(
        yaml.safe_dump(summary, sort_keys=False),
        encoding="utf-8",
    )
    return SysIdResult(
        best_parameters=best_parameters,
        best_mse=best_mse,
        generations=len(history),
        output_dir=output_dir,
        best_parameters_path=best_parameters_path,
        optimization_log_path=optimization_log_path,
        run_summary_path=run_summary_path,
        identified_model_path=model_artifact.path if model_artifact else None,
        identified_usd_path=(
            model_artifact.path
            if model_artifact is not None and model_artifact.format == "usd"
            else None
        ),
    )
