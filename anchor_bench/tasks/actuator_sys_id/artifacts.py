"""Artifact writing for completed actuator system-identification runs."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence, Union

import yaml

from anchor_bench.schema.actuator_sys_id import ActuatorSysIdConfig
from anchor_bench.tasks.actuator_sys_id.model import (
    Candidate,
    GenerationRecord,
    SysIdResult,
)
from anchor_bench.tasks.actuator_sys_id.usd import write_identified_usd_layer


def write_artifacts(
    output_dir: Path,
    *,
    best_parameters: Candidate,
    best_mse: float,
    history: Sequence[GenerationRecord],
    source_usd_path: Union[str, Path],
    config: ActuatorSysIdConfig,
) -> SysIdResult:
    """Write the fitted parameters, USD layer, log, and resolved run summary."""

    best_parameters_path = output_dir / "best_params.yaml"
    optimization_log_path = output_dir / "optimization_log.csv"
    run_summary_path = output_dir / "run_summary.yaml"
    identified_usd_path = output_dir / "identified.usda"

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
    run_summary_path.write_text(
        yaml.safe_dump(summary, sort_keys=False),
        encoding="utf-8",
    )
    write_identified_usd_layer(
        source_usd_path,
        identified_usd_path,
        best_parameters,
        config.model_dump_json(),
    )
    return SysIdResult(
        best_parameters=best_parameters,
        best_mse=best_mse,
        generations=len(history),
        output_dir=output_dir,
        best_parameters_path=best_parameters_path,
        optimization_log_path=optimization_log_path,
        run_summary_path=run_summary_path,
        identified_usd_path=identified_usd_path,
    )
