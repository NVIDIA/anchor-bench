# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load repeated endpoint measurements without changing experiment weights."""

import csv
import hashlib
import io
import math
import statistics
from pathlib import Path

from anchor_bench.errors import DatasetLoadError
from anchor_bench.schema.mpm_sys_id import MpmSysIdConfig


def load_experiments(config: MpmSysIdConfig) -> list[dict]:
    results = []
    for experiment in config.experiments:
        if not experiment.enabled:
            continue
        columns = [
            "width_m",
            "length_m",
            "height_m" if experiment.simulation.type == "sand_pour" else "depth_m",
        ]
        path = Path(experiment.input_data.source)
        try:
            raw = path.read_bytes()
            reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
            if reader.fieldnames != columns:
                raise ValueError(f"Expected CSV header {','.join(columns)}")
            trials = []
            for row in reader:
                if set(row) != set(columns):
                    raise ValueError("Unexpected extra CSV cells")
                trials.append([float(row[key]) for key in columns])
            if not trials or any(
                not math.isfinite(v) or v <= 0 for row in trials for v in row
            ):
                raise ValueError(
                    "Require at least one repeat with finite positive measurements in meters"
                )
        except (OSError, UnicodeError, ValueError, TypeError, csv.Error) as exc:
            raise DatasetLoadError(f"{experiment.id}: {path}: {exc}") from exc
        target = [statistics.mean(col) for col in zip(*trials)]
        results.append(
            {
                "id": experiment.id,
                "weight": experiment.weight,
                "simulation": experiment.simulation.model_dump(mode="json"),
                "input_data": {
                    "type": "csv",
                    "source": str(path),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                },
                "dimensions": columns,
                "trials": trials,
                "target": target,
                "sample_std_m": [
                    statistics.stdev(col) if len(trials) > 1 else None
                    for col in zip(*trials)
                ],
            }
        )
    return results
