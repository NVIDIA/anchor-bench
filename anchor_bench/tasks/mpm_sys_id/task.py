# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resumable material fitting with isolated candidate evaluation."""

import hashlib
import importlib.metadata
import json
import logging
import signal
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from filelock import FileLock, Timeout

from anchor_bench.errors import AnchorBenchError
from anchor_bench.mlops import get_active_run
from anchor_bench.schema.mpm_sys_id import DEFAULT_BOUNDS, MpmSysIdConfig
from anchor_bench.tasks.base_task import BaseTask

from .artifacts import read_json, write_json
from .dataset import load_experiments
from .objective import score

logger = logging.getLogger(__name__)


class MpmSysIdTask(BaseTask[MpmSysIdConfig]):
    @contextmanager
    def output_context(
        self, output: Path, *, force: bool = False, resume: bool = False
    ):
        try:
            with FileLock(str(output / ".run.lock"), timeout=0):
                experiments = load_experiments(self.config)
                # Versions are part of identity: optimizer replay and physics may change.
                versions = {
                    p: importlib.metadata.version(p)
                    for p in ("newton", "warp-lang", "numpy", "scikit-learn")
                }
                identity = self.config.model_dump(
                    mode="json",
                    exclude={
                        "output_dir",
                        "description",
                        "tags",
                        "mlops",
                        "experiments",
                    },
                )
                identity["experiments"] = experiments
                manifest = {
                    "schema_version": 1,
                    "implementation_version": 1,
                    "configuration": identity,
                    "versions": versions,
                }
                path = output / "inputs.json"
                if path.exists():
                    if read_json(path) != manifest:
                        raise AnchorBenchError(
                            "MPM inputs or dependency versions changed; use a new output directory"
                        )
                    if force:
                        raise AnchorBenchError(
                            "MPM --force does not overwrite checkpoints; use a new output directory"
                        )
                    if not (output / ".completed").exists() and not resume:
                        raise AnchorBenchError(
                            "Incomplete MPM run exists; pass --resume with the same output directory"
                        )
                else:
                    if resume:
                        raise AnchorBenchError(
                            "Cannot resume: inputs.json does not exist"
                        )
                    if (output / "trials").exists() or (output / ".completed").exists():
                        raise AnchorBenchError(
                            "Output contains artifacts without MPM inputs; use a new directory"
                        )
                    write_json(path, manifest)
                self.experiments = experiments
                yield
        except Timeout as exc:
            raise AnchorBenchError(f"Another MPM process owns {output}") from exc
        except importlib.metadata.PackageNotFoundError as exc:
            raise AnchorBenchError(
                "Install MPM dependencies with uv sync --extra mpm"
            ) from exc

    def _evaluate(self, request: dict, directory: Path) -> dict:
        directory.mkdir(parents=True, exist_ok=True)
        request_path = directory / "request.json"
        response_path = directory / "response.json"
        if request_path.exists():
            if read_json(request_path) != request:
                raise AnchorBenchError(f"Candidate replay mismatch: {request_path}")
        else:
            write_json(request_path, request)
        digest = hashlib.sha256(request_path.read_bytes()).hexdigest()
        if not response_path.exists():
            with (directory / "worker.log").open("a") as log:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "anchor_bench.tasks.mpm_sys_id.worker",
                        str(request_path),
                        str(response_path),
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                try:
                    code = process.wait(
                        timeout=float(self.config.execution.worker_timeout_s)
                    )
                    if code:
                        raise AnchorBenchError(
                            f"MPM worker exited {code}; inspect {directory / 'worker.log'}"
                        )
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    return {
                        e["id"]: {"valid": False, "failure_reason": "Worker timed out"}
                        for e in self.experiments
                    }
                except BaseException:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                    raise
        response = read_json(response_path)
        if response["request_sha256"] != digest:
            raise AnchorBenchError("Worker response does not match candidate request")
        return response["measurements"]

    def run(self, output_directory: Path) -> Path:
        from .optimizer import decode, propose

        output = output_directory.resolve()
        trials = output / "trials"
        trials.mkdir(exist_ok=True)
        history = []
        for path in sorted(trials.glob("*/result.json")):
            record = read_json(path)
            if record["evaluation"] != len(history) + 1:
                raise AnchorBenchError(
                    "Non-contiguous trial history; cannot resume safely"
                )
            history.append(record)
        bounds = {
            name: [b.lower, b.upper, b.scale, b.offset]
            for name in DEFAULT_BOUNDS
            for b in [self.config.material.optimized[name]]
        }
        stopping = self.config.optimizer.stopping
        tracked = get_active_run()
        if tracked is not None:
            import pandas as pd

            for experiment in self.experiments:
                tracked.log_dataset(
                    pd.DataFrame(
                        experiment["trials"], columns=experiment["dimensions"]
                    ),
                    source=experiment["input_data"]["source"],
                    digest=experiment["input_data"]["sha256"],
                    name=experiment["id"],
                    path=experiment["input_data"]["source"],
                    upload=bool(self.config.mlops and self.config.mlops.upload_dataset),
                )
        log_handler = logging.FileHandler(output / "runner.log")
        logger.addHandler(log_handler)

        def persist(state: str) -> None:
            valid = [h for h in history if h["valid"]]
            if valid:
                write_json(output / "best.json", min(valid, key=lambda h: h["error"]))
            write_json(
                output / "checkpoint.json",
                {
                    "schema_version": 1,
                    "next_evaluation": len(history) + 1,
                    "seed": self.config.execution.seed,
                },
            )
            # Per-trial result.json is authoritative; rebuild this derived index on resume.
            temporary = output / "history.jsonl.tmp"
            temporary.write_text(
                "".join(json.dumps(h, allow_nan=False) + "\n" for h in history)
            )
            temporary.replace(output / "history.jsonl")
            write_json(
                output / "status.json",
                {
                    "state": state,
                    "completed": len(history),
                    "valid": len(valid),
                    "current_evaluation": len(history) + 1
                    if state == "running"
                    else None,
                },
            )

        previous_handler = None
        if threading.current_thread() is threading.main_thread():

            def interrupted(signum, frame):
                raise KeyboardInterrupt

            previous_handler = signal.signal(signal.SIGTERM, interrupted)
        try:
            persist("running")
            while True:
                if stopping.mode == "bounded":
                    if len(history) >= int(stopping.max_evaluations or 0):
                        break
                    if stopping.target_error is not None and any(
                        h["valid"]
                        and all(
                            e["error"] <= stopping.target_error
                            for e in h["experiments"].values()
                        )
                        for h in history
                    ):
                        break
                point, strategy = propose(
                    history,
                    bounds,
                    seed=self.config.execution.seed,
                    initial_samples=self.config.optimizer.initial_samples,
                    pool_size=self.config.optimizer.pool_size,
                    exploration=self.config.optimizer.exploration,
                )
                material = {**decode(point, bounds), **self.config.material.fixed}
                evaluation = len(history) + 1
                directory = trials / f"{evaluation:08d}"
                worker_config = self.config.model_dump(
                    mode="json", exclude={"output_dir", "description", "tags", "mlops"}
                )
                worker_config["experiments"] = [
                    e.model_dump(mode="json")
                    for e in self.config.experiments
                    if e.enabled
                ]
                request = {
                    "schema_version": 1,
                    "evaluation": evaluation,
                    "material": material,
                    "config": worker_config,
                }
                start = time.monotonic()
                measurements = self._evaluate(request, directory)
                record = {
                    "schema_version": 1,
                    "evaluation": evaluation,
                    "point": point,
                    "strategy": strategy,
                    "material": material,
                    "elapsed_s": time.monotonic() - start,
                    **score(self.experiments, measurements),
                }
                write_json(directory / "result.json", record)
                history.append(record)
                persist("running")
                logger.info(
                    "MPM evaluation=%d valid=%s combined_relative_rms_percent=%s",
                    evaluation,
                    record["valid"],
                    100 * record["error"] if record["valid"] else None,
                )
                if tracked is not None:
                    metrics = {"optimization/valid": float(record["valid"])}
                    valid_records = [h for h in history if h["valid"]]
                    metrics["optimization/completed"] = float(len(history))
                    metrics["optimization/valid_count"] = float(len(valid_records))
                    if valid_records:
                        best = min(valid_records, key=lambda h: h["error"])
                        metrics["optimization/best_relative_rms"] = best["error"]
                        metrics.update(
                            {
                                f"material/best/{name}": float(value)
                                for name, value in best["material"].items()
                            }
                        )
                    if record["valid"]:
                        metrics["optimization/relative_rms"] = record["error"]
                        for name, result in record["experiments"].items():
                            metrics[f"experiments/{name}/relative_rms"] = result[
                                "error"
                            ]
                    tracked.log_metrics(metrics, step=evaluation)
            persist("completed")
            if tracked is not None:
                tracked.log_artifact(
                    output, collection="mpm-sysid-output", artifact_type="mpm-sysid"
                )
            return output_directory
        except KeyboardInterrupt:
            persist("interrupted")
            raise
        except BaseException:
            persist("failed")
            raise
        finally:
            if previous_handler is not None:
                signal.signal(signal.SIGTERM, previous_handler)
            logger.removeHandler(log_handler)
            log_handler.close()
