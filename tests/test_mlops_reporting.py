from pathlib import Path
from typing import Mapping, Optional, Union

import pandas as pd
import pytest

from anchor_bench.mlops import (
    MLOpsRun,
    bind_active_run,
    flatten_params,
    get_active_run,
)
from anchor_bench.mlops.base import ParameterValue


class RecordingRun(MLOpsRun):
    def __init__(self) -> None:
        self.params: dict[str, ParameterValue] = {}
        self.metrics: list[tuple[dict[str, float], Optional[int]]] = []
        self.tags: list[dict[str, str]] = []
        self.artifacts: list[tuple[Path, Optional[str], str]] = []

    @property
    def run_id(self) -> str:
        return "recording-run"

    def log_params(self, params: Mapping[str, ParameterValue]) -> None:
        self.params.update(params)

    def log_metrics(
        self,
        metrics: Mapping[str, float],
        *,
        step: Optional[int] = None,
    ) -> None:
        self.metrics.append((dict(metrics), step))

    def set_tags(self, tags: Mapping[str, str]) -> None:
        self.tags.append(dict(tags))

    def log_artifact(
        self,
        path: Union[str, Path],
        *,
        collection: Optional[str] = None,
        artifact_type: str = "anchor-bench",
    ) -> None:
        self.artifacts.append((Path(path), collection, artifact_type))


@pytest.mark.parametrize("upload", [False, True])
def test_default_dataset_logging_records_lineage_and_optional_payload(
    tmp_path: Path,
    upload: bool,
) -> None:
    dataset = pd.DataFrame({"experiment": ["wave", "step"], "value": [1.0, 2.0]})
    source_path = tmp_path / "dataset.parquet"
    source_path.write_text("dataset", encoding="utf-8")
    run = RecordingRun()

    run.log_dataset(
        dataset,
        source="datasets/actuator.parquet",
        digest="abc123",
        name="training-dataset",
        context="training",
        path=source_path,
        metadata={"format": "parquet", "size_bytes": 7},
        upload=upload,
    )

    assert run.params == {
        "datasets/training-dataset/source": "datasets/actuator.parquet",
        "datasets/training-dataset/digest": "abc123",
        "datasets/training-dataset/context": "training",
        "datasets/training-dataset/rows": 2,
        "datasets/training-dataset/columns": '["experiment","value"]',
        "datasets/training-dataset/uploaded": upload,
        "datasets/training-dataset/metadata/format": "parquet",
        "datasets/training-dataset/metadata/size_bytes": 7,
    }
    assert run.artifacts == (
        [(source_path, "training-dataset", "dataset")] if upload else []
    )


def test_flatten_params_encodes_nested_json_values() -> None:
    assert flatten_params(
        {
            "epochs": 4,
            "nested": {"enabled": True, "missing": None},
            "features": ["position", "velocity"],
            "empty": {},
        }
    ) == {
        "epochs": 4,
        "nested/enabled": True,
        "nested/missing": "null",
        "features/0": "position",
        "features/1": "velocity",
        "empty": "{}",
    }


def test_active_run_binding_is_nested_and_context_local() -> None:
    outer = RecordingRun()
    inner = RecordingRun()

    assert get_active_run() is None
    with bind_active_run(outer) as bound_outer:
        assert bound_outer is outer
        assert get_active_run() is outer
        with bind_active_run(inner) as bound_inner:
            assert bound_inner is inner
            assert get_active_run() is inner
        assert get_active_run() is outer
    assert get_active_run() is None
