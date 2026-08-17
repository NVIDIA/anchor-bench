from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

import anchor_bench.train.data_processors as data_processor_registry
from anchor_bench.core import SimulationContext
from anchor_bench.train import BaseDataProcessor, DataProcessorRunner


class GeneratedFeatureDataProcessor(BaseDataProcessor):
    """Generate one entity-qualified feature for runner tests."""

    def process(self) -> pd.DataFrame:
        """Return a feature aligned to the source index."""

        return pd.DataFrame(
            {"joint/generated": self.dataset["joint/actual"] * 2.0},
            index=self.dataset.index,
        )


class DependentFeatureDataProcessor(BaseDataProcessor):
    """Consume a feature produced earlier in configuration order."""

    def process(self) -> pd.DataFrame:
        """Return a feature derived from the preceding generated column."""

        return pd.DataFrame(
            {"joint/dependent": self.dataset["joint/generated"] + 1.0},
            index=self.dataset.index,
        )


class MisalignedDataProcessor(BaseDataProcessor):
    """Return generated data in the wrong row order."""

    def process(self) -> pd.DataFrame:
        """Return a feature with a reversed index."""

        return pd.DataFrame(
            {"joint/misaligned": [1.0, 2.0]},
            index=self.dataset.index[::-1],
        )


class StaticOutputDataProcessor(BaseDataProcessor):
    """Return the test-selected output without modifying it."""

    output: Any = None

    def process(self) -> pd.DataFrame:
        """Return the configured test output."""

        return self.output


class FakeNewtonArray:
    """Expose Newton-style host conversion for runner test values."""

    def __init__(self, values: list[float]) -> None:
        """Store values returned by the fake device array."""

        self.values = values

    def numpy(self) -> np.ndarray:
        """Return values as a NumPy array."""

        return np.asarray(self.values)


class RunnerSimulationContext(SimulationContext):
    """Provide the shared context required by runner data processors."""

    def _load(self, usd_path: Path) -> tuple[Any, Any]:
        """Return inert model and state objects."""

        model = SimpleNamespace(
            joint_label=["joint"],
            joint_qd_start=FakeNewtonArray([0, 1]),
            joint_target_ke=FakeNewtonArray([2.0]),
            joint_target_kd=FakeNewtonArray([0.5]),
        )
        return model, object()


@pytest.fixture(autouse=True)
def simulation_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> RunnerSimulationContext:
    """Create the shared simulation context before running data processors."""

    monkeypatch.setattr(SimulationContext, "_instance", None)
    source = tmp_path / "robot.usda"
    source.write_text("#usda 1.0", encoding="utf-8")
    return RunnerSimulationContext(str(source))


@pytest.fixture
def dataset() -> pd.DataFrame:
    """Create one resampled dataset for runner tests."""

    index = pd.MultiIndex.from_tuples(
        [("wave", 0), ("wave", 20_000_000)],
        names=["experiment", "time_ns"],
    )
    return pd.DataFrame({"joint/actual": [1.0, 2.0]}, index=index)


def _register(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    data_processor_type: type[BaseDataProcessor],
) -> None:
    """Temporarily add an in-repo test data processor to the fixed registry."""

    monkeypatch.setitem(
        data_processor_registry.DATA_PROCESSORS,
        name,
        (__name__, data_processor_type.__name__),
    )


def test_runs_builtin_data_processor_by_name() -> None:
    """Resolve and execute a packaged data processor through its fixed name."""

    index = pd.MultiIndex.from_tuples(
        [("wave", 0), ("wave", 20_000_000)],
        names=["experiment", "time_ns"],
    )
    dataset = pd.DataFrame(
        {
            "joint/command_q": [2.0, 4.0],
            "joint/actual_q": [1.0, 1.5],
        },
        index=index,
    )

    result = DataProcessorRunner.run(
        dataset,
        {"position_error": "position_error"},
    )

    assert result["joint/position_error"].tolist() == [1.0, 2.5]


def test_runs_usd_backed_data_processor_in_configuration_order() -> None:
    """Expose an earlier feature to a USD-backed data processor."""

    dataset = pd.DataFrame(
        {
            "joint/command_q": [2.0, 4.0],
            "joint/actual_q": [1.0, 1.5],
            "joint/actual_dq": [0.5, 1.0],
        }
    )

    result = DataProcessorRunner.run(
        dataset,
        {
            "position_error": "position_error",
            "solver_pd": "solver_pd",
        },
    )

    assert result["joint/position_error"].tolist() == [1.0, 2.5]
    assert result["joint/solver_pd"].tolist() == [1.75, 4.5]


def test_runs_data_processors_in_configuration_order(
    dataset: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Append generated features and expose them to later data processors."""

    _register(monkeypatch, "generated", GeneratedFeatureDataProcessor)
    _register(monkeypatch, "dependent", DependentFeatureDataProcessor)
    caplog.set_level(
        logging.DEBUG,
        logger="anchor_bench.tasks.train.data_processor_runner",
    )

    result = DataProcessorRunner.run(
        dataset,
        {
            "actual": None,
            "generated": "generated",
            "dependent": "dependent",
        },
    )

    assert result.columns.tolist() == [
        "joint/actual",
        "joint/generated",
        "joint/dependent",
    ]
    assert result["joint/generated"].tolist() == [2.0, 4.0]
    assert result["joint/dependent"].tolist() == [3.0, 5.0]
    assert result.index.equals(dataset.index)
    assert dataset.columns.tolist() == ["joint/actual"]
    assert [message for message in caplog.messages if "data processor" in message] == [
        "Starting data processor name=generated feature=generated rows=2",
        (
            "Completed data processor name=generated feature=generated "
            "generated_columns=1"
        ),
        "Starting data processor name=dependent feature=dependent rows=2",
        (
            "Completed data processor name=dependent feature=dependent "
            "generated_columns=1"
        ),
    ]


def test_keeps_existing_features_without_copying_data(
    dataset: pd.DataFrame,
) -> None:
    """Share existing column storage when no preprocessing is requested."""

    result = DataProcessorRunner.run(dataset, {"actual": None})

    assert result.equals(dataset)
    assert result is not dataset
    assert np.shares_memory(
        result["joint/actual"].to_numpy(copy=False),
        dataset["joint/actual"].to_numpy(copy=False),
    )


def test_rejects_misaligned_generated_data(
    dataset: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Require generated rows to retain the dataset index."""

    _register(monkeypatch, "misaligned", MisalignedDataProcessor)

    with pytest.raises(ValueError, match="returned a different index"):
        DataProcessorRunner.run(dataset, {"misaligned": "misaligned"})


@pytest.mark.parametrize(
    ("output_kind", "error_type", "message"),
    [
        ("not_dataframe", TypeError, "must return a DataFrame"),
        ("empty", ValueError, "returned an empty DataFrame"),
        ("duplicate", ValueError, "returned duplicate columns"),
        ("collision", ValueError, "Generated columns already exist"),
        ("unrelated", ValueError, "returned columns unrelated"),
    ],
)
def test_rejects_invalid_generated_data(
    dataset: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    output_kind: str,
    error_type: type[Exception],
    message: str,
) -> None:
    """Reject generated output that violates the configured feature contract."""

    if output_kind == "not_dataframe":
        output = pd.Series([1.0, 2.0], index=dataset.index)
    elif output_kind == "empty":
        output = pd.DataFrame(index=dataset.index)
    elif output_kind == "duplicate":
        output = pd.DataFrame(
            [[1.0, 2.0], [3.0, 4.0]],
            columns=pd.Index(["joint/generated", "joint/generated"]),
            index=dataset.index,
        )
    elif output_kind == "collision":
        output = pd.DataFrame(
            {"joint/actual": [3.0, 4.0]},
            index=dataset.index,
        )
    else:
        output = pd.DataFrame(
            {"joint/other": [3.0, 4.0]},
            index=dataset.index,
        )

    _register(monkeypatch, "static", StaticOutputDataProcessor)
    monkeypatch.setattr(StaticOutputDataProcessor, "output", output)

    with pytest.raises(error_type, match=message):
        DataProcessorRunner.run(dataset, {"generated": "static"})


@pytest.mark.parametrize(
    "name",
    ["missing", "external.package:DataProcessor"],
)
def test_rejects_names_outside_builtin_registry(
    dataset: pd.DataFrame,
    name: str,
) -> None:
    """Reject unknown aliases and external module entrypoints."""

    with pytest.raises(AttributeError, match="Unknown data processor"):
        DataProcessorRunner.run(dataset, {"generated": name})
