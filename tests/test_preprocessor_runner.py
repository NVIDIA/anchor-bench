from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

import anchor_bench.train.preprocessors as preprocessor_registry
from anchor_bench.core import SimulationContext
from anchor_bench.train import BasePreprocessor, PreprocessorRunner


class GeneratedFeaturePreprocessor(BasePreprocessor):
    """Generate one entity-qualified feature for runner tests."""

    def process(self) -> pd.DataFrame:
        """Return a feature aligned to the source index."""

        return pd.DataFrame(
            {"joint/generated": self.dataset["joint/actual"] * 2.0},
            index=self.dataset.index,
        )


class DependentFeaturePreprocessor(BasePreprocessor):
    """Consume a feature produced earlier in configuration order."""

    def process(self) -> pd.DataFrame:
        """Return a feature derived from the preceding generated column."""

        return pd.DataFrame(
            {"joint/dependent": self.dataset["joint/generated"] + 1.0},
            index=self.dataset.index,
        )


class MisalignedPreprocessor(BasePreprocessor):
    """Return generated data in the wrong row order."""

    def process(self) -> pd.DataFrame:
        """Return a feature with a reversed index."""

        return pd.DataFrame(
            {"joint/misaligned": [1.0, 2.0]},
            index=self.dataset.index[::-1],
        )


class StaticOutputPreprocessor(BasePreprocessor):
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
    """Provide the shared context required by runner preprocessors."""

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
    """Create the shared simulation context before running preprocessors."""

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
    preprocessor_type: type[BasePreprocessor],
) -> None:
    """Temporarily add an in-repo test preprocessor to the fixed registry."""

    monkeypatch.setitem(
        preprocessor_registry.PREPROCESSORS,
        name,
        (__name__, preprocessor_type.__name__),
    )


def test_runs_builtin_preprocessor_by_name() -> None:
    """Resolve and execute a packaged preprocessor through its fixed name."""

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

    result = PreprocessorRunner.run(
        dataset,
        {"position_error": "position_error"},
    )

    assert result["joint/position_error"].tolist() == [1.0, 2.5]


def test_runs_usd_backed_preprocessor_in_configuration_order() -> None:
    """Expose an earlier feature to a USD-backed preprocessor."""

    dataset = pd.DataFrame(
        {
            "joint/command_q": [2.0, 4.0],
            "joint/actual_q": [1.0, 1.5],
            "joint/actual_dq": [0.5, 1.0],
        }
    )

    result = PreprocessorRunner.run(
        dataset,
        {
            "position_error": "position_error",
            "solver_pd": "solver_pd",
        },
    )

    assert result["joint/position_error"].tolist() == [1.0, 2.5]
    assert result["joint/solver_pd"].tolist() == [1.75, 4.5]


def test_runs_preprocessors_in_configuration_order(
    dataset: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Append generated features and expose them to later preprocessors."""

    _register(monkeypatch, "generated", GeneratedFeaturePreprocessor)
    _register(monkeypatch, "dependent", DependentFeaturePreprocessor)

    result = PreprocessorRunner.run(
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


def test_keeps_existing_features_without_copying_data(
    dataset: pd.DataFrame,
) -> None:
    """Share existing column storage when no preprocessing is requested."""

    result = PreprocessorRunner.run(dataset, {"actual": None})

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

    _register(monkeypatch, "misaligned", MisalignedPreprocessor)

    with pytest.raises(ValueError, match="returned a different index"):
        PreprocessorRunner.run(dataset, {"misaligned": "misaligned"})


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

    _register(monkeypatch, "static", StaticOutputPreprocessor)
    monkeypatch.setattr(StaticOutputPreprocessor, "output", output)

    with pytest.raises(error_type, match=message):
        PreprocessorRunner.run(dataset, {"generated": "static"})


@pytest.mark.parametrize(
    "name",
    ["missing", "external.package:Preprocessor"],
)
def test_rejects_names_outside_builtin_registry(
    dataset: pd.DataFrame,
    name: str,
) -> None:
    """Reject unknown aliases and external module entrypoints."""

    with pytest.raises(AttributeError, match="Unknown preprocessor"):
        PreprocessorRunner.run(dataset, {"generated": name})
