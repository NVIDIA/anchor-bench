from __future__ import annotations

from typing import Any

from pydantic import Field, StrictBool, field_validator

from anchor_bench.core.config_model import ConfigModel


EXPERIMENT_ACTUATOR_TYPES = {"pd", "pid", "mlp", "lstm", "gru", "fmu", "torque_passthrough", "implicit"}
EXPERIMENT_BENCHMARK_PARAMS = {"initial", "fitted"}

DatasetMotion = str | dict[str, str]
DatasetMotionSelection = str | list[DatasetMotion] | None
AnalysisSelector = str | list[str] | None


def _parse_seeds(value: Any) -> list[int] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError("must be a list")
    return [int(seed) for seed in value]


def _parse_selector(value: Any) -> AnalysisSelector:
    if value is None or isinstance(value, str):
        return value
    if not isinstance(value, list):
        raise ValueError("must be a list or string")
    return [str(item) for item in value]


def _parse_dataset_motions(value: Any) -> DatasetMotionSelection:
    if value is None or isinstance(value, str):
        return value
    if not isinstance(value, list):
        raise ValueError("must be a list or string")
    motions: list[DatasetMotion] = []
    for index, item in enumerate(value):
        if isinstance(item, str):
            motions.append(item)
            continue
        if not isinstance(item, dict):
            raise ValueError(f"[{index}] must be a string or mapping")
        unknown = sorted(set(item) - {"name", "motion", "path"})
        if unknown:
            raise ValueError(f"[{index}] has unknown keys: {', '.join(unknown)}")
        motions.append({str(key): str(entry) for key, entry in item.items()})
    return motions


class ExperimentOutputs(ConfigModel):
    benchmark: str | None = None
    analysis: str | None = None


class ExperimentBenchmark(ConfigModel):
    num_envs: int | None = Field(default=None, ge=1)
    headless: StrictBool | None = None


class ExperimentDataset(ConfigModel):
    real_root: str
    motions: DatasetMotionSelection = None

    @field_validator("motions", mode="before")
    @classmethod
    def _validate_motions(cls, value: Any) -> DatasetMotionSelection:
        return _parse_dataset_motions(value)


class ExperimentActuatorModel(ConfigModel):
    type: str
    initial_params: str | None = None
    fitted_params: str | None = None
    benchmark_params: str = "fitted"

    @field_validator("type")
    @classmethod
    def _validate_type(cls, value: str) -> str:
        if value not in EXPERIMENT_ACTUATOR_TYPES:
            raise ValueError(f"{value!r} must be one of {sorted(EXPERIMENT_ACTUATOR_TYPES)}")
        return value

    @field_validator("benchmark_params")
    @classmethod
    def _validate_benchmark_params(cls, value: str) -> str:
        if value not in EXPERIMENT_BENCHMARK_PARAMS:
            raise ValueError(f"{value!r} must be one of {sorted(EXPERIMENT_BENCHMARK_PARAMS)}")
        return value


class ExperimentAnalysisPlots(ConfigModel):
    joints: StrictBool | None = None
    end_effector: StrictBool | None = None
    require_end_effector: StrictBool | None = None


class ExperimentAnalysis(ConfigModel):
    seeds: list[int] | None = None
    datasets: AnalysisSelector = None
    motions: AnalysisSelector = None
    actuator_models: AnalysisSelector = None
    plots: ExperimentAnalysisPlots | None = None

    @field_validator("seeds", mode="before")
    @classmethod
    def _validate_seeds(cls, value: Any) -> list[int] | None:
        return _parse_seeds(value)

    @field_validator("datasets", "motions", "actuator_models", mode="before")
    @classmethod
    def _validate_selectors(cls, value: Any) -> AnalysisSelector:
        return _parse_selector(value)


class ExperimentFile(ConfigModel):
    schema_version: int = Field(ge=1)
    robot: str
    outputs: ExperimentOutputs = Field(default_factory=ExperimentOutputs)
    seeds: list[int] | None = None
    benchmark: ExperimentBenchmark = Field(default_factory=ExperimentBenchmark)
    datasets: dict[str, ExperimentDataset]
    actuator_models: dict[str, ExperimentActuatorModel]
    analysis: ExperimentAnalysis = Field(default_factory=ExperimentAnalysis)

    @field_validator("seeds", mode="before")
    @classmethod
    def _validate_seeds(cls, value: Any) -> list[int] | None:
        return _parse_seeds(value)

    @field_validator("datasets")
    @classmethod
    def _validate_datasets(cls, value: dict[str, ExperimentDataset]) -> dict[str, ExperimentDataset]:
        if not value:
            raise ValueError("must be a non-empty mapping")
        return value

    @field_validator("actuator_models")
    @classmethod
    def _validate_actuator_models(
        cls,
        value: dict[str, ExperimentActuatorModel],
    ) -> dict[str, ExperimentActuatorModel]:
        if not value:
            raise ValueError("must be a non-empty mapping")
        return value
