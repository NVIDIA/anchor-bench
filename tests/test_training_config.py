from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from anchor_bench.schema.dataset import ParquetDatasetConfig
from anchor_bench.schema.parse import parse
from anchor_bench.schema.training import (
    GruModelConfig,
    HuberLossConfig,
    IntegerSearchParameter,
    TrainingConfig,
)


@pytest.fixture
def valid_config_data():
    return {
        "type": "training",
        "description": "Train an actuator network.",
        "tags": ["actuator", "baseline"],
        "dataset": {
            "type": "parquet",
            "source": "datasets/actuator.parquet",
            "resampling": {
                "type": "uniform",
                "dt": 0.02,
                "time_range": "intersection",
                "interpolation": "linear",
            },
        },
        "simulation": {
            "backend": "newton",
            "usd_path": "assets/robots/so101.usd",
        },
        "input_features": {
            "position": "joint_position",
            "velocity": "joint_velocity",
            "command": None,
        },
        "output_features": "joint_effort",
        "mapping": {
            "actuator": {
                "inputs": ["position", "velocity", "command"],
                "outputs": ["joint_effort"],
            }
        },
        "model": {
            "type": "gru",
            "hidden_size": 128,
            "num_layers": 2,
            "dropout": 0.1,
            "output": {
                "activation": "identity",
                "scale": 1.0,
            },
        },
        "training": {
            "epochs": 100,
            "batch_size": 64,
            "seed": 42,
            "validation_ratio": 0.2,
            "sequence": {
                "length": 32,
                "stride": 8,
                "hidden_state": "reset",
            },
            "optimizer": {
                "type": "adamw",
                "learning_rate": 0.001,
                "weight_decay": 0.01,
            },
            "loss": {
                "type": "huber",
                "delta": 1.0,
            },
            "gradient": {
                "clip_norm": 1.0,
            },
            "augmentation": {
                "input_noise_std": 0.01,
            },
        },
        "search": {
            "schema_version": 1,
            "method": "optuna",
            "trials": 20,
            "objective": {
                "metric": "validation_loss",
                "direction": "minimize",
            },
            "search_space": {
                "model.hidden_size": {
                    "data_type": "integer",
                    "min": 64,
                    "max": 256,
                    "step": 64,
                    "scale": "linear",
                },
                "training.optimizer.learning_rate": {
                    "data_type": "float",
                    "min": 0.00001,
                    "max": 0.01,
                    "scale": "log",
                },
                "training.sequence.hidden_state": {
                    "data_type": "categorical",
                    "values": ["reset"],
                },
            },
        },
        "validation": {
            "validators": [
                {
                    "name": "held-out loss",
                    "type": "offline",
                    "entrypoint": "validators:held_out_loss",
                    "config": {"split": "test"},
                }
            ]
        },
    }


def test_model_matches_general_training_schema(valid_config_data):
    config = TrainingConfig.model_validate(valid_config_data)

    assert config.type == "training"
    assert config.output_dir == "outputs/training"
    assert isinstance(config.dataset, ParquetDatasetConfig)
    assert config.dataset.source == "datasets/actuator.parquet"
    assert config.dataset.resampling.type == "uniform"
    assert config.dataset.resampling.dt == 0.02
    assert config.dataset.resampling.time_range == "intersection"
    assert config.dataset.resampling.interpolation == "linear"
    assert config.simulation.backend == "newton"
    assert config.simulation.usd_path == "assets/robots/so101.usd"
    assert isinstance(config.model, GruModelConfig)
    assert isinstance(config.training.loss, HuberLossConfig)
    assert config.search is not None
    assert isinstance(
        config.search.search_space["model.hidden_size"],
        IntegerSearchParameter,
    )


def test_parser_selects_training_model(valid_config_data, tmp_path: Path):
    path = tmp_path / "training.yaml"
    path.write_text(yaml.safe_dump(valid_config_data), encoding="utf-8")

    config = parse(path)

    assert isinstance(config, TrainingConfig)


def test_optional_metadata_and_search_can_be_omitted(valid_config_data):
    data = deepcopy(valid_config_data)
    data.pop("description")
    data.pop("tags")
    data.pop("search")
    data["input_features"] = {"actual_q": None}

    config = TrainingConfig.model_validate(data)

    assert config.description is None
    assert config.tags is None
    assert config.search is None


def test_simulation_defaults_to_newton(valid_config_data):
    data = deepcopy(valid_config_data)
    data["simulation"].pop("backend")

    config = TrainingConfig.model_validate(data)

    assert config.simulation.backend == "newton"


def test_requires_simulation(valid_config_data):
    data = deepcopy(valid_config_data)
    data.pop("simulation")
    data["input_features"] = {"actual_q": None}

    with pytest.raises(ValidationError, match="simulation"):
        TrainingConfig.model_validate(data)


def test_rejects_blank_preprocessor_name(valid_config_data):
    data = deepcopy(valid_config_data)
    data["input_features"]["position"] = ""

    with pytest.raises(
        ValidationError, match="String should have at least 1 character"
    ):
        TrainingConfig.model_validate(data)


@pytest.mark.parametrize(
    "field_name",
    ["description", "tags", "search"],
)
def test_optional_fields_reject_explicit_null(valid_config_data, field_name):
    data = deepcopy(valid_config_data)
    data[field_name] = None

    with pytest.raises(ValidationError, match=f"{field_name} cannot be null"):
        TrainingConfig.model_validate(data)


def test_model_rejects_old_gru_task_discriminator(valid_config_data):
    data = deepcopy(valid_config_data)
    data["type"] = "gru"

    with pytest.raises(ValidationError):
        TrainingConfig.model_validate(data)


def test_resampling_rejects_non_positive_dt(valid_config_data):
    data = deepcopy(valid_config_data)
    data["dataset"]["resampling"]["dt"] = 0.0

    with pytest.raises(ValidationError):
        TrainingConfig.model_validate(data)


def test_resampling_requires_type(valid_config_data):
    data = deepcopy(valid_config_data)
    data["dataset"]["resampling"].pop("type")

    with pytest.raises(ValidationError):
        TrainingConfig.model_validate(data)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("type", "reference"),
        ("time_range", "union"),
        ("interpolation", "nearest"),
    ],
)
def test_resampling_rejects_unsupported_policy(
    valid_config_data,
    field_name,
    value,
):
    data = deepcopy(valid_config_data)
    data["dataset"]["resampling"][field_name] = value

    with pytest.raises(ValidationError):
        TrainingConfig.model_validate(data)


def test_resampling_rejects_reference_field(valid_config_data):
    data = deepcopy(valid_config_data)
    data["dataset"]["resampling"]["reference_field"] = "actual_q"

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        TrainingConfig.model_validate(data)


def test_simulation_rejects_unsupported_backend(valid_config_data):
    data = deepcopy(valid_config_data)
    data["simulation"]["backend"] = "mujoco"

    with pytest.raises(ValidationError):
        TrainingConfig.model_validate(data)


def test_model_rejects_extra_nested_properties(valid_config_data):
    data = deepcopy(valid_config_data)
    data["model"]["unexpected"] = True

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        TrainingConfig.model_validate(data)


def test_mapping_rejects_duplicate_joint_names(valid_config_data):
    data = deepcopy(valid_config_data)
    data["mapping"]["actuator"]["inputs"] = ["position", "position"]

    with pytest.raises(ValidationError, match="joint names must be unique"):
        TrainingConfig.model_validate(data)


def test_search_rejects_parameter_name_without_object_path(valid_config_data):
    data = deepcopy(valid_config_data)
    parameter = data["search"]["search_space"].pop("model.hidden_size")
    data["search"]["search_space"]["hidden_size"] = parameter

    with pytest.raises(ValidationError, match="String should match pattern"):
        TrainingConfig.model_validate(data)


def test_search_rejects_duplicate_categorical_values(valid_config_data):
    data = deepcopy(valid_config_data)
    data["search"]["search_space"]["training.sequence.hidden_state"]["values"] = [
        "reset",
        "reset",
    ]

    with pytest.raises(
        ValidationError,
        match="categorical search values must be unique",
    ):
        TrainingConfig.model_validate(data)


@pytest.mark.parametrize(
    ("parameter_name", "minimum", "maximum"),
    [
        ("model.hidden_size", 256, 64),
        ("training.optimizer.learning_rate", 0.01, 0.00001),
    ],
)
def test_search_rejects_minimum_greater_than_maximum(
    valid_config_data,
    parameter_name,
    minimum,
    maximum,
):
    data = deepcopy(valid_config_data)
    parameter = data["search"]["search_space"][parameter_name]
    parameter["min"] = minimum
    parameter["max"] = maximum

    with pytest.raises(
        ValidationError,
        match="min must be less than or equal to max",
    ):
        TrainingConfig.model_validate(data)


def test_integer_log_search_rejects_minimum_below_one(valid_config_data):
    data = deepcopy(valid_config_data)
    parameter = data["search"]["search_space"]["model.hidden_size"]
    parameter["min"] = 0
    parameter["scale"] = "log"

    with pytest.raises(
        ValidationError,
        match="min must be at least 1 when scale is 'log'",
    ):
        TrainingConfig.model_validate(data)


def test_float_log_search_rejects_non_positive_minimum(valid_config_data):
    data = deepcopy(valid_config_data)
    parameter = data["search"]["search_space"]["training.optimizer.learning_rate"]
    parameter["min"] = 0.0

    with pytest.raises(
        ValidationError,
        match="min must be greater than 0 when scale is 'log'",
    ):
        TrainingConfig.model_validate(data)


def test_integer_log_search_rejects_step_other_than_one(valid_config_data):
    data = deepcopy(valid_config_data)
    parameter = data["search"]["search_space"]["model.hidden_size"]
    parameter["scale"] = "log"
    parameter["step"] = 2

    with pytest.raises(
        ValidationError,
        match="step must be 1 when scale is 'log'",
    ):
        TrainingConfig.model_validate(data)


def test_float_log_search_rejects_step(valid_config_data):
    data = deepcopy(valid_config_data)
    parameter = data["search"]["search_space"]["training.optimizer.learning_rate"]
    parameter["step"] = 0.001

    with pytest.raises(
        ValidationError,
        match="step must be omitted when scale is 'log'",
    ):
        TrainingConfig.model_validate(data)


@pytest.mark.parametrize(
    "parameter_name",
    ["model.hidden_size", "training.optimizer.learning_rate"],
)
def test_search_rejects_explicit_null_step(valid_config_data, parameter_name):
    data = deepcopy(valid_config_data)
    parameter = data["search"]["search_space"][parameter_name]
    parameter["step"] = None

    with pytest.raises(ValidationError, match="step must be"):
        TrainingConfig.model_validate(data)


def test_model_rejects_coerced_training_integer(valid_config_data):
    data = deepcopy(valid_config_data)
    data["training"]["epochs"] = "100"

    with pytest.raises(ValidationError):
        TrainingConfig.model_validate(data)
