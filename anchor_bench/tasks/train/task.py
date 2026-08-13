"""Training task implementation."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import optuna
import pandas as pd

from anchor_bench.core.simulation.registry import create_simulation_context
from anchor_bench.data import DatasetLoader
from anchor_bench.mlops import get_active_run
from anchor_bench.schema.training import (
    CategoricalSearchParameter,
    FloatSearchParameter,
    IntegerSearchParameter,
    SearchParameter,
    TrainingConfig,
)
from anchor_bench.tasks.base_task import BaseTask
from anchor_bench.tasks.train.data_processor_runner import DataProcessorRunner

if TYPE_CHECKING:
    from anchor_bench.tasks.train.trainer import TrainingResult

TRAJECTORIES_FILENAME = "trajectories.parquet"


@dataclass(frozen=True)
class EntityMapping:
    """Describe one ordered input-to-output entity mapping."""

    inputs: Sequence[str]
    outputs: Sequence[str]


@dataclass(frozen=True)
class MappedTrainingSequence:
    """Preserve one experiment and mapping as a recurrent sequence."""

    experiment: object
    name: str
    inputs: pd.DataFrame
    targets: pd.DataFrame


@dataclass(frozen=True)
class SearchTrial:
    """Record one completed search trial."""

    number: int
    parameters: Mapping[str, Any]
    value: float


@dataclass(frozen=True)
class SearchResult:
    """Return the best training run and the complete trial history."""

    training_result: TrainingResult
    best_config: TrainingConfig
    validation_sequences: tuple[MappedTrainingSequence, ...]
    objective_metric: str
    objective_direction: str
    best_trial: int
    best_value: float
    best_parameters: Mapping[str, Any]
    trials: tuple[SearchTrial, ...]


class TrainingTask(BaseTask[TrainingConfig]):
    """Prepare data, train the configured network, and persist trajectories."""

    @classmethod
    def name(cls) -> str:
        """Return the training configuration discriminator."""

        return "training"

    def run(self, output_directory: Path) -> Path:
        """Train the network and persist its mapped trajectories."""

        mlops_run = get_active_run()

        # Track the acquired source before processors transform it so lineage
        # remains tied to reusable input bytes.
        dataset = DatasetLoader.load(
            self.config.dataset.source,
            dt=float(self.config.dataset.resampling.dt),
            dataset_tracker=mlops_run,
            dataset_name="training-dataset",
            dataset_context="training",
            upload_dataset=(
                self.config.mlops.upload_dataset if self.config.mlops else False
            ),
        )

        simulation = self.config.simulation
        create_simulation_context(
            simulation.backend,
            simulation.usd_path,
            fixed_base=simulation.fixed_base,
        )

        dataset = DataProcessorRunner.run(dataset, self.config.input_features)
        dataset = DataProcessorRunner.run(dataset, self.config.output_features)
        mappings = {
            name: EntityMapping(inputs=mapping.inputs, outputs=mapping.outputs)
            for name, mapping in self.config.mapping.items()
        }
        sequences = self._map_features(
            dataset,
            input_features=list(self.config.input_features),
            output_features=list(self.config.output_features),
            mappings=mappings,
        )
        from anchor_bench.tasks.train.trainer import Trainer

        search_result = None
        if self.config.search is None:
            trainer = Trainer(
                self.config,
                sequences,
                checkpoint_directory=output_directory / "checkpoints",
            )
            result = trainer.train()
            validation_sequences = trainer.data_loaders.validation_sequences
            artifact_config = self.config
        else:
            search_result = self._search(sequences, output_directory)
            result = search_result.training_result
            validation_sequences = search_result.validation_sequences
            artifact_config = search_result.best_config

        from anchor_bench.tasks.train.data import create_validation_dataset
        from anchor_bench.tasks.train.validator_runner import ValidatorRunner

        validation_dataset = create_validation_dataset(
            dataset,
            validation_sequences,
        )

        validation_model = result.normalization.wrap(result.model)
        validation_results = ValidatorRunner.run(
            self.config.validation.validators,
            model=validation_model,
            validation_dataset=validation_dataset,
        )

        from anchor_bench.tasks.train.artifacts import save_training_artifacts

        save_training_artifacts(
            result,
            sequences,
            validation_results,
            artifact_config,
            output_directory,
            search_result=search_result,
        )
        self._write_trajectories(
            sequences,
            output_directory / TRAJECTORIES_FILENAME,
        )
        # Validator metrics follow the epoch or trial series without moving the
        # run-global step backwards; the completed directory is uploaded last.
        if mlops_run is not None:
            validation_metrics = {
                f"validation/{validation.name}/{name}": float(value)
                for validation in validation_results
                for name, value in validation.metrics.items()
            }
            if validation_metrics:
                step = (
                    self.config.search.trials
                    if self.config.search is not None
                    else self.config.training.epochs
                )
                mlops_run.log_metrics(validation_metrics, step=step)
            mlops_run.log_artifact(
                output_directory,
                collection="training-output",
                artifact_type="model",
            )
        return output_directory

    def _search(
        self,
        sequences: Sequence[MappedTrainingSequence],
        output_directory: Path,
    ) -> SearchResult:
        """Wrap repeated trainer executions in the configured Optuna loop."""

        search = self.config.search
        mlops_run = get_active_run()
        if search is None:
            raise RuntimeError("Training search configuration is missing.")
        if search.objective.metric != "validation_loss":
            raise ValueError(
                "Search currently supports only the 'validation_loss' objective."
            )

        study = optuna.create_study(
            direction=search.objective.direction,
            sampler=optuna.samplers.TPESampler(seed=self.config.training.seed),
        )
        trials = []
        best: (
            tuple[
                float,
                TrainingResult,
                TrainingConfig,
                tuple[MappedTrainingSequence, ...],
                Mapping[str, Any],
                int,
            ]
            | None
        ) = None

        def objective(trial: optuna.Trial) -> float:
            """Train and score one sampled configuration."""

            from anchor_bench.tasks.train.trainer import Trainer

            nonlocal best
            trial_config, parameters = self._trial_config(trial)
            # Every trial restarts its epoch count, so suppress nested epoch
            # curves and report one objective value at the run-global trial
            # number instead.
            trainer = Trainer(
                trial_config,
                sequences,
                checkpoint_directory=(
                    output_directory / "checkpoints" / f"trial-{trial.number:04d}"
                ),
                report_metrics=False,
                checkpoint_collection=(f"search-trial-{trial.number:04d}-checkpoints"),
            )
            result = trainer.train()
            value = self._validation_loss(result)
            if not isfinite(value):
                raise optuna.TrialPruned("Validation loss is not finite.")
            if mlops_run is not None:
                mlops_run.log_metrics(
                    {"search/validation_loss": value},
                    step=trial.number,
                )
            trials.append(
                SearchTrial(
                    number=trial.number,
                    parameters=parameters,
                    value=value,
                )
            )
            if best is None or self._is_better(value, best[0]):
                best = (
                    value,
                    result,
                    trial_config,
                    trainer.data_loaders.validation_sequences,
                    parameters,
                    trial.number,
                )
            return value

        study.optimize(objective, n_trials=search.trials)
        if best is None:
            raise RuntimeError("Hyperparameter search did not complete any trials.")

        value, result, config, held_out, parameters, trial_number = best
        return SearchResult(
            training_result=result,
            best_config=config,
            validation_sequences=held_out,
            objective_metric=search.objective.metric,
            objective_direction=search.objective.direction,
            best_trial=trial_number,
            best_value=value,
            best_parameters=parameters,
            trials=tuple(trials),
        )

    def _trial_config(
        self,
        trial: optuna.Trial,
    ) -> tuple[TrainingConfig, dict[str, Any]]:
        """Sample dotted YAML fields and revalidate the resulting config."""

        search = self.config.search
        if search is None:
            raise RuntimeError("Training search configuration is missing.")
        data = self.config.model_dump(mode="python", exclude_none=True)
        data.pop("search", None)
        parameters = {}
        for path, parameter in search.search_space.items():
            value = _suggest(trial, path, parameter)
            _set_path(data, path, value)
            parameters[path] = value
        return TrainingConfig.model_validate(data), parameters

    def _is_better(self, value: float, current: float) -> bool:
        """Compare objective values using the configured direction."""

        search = self.config.search
        if search is None:
            raise RuntimeError("Training search configuration is missing.")
        if search.objective.direction == "minimize":
            return value < current
        return value > current

    @staticmethod
    def _validation_loss(result: TrainingResult) -> float:
        """Return the final validation loss associated with the returned model."""

        if not result.history or result.history[-1].validation is None:
            raise ValueError(
                "Search objective 'validation_loss' requires validation windows."
            )
        return float(result.history[-1].validation)

    @staticmethod
    def _write_trajectories(
        sequences: Sequence[MappedTrainingSequence],
        path: Path,
    ) -> None:
        """Write mapped inputs and targets to one long-form Parquet artifact."""

        records = []
        for sequence_index, sequence in enumerate(sequences):
            for kind, values in (
                ("input", sequence.inputs),
                ("target", sequence.targets),
            ):
                index_names = [name for name in values.index.names if name is not None]
                if len(index_names) != values.index.nlevels:
                    raise ValueError(
                        "Mapped trajectory index levels must all have names."
                    )
                frame = values.reset_index().melt(
                    id_vars=index_names,
                    var_name="feature",
                    value_name="value",
                )
                frame.insert(0, "kind", kind)
                frame.insert(0, "mapping", sequence.name)
                frame.insert(0, "sequence", sequence_index)
                records.append(frame)

        if not records:
            raise ValueError("Training did not produce any mapped trajectories.")
        pd.concat(records, ignore_index=True).to_parquet(path, index=False)

    @staticmethod
    def _map_features(
        dataset: pd.DataFrame,
        *,
        input_features: Sequence[str],
        output_features: Sequence[str],
        mappings: Mapping[str, EntityMapping],
    ) -> list[MappedTrainingSequence]:
        """Prepare one ordered training sequence for every configured mapping."""

        TrainingTask._validate_map_features(dataset, mappings=mappings)
        resolved_mappings = []
        for name, mapping in mappings.items():
            input_columns, output_columns = TrainingTask._resolve_mapping_columns(
                dataset,
                input_features=input_features,
                output_features=output_features,
                mapping=mapping,
            )
            resolved_mappings.append((name, input_columns, output_columns))

        mapped_sequences = []
        for experiment, episode in dataset.groupby(level="experiment", sort=False):
            for name, input_columns, output_columns in resolved_mappings:
                inputs, targets = TrainingTask._select_mapped_columns(
                    episode,
                    input_columns=input_columns,
                    output_columns=output_columns,
                )
                mapped_sequences.append(
                    MappedTrainingSequence(
                        experiment=experiment,
                        name=name,
                        inputs=inputs,
                        targets=targets,
                    )
                )
        return mapped_sequences

    @staticmethod
    def _validate_map_features(
        dataset: pd.DataFrame,
        *,
        mappings: Mapping[str, EntityMapping],
    ) -> None:
        """Validate shared mapping shapes and recurrent sequence indexing."""

        if not mappings:
            raise ValueError("Training requires at least one entity mapping.")
        if "experiment" not in dataset.index.names:
            raise ValueError("Mapped training data requires an experiment index.")

        expected_shape = None
        for name, mapping in mappings.items():
            if not mapping.inputs or not mapping.outputs:
                raise ValueError(
                    f"Mapping {name!r} requires at least one input and output entity."
                )

            shape = (len(mapping.inputs), len(mapping.outputs))
            if expected_shape is None:
                expected_shape = shape
            elif shape != expected_shape:
                raise ValueError(
                    "Shared mappings must have identical input and output widths; "
                    f"mapping {name!r} has shape {shape[0]}->{shape[1]}, expected "
                    f"{expected_shape[0]}->{expected_shape[1]}."
                )

    @staticmethod
    def _resolve_mapping_columns(
        dataset: pd.DataFrame,
        *,
        input_features: Sequence[str],
        output_features: Sequence[str],
        mapping: EntityMapping,
    ) -> tuple[list[str], list[str]]:
        """Resolve and validate one mapping's ordered dataset columns."""

        input_columns = [
            f"{entity}/{feature}"
            for feature in input_features
            for entity in mapping.inputs
        ]
        output_columns = [
            f"{entity}/{feature}"
            for feature in output_features
            for entity in mapping.outputs
        ]
        required_columns = list(dict.fromkeys(input_columns + output_columns))
        missing_columns = [
            column for column in required_columns if column not in dataset.columns
        ]
        if missing_columns:
            names = ", ".join(missing_columns)
            raise ValueError(f"Dataset is missing mapped feature columns: {names}.")
        return input_columns, output_columns

    @staticmethod
    def _select_mapped_columns(
        dataset: pd.DataFrame,
        *,
        input_columns: Sequence[str],
        output_columns: Sequence[str],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Copy mapped inputs and targets from one experiment."""

        inputs = dataset.loc[:, input_columns].copy()
        targets = dataset.loc[:, output_columns].copy()
        return inputs, targets


def _suggest(
    trial: optuna.Trial,
    name: str,
    parameter: SearchParameter,
) -> Any:
    """Sample one typed search parameter through Optuna."""

    if isinstance(parameter, CategoricalSearchParameter):
        return trial.suggest_categorical(name, parameter.values)
    if isinstance(parameter, IntegerSearchParameter):
        return trial.suggest_int(
            name,
            parameter.min,
            parameter.max,
            step=parameter.step or 1,
            log=parameter.scale == "log",
        )
    if isinstance(parameter, FloatSearchParameter):
        return trial.suggest_float(
            name,
            float(parameter.min),
            float(parameter.max),
            step=None if parameter.step is None else float(parameter.step),
            log=parameter.scale == "log",
        )
    raise TypeError(f"Unsupported search parameter {name!r}.")


def _set_path(data: dict[str, Any], path: str, value: Any) -> None:
    """Set one existing dotted configuration path."""

    parts = path.split(".")
    current: Any = data
    for part in parts[:-1]:
        if not isinstance(current, dict) or part not in current:
            raise ValueError(f"Search parameter path {path!r} does not exist.")
        current = current[part]
    if not isinstance(current, dict) or parts[-1] not in current:
        raise ValueError(f"Search parameter path {path!r} does not exist.")
    current[parts[-1]] = value
