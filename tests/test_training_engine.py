from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from anchor_bench.training.episode import FeatureBlock, TrainingEpisode
from anchor_bench.training.metadata import (
    FrictionBaseline,
    PDBaseline,
    RuntimeHashes,
    RuntimeMapping,
)
from anchor_bench.training.schema import (
    ResolvedTrainingMapping,
    TrainingFeature,
    TrainingFit,
    TrainingNetwork,
)
from anchor_bench.training.trainer import (
    TrainingHyperParameters,
    TrainingJob,
    _normalization_stats,
    _prepare_episode,
    _runtime_metadata,
)


_HASHES = RuntimeHashes(
    profile_sha256="0" * 64,
    resolved_config_sha256="1" * 64,
    sysid_state_sha256="2" * 64,
    asset_sha256="3" * 64,
    source_sha256="4" * 64,
)


def _residual_fixture(tmp_path: Path) -> tuple[TrainingJob, TrainingEpisode]:
    mapping = ResolvedTrainingMapping(
        input_joints=("a", "b"),
        output_joints=("b", "a"),
    )
    position = np.array(
        [
            [[1.0, 10.0], [2.0, 20.0]],
            [[3.0, 30.0], [4.0, 40.0]],
            [[5.0, 50.0], [6.0, 60.0]],
        ],
        dtype=np.float32,
    )
    position_error = np.array(
        [[1.0, 2.0], [2.0, 1.0], [3.0, 0.0]],
        dtype=np.float32,
    )[:, :, None]
    velocity = np.array(
        [[0.5, -1.0], [1.0, 2.0], [-2.0, 0.5]],
        dtype=np.float32,
    )[:, :, None]
    episode = TrainingEpisode(
        timestamps=np.arange(3, dtype=np.float64),
        mapping=mapping,
        feature_blocks={
            "position": FeatureBlock(position, "input", ("value", "aux")),
            "position_error": FeatureBlock(position_error, "input", ("value",)),
            "velocity": FeatureBlock(velocity, "input", ("value",)),
        },
        measured_torque=np.array(
            [[10.0, 5.0], [4.0, 8.0], [1.0, 12.0]],
            dtype=np.float32,
        ),
    )
    runtime = RuntimeMapping(
        input_joints=("a", "b"),
        output_joints=("b", "a"),
        pd_baseline=PDBaseline(
            joint_names=("a", "b"),
            kp=(2.0, 3.0),
            kd=(0.5, 1.0),
        ),
        friction_baseline=FrictionBaseline(
            joint_names=("b", "a"),
            dry=(0.2, 0.4),
            viscous=(0.1, 0.2),
        ),
        effort_limits=(20.0, 30.0),
        clamping="residual_output",
    )
    job = TrainingJob(
        train_episodes=(episode,),
        val_episodes=(),
        output_dir=tmp_path,
        target="torque_residual",
        mappings=(mapping,),
        mapping_runtime=(runtime,),
        input_features=(
            TrainingFeature(name="position", domain="input", components=("value", "aux")),
            TrainingFeature(name="solver_pd", domain="input"),
            TrainingFeature(name="previous_torque", domain="output"),
        ),
        network=TrainingNetwork(hidden_size=12, num_layers=2, dropout=0.1),
        fit=TrainingFit(final_epochs=1, seed=7),
        sample_dt_s=0.01,
        hashes=_HASHES,
    )
    return job, episode


def test_residual_target_uses_vector_mapping_physics_and_output_order(tmp_path: Path) -> None:
    job, episode = _residual_fixture(tmp_path)

    features, target = _prepare_episode(job, episode)

    np.testing.assert_allclose(
        target,
        [
            [3.3, 2.75],
            [2.6, 3.9],
            [1.25, 5.8],
        ],
        rtol=1e-6,
    )
    # position is feature-major, then joint-major, then component-major;
    # solver_pd stays on the input axis and previous torque uses output order.
    np.testing.assert_allclose(
        features,
        [
            [1, 10, 2, 20, 1.75, 7.0, 0.0, 0.0],
            [3, 30, 4, 40, 3.5, 1.0, 3.3, 2.75],
            [5, 50, 6, 60, 7.0, -0.5, 2.6, 3.9],
        ],
        rtol=1e-6,
    )


def test_full_target_and_previous_torque_are_independent_of_gru_architecture(tmp_path: Path) -> None:
    residual_job, episode = _residual_fixture(tmp_path)
    runtime = residual_job.mapping_runtime[0].model_copy(
        update={
            "pd_baseline": None,
            "friction_baseline": None,
            "clamping": "total_output",
        }
    )
    job = TrainingJob(
        train_episodes=(episode,),
        val_episodes=(),
        output_dir=tmp_path,
        target="torque",
        mappings=residual_job.mappings,
        mapping_runtime=(runtime,),
        input_features=(
            TrainingFeature(name="position", domain="input", components=("value", "aux")),
            TrainingFeature(name="previous_torque", domain="output"),
        ),
        network=TrainingNetwork(hidden_size=3),
        fit=TrainingFit(final_epochs=1),
        sample_dt_s=0.01,
        hashes=_HASHES,
    )

    features, target = _prepare_episode(job, episode)

    np.testing.assert_array_equal(target, episode.measured_torque)
    np.testing.assert_array_equal(features[0, -2:], [0.0, 0.0])
    np.testing.assert_array_equal(features[1:, -2:], episode.measured_torque[:-1])


def test_normalization_pools_rows_and_rejects_any_zero_width(tmp_path: Path) -> None:
    job, _ = _residual_fixture(tmp_path)
    input_size = 8
    output_size = 2
    sequences = [
        (
            np.zeros((1, input_size), dtype=np.float32),
            np.zeros((1, output_size), dtype=np.float32),
        ),
        (
            np.full((3, input_size), 2.0, dtype=np.float32),
            np.full((3, output_size), 4.0, dtype=np.float32),
        ),
    ]

    normalization = _normalization_stats(job, sequences)

    assert normalization.inputs.mean["position"] == (1.5, 1.5, 1.5, 1.5)
    assert normalization.targets.mean["torque_residual"] == (3.0, 3.0)

    bad_inputs = np.array(sequences[1][0], copy=True)
    bad_inputs[:, 0] = 0.0
    with pytest.raises(ValueError, match="standard deviations must be non-zero"):
        _normalization_stats(
            job,
            [
                sequences[0],
                (bad_inputs, sequences[1][1]),
            ],
        )


def test_schema_v3_metadata_carries_dimensions_physics_and_hashes(tmp_path: Path) -> None:
    job, episode = _residual_fixture(tmp_path)
    features, target = _prepare_episode(job, episode)
    normalization = _normalization_stats(job, [(features, target)])

    metadata = _runtime_metadata(
        job,
        normalization,
        TrainingHyperParameters(hidden_size=12, num_layers=2, dropout=0.1),
    )

    assert metadata.schema_version == 3
    assert metadata.input_size == 8
    assert metadata.output_size == 2
    assert [feature.width for feature in metadata.input_features] == [4, 2, 2]
    assert metadata.flattening_order == ("feature", "joint", "component")
    assert metadata.target_columns == ("torque_residual",)
    assert metadata.mappings[0].pd_baseline is not None
    assert metadata.mappings[0].pd_baseline.kp == (2.0, 3.0)
    assert metadata.previous_torque_derivation is not None
    assert metadata.previous_torque_derivation.initialization == "physical_zero"
    assert metadata.hashes == _HASHES


def test_job_rejects_runtime_axis_mismatch_and_reserved_stats_metadata(tmp_path: Path) -> None:
    job, episode = _residual_fixture(tmp_path)
    wrong = job.mapping_runtime[0].model_copy(
        update={
            "input_joints": ("x", "y"),
            "output_joints": ("y", "x"),
            "pd_baseline": job.mapping_runtime[0].pd_baseline.model_copy(
                update={"joint_names": ("x", "y")}
            ),
            "friction_baseline": job.mapping_runtime[0].friction_baseline.model_copy(
                update={"joint_names": ("y", "x")}
            ),
        }
    )
    with pytest.raises(ValueError, match="axes must exactly match"):
        TrainingJob(
            train_episodes=(episode,),
            val_episodes=(),
            output_dir=tmp_path,
            target=job.target,
            mappings=job.mappings,
            mapping_runtime=(wrong,),
            input_features=job.input_features,
            network=job.network,
            fit=job.fit,
            sample_dt_s=job.sample_dt_s,
            hashes=job.hashes,
        )
    with pytest.raises(ValueError, match="reserved fields: normalization"):
        TrainingJob(
            train_episodes=(episode,),
            val_episodes=(),
            output_dir=tmp_path,
            target=job.target,
            mappings=job.mappings,
            mapping_runtime=job.mapping_runtime,
            input_features=job.input_features,
            network=job.network,
            fit=job.fit,
            sample_dt_s=job.sample_dt_s,
            hashes=job.hashes,
            stats_metadata={"normalization": {}},
        )


def test_export_rejects_metadata_free_checkpoint_before_model_construction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import anchor_bench.training.export as exporter

    class FakeTorch:
        @staticmethod
        def load(path: Path, map_location: str) -> dict[str, object]:
            return {"input_size": 1, "output_size": 1, "model_state_dict": {}}

    monkeypatch.setattr(exporter, "_import_torch", lambda: FakeTorch)
    with pytest.raises(ValueError, match="missing required schema-v3 runtime_metadata"):
        exporter.export_gru_checkpoint(tmp_path / "best.pt")


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="TorchScript round trip requires torch")
def test_export_embeds_schema_v3_metadata_and_uses_fixed_default_path(tmp_path: Path) -> None:
    import torch

    from anchor_bench.training.export import (
        build_runtime_gru,
        export_gru_checkpoint,
        serialize_runtime_metadata,
    )
    from anchor_bench.training.metadata import RuntimeGRUMetadata

    job, episode = _residual_fixture(tmp_path)
    features, target = _prepare_episode(job, episode)
    normalization = _normalization_stats(job, [(features, target)])
    hparams = TrainingHyperParameters(hidden_size=4, num_layers=1, dropout=0.0)
    metadata = _runtime_metadata(job, normalization, hparams)
    checkpoint_path = tmp_path / "checkpoints" / "best.pt"
    checkpoint_path.parent.mkdir()
    model = build_runtime_gru(
        input_size=metadata.input_size,
        output_size=metadata.output_size,
        hidden_size=4,
        num_layers=1,
        dropout=0.0,
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_size": metadata.input_size,
            "output_size": metadata.output_size,
            "runtime_metadata": metadata.model_dump(mode="json", exclude_none=True),
            "hyperparameters": hparams.__dict__,
        },
        checkpoint_path,
    )

    output = export_gru_checkpoint(checkpoint_path)

    assert output == tmp_path / "network.pt"
    extra = {"metadata.json": ""}
    loaded = torch.jit.load(str(output), _extra_files=extra)
    embedded = RuntimeGRUMetadata.model_validate(json.loads(extra["metadata.json"]))
    assert embedded == metadata
    assert extra["metadata.json"].decode() == serialize_runtime_metadata(metadata)
    prediction, hidden = loaded(
        torch.zeros(2, 1, metadata.input_size),
        torch.zeros(1, 2, 4),
    )
    assert tuple(prediction.shape) == (2, 1, metadata.output_size)
    assert tuple(hidden.shape) == (1, 2, 4)
