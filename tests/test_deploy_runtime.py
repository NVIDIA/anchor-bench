from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest
import yaml

from anchor_bench.core.hashing import sha256_file, sha256_hex
from anchor_bench.deploy.schema import FrozenDeployState
from anchor_bench.training.export import serialize_runtime_metadata
from anchor_bench.training.metadata import (
    ArmatureValues,
    DynamicBiasDerivation,
    FrictionBaseline,
    GRURecurrentDimensions,
    NormalizationValues,
    PDBaseline,
    PreviousTorqueDerivation,
    RuntimeFeature,
    RuntimeGRUMetadata,
    RuntimeHashes,
    RuntimeMapping,
    RuntimeNormalization,
)


class _Array:
    """Small NumPy-backed stand-in for a Warp model or constructor array."""

    def __init__(self, values, *, dtype="float32", device="cpu") -> None:
        """Store copied values and the runtime attributes used by tests."""

        self.values = np.asarray(values).copy()
        self.dtype = dtype
        self.device = device
        self.shape = self.values.shape

    def numpy(self):
        """Return the current host representation."""

        return self.values.copy()

    def assign(self, values) -> None:
        """Replace values while preserving the fake array identity."""

        self.values = np.asarray(values).copy()
        self.shape = self.values.shape

    def __len__(self) -> int:
        """Return the first-axis length."""

        return len(self.values)


class _Warp:
    """Fake Warp namespace recording dtype and device construction."""

    float32 = "float32"
    uint32 = "uint32"

    @staticmethod
    def array(values, *, dtype, device):
        """Construct one fake device array."""

        return _Array(values, dtype=dtype, device=device)


class _JointTargetMode:
    """Pinned JointTargetMode behavior needed by frozen gain installation."""

    NONE = 0
    POSITION = 1
    VELOCITY = 2
    POSITION_VELOCITY = 3
    EFFORT = 4

    @staticmethod
    def from_gains(kp, kd, *, has_drive=False):
        """Mirror Newton's default gain-to-target-mode inference."""

        if not has_drive:
            return _JointTargetMode.NONE
        if kp != 0.0:
            return _JointTargetMode.POSITION
        if kd != 0.0:
            return _JointTargetMode.VELOCITY
        return _JointTargetMode.EFFORT


class _ControllerPD:
    """Capture explicit PD gains supplied by the deployment builder."""

    def __init__(self, *, kp, kd) -> None:
        """Store gain arrays."""

        self.kp = kp
        self.kd = kd


class _Clamp:
    """Capture one total-effort clamp array."""

    def __init__(self, *, max_effort) -> None:
        """Store the effort limits."""

        self.max_effort = max_effort


class _PDActuator:
    """Newton-Actuator-shaped fake for explicit PD construction tests."""

    def __init__(self, **kwargs) -> None:
        """Store constructor arguments and subsequent step calls."""

        self.kwargs = kwargs
        self.steps = []

    def state(self):
        """Return no state for a stateless PD controller."""

        return None

    def step(self, *args, **kwargs) -> None:
        """Record one deployment step."""

        self.steps.append((args, kwargs))
        control = args[1]
        efforts = control.joint_f.numpy()
        efforts[self.kwargs["effort_indices"].numpy()] += 1.0
        control.joint_f.assign(efforts)


class _RecurrentState:
    """Fake recurrent state recording reset boundaries."""

    def __init__(self) -> None:
        """Initialize an empty reset log."""

        self.resets = []

    def reset(self, mask=None) -> None:
        """Record one full or per-environment reset."""

        self.resets.append(mask)


class _NeuralActuator:
    """Stateful mapping actuator returned through the explicit factory."""

    def __init__(self) -> None:
        """Initialize step and allocated-state logs."""

        self.steps = []
        self.states = []
        self.effort_indices = ()

    def is_stateful(self) -> bool:
        """Declare required recurrent statefulness."""

        return True

    def state(self):
        """Allocate one independent recurrent buffer."""

        value = _RecurrentState()
        self.states.append(value)
        return value

    def step(self, *args, **kwargs) -> None:
        """Record one mapped inference step."""

        self.steps.append((args, kwargs))
        control = args[1]
        efforts = control.joint_f.numpy()
        efforts[list(self.effort_indices)] += 2.0
        control.joint_f.assign(efforts)


class _Model:
    """Two-world one-DOF robot model with one intentionally omitted joint."""

    def __init__(self) -> None:
        """Build world-repeated labels, starts, and mutable physical arrays."""

        self.device = "cpu"
        self.use_coord_layout_targets = False
        self.joint_label = [
            "env0/context",
            "env0/elbow",
            "env0/shoulder",
            "env0/omitted",
            "env1/context",
            "env1/elbow",
            "env1/shoulder",
            "env1/omitted",
        ]
        starts = np.arange(9, dtype=np.int32)
        self.joint_q_start = _Array(starts, dtype="int32")
        self.joint_qd_start = _Array(starts, dtype="int32")
        self.joint_world = _Array([0, 0, 0, 0, 1, 1, 1, 1], dtype="int32")
        self.joint_target_ke = _Array([100.0] * 8)
        self.joint_target_kd = _Array([10.0] * 8)
        self.joint_target_mode = _Array([9] * 8, dtype="int32")
        self.joint_friction = _Array([0.7] * 8)
        self.joint_damping = _Array([0.8] * 8)
        self.joint_armature = _Array([0.9] * 8)


class _Solver:
    """Fake MuJoCo solver supporting notifications and reusable dynamic bias."""

    def __init__(self, model) -> None:
        """Bind a model and initialize call logs."""

        self.model = model
        self.notifications = []
        self.bias_calls = []

    def notify_model_changed(self, flags) -> None:
        """Record the requested pinned model-change flag."""

        self.notifications.append(flags)

    def eval_dynamic_bias(self, state, out=None):
        """Return and then reuse one fake full-DOF dynamic-bias array."""

        result = out if out is not None else object()
        self.bias_calls.append((state, out, result))
        return result


class _Control:
    """Fake Newton control carrying the scatter-add effort array."""

    def __init__(self, efforts=None) -> None:
        """Initialize full-model efforts, including omitted joint values."""

        self.joint_f = _Array(efforts if efforts is not None else [0.0] * 8)


def _clear_owned_efforts(efforts, indices, device) -> None:
    """Clear only injected owned indices while preserving every omission."""

    values = efforts.numpy()
    values[indices.numpy()] = 0.0
    efforts.assign(values)


def _bindings():
    """Return host-safe constructors matching the pinned Newton surface."""

    from anchor_bench.deploy.runtime import NewtonActuatorBindings

    return NewtonActuatorBindings(
        warp=_Warp,
        actuator_type=_PDActuator,
        controller_pd_type=_ControllerPD,
        clamping_max_effort_type=_Clamp,
        joint_target_mode_type=_JointTargetMode,
        joint_dof_properties_flag="JOINT_DOF_PROPERTIES",
        clear_owned_efforts=_clear_owned_efforts,
    )


def _metadata(*, dynamic_bias: bool = False) -> RuntimeGRUMetadata:
    """Build one residual schema-v3 mapping accepted by deployment."""

    features = [
        RuntimeFeature(
            name="position_error",
            domain="input",
            components=("value",),
            width=2,
        ),
        RuntimeFeature(
            name="previous_torque",
            domain="output",
            components=("value",),
            width=1,
        ),
    ]
    input_mean = {"position_error": (0.0, 0.0), "previous_torque": 0.0}
    input_std = {"position_error": (1.0, 1.0), "previous_torque": 1.0}
    if dynamic_bias:
        features.append(
            RuntimeFeature(
                name="dynamic_bias",
                domain="input",
                components=("value",),
                width=2,
            )
        )
        input_mean["dynamic_bias"] = (0.0, 0.0)
        input_std["dynamic_bias"] = (1.0, 1.0)
    mapping = RuntimeMapping(
        input_joints=("context", "elbow"),
        output_joints=("elbow",),
        pd_baseline=PDBaseline(
            joint_names=("context", "elbow"),
            kp=(11.0, 22.0),
            kd=(0.5, 0.75),
        ),
        friction_baseline=FrictionBaseline(
            joint_names=("elbow",),
            dry=0.12,
            viscous=0.03,
        ),
        armature=ArmatureValues(joint_names=("elbow",), values=0.02),
        effort_limits=25.0,
        clamping="residual_output",
    )
    return RuntimeGRUMetadata(
        input_size=sum(feature.width for feature in features),
        output_size=1,
        recurrent=GRURecurrentDimensions(hidden_size=8, num_layers=1),
        input_features=tuple(features),
        mappings=(mapping,),
        target_columns=("torque_residual",),
        sample_dt_s=0.002,
        normalization=RuntimeNormalization(
            inputs=NormalizationValues(mean=input_mean, std=input_std),
            targets=NormalizationValues(
                mean={"torque_residual": 0.0},
                std={"torque_residual": 1.0},
            ),
        ),
        previous_torque_derivation=PreviousTorqueDerivation(
            value_space="residual_torque"
        ),
        dynamic_bias_derivation=(
            DynamicBiasDerivation(
                source="newton.solvers.SolverMuJoCo.eval_dynamic_bias",
                provenance={"test": True},
            )
            if dynamic_bias
            else None
        ),
        hashes=RuntimeHashes(
            profile_sha256="a" * 64,
            resolved_config_sha256="b" * 64,
            sysid_state_sha256="c" * 64,
            asset_sha256="d" * 64,
            source_sha256="e" * 64,
        ),
    )


def _frozen(tmp_path: Path, *, dynamic_bias: bool = False) -> tuple[FrozenDeployState, Path]:
    """Write an authenticated archive and return its matching frozen state."""

    metadata = _metadata(dynamic_bias=dynamic_bias)
    raw_metadata = serialize_runtime_metadata(metadata).encode("utf-8")
    artifact = tmp_path / "outputs/train/demo/arm/seed-42" / ("f" * 64) / "network.pt"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(artifact, "w") as archive:
        archive.writestr("archive/extra/metadata.json", raw_metadata)
    payload = {
        "schema_version": 1,
        "simulation": {
            "physics_freq": 500,
            "control_freq": 500,
            "render_freq": 500,
            "integrator": "implicitfast",
            "solver_iterations": 10,
            "line_search_iterations": 10,
        },
        "ownership": {"elbow": "arm_network", "shoulder": "shoulder_pd"},
        "composition": {
            "neural": [
                {
                    "name": "arm_network",
                    "model": "arm_gru",
                    "accepted_model_revision": 2,
                    "artifact": {
                        "path": artifact.relative_to(tmp_path).as_posix(),
                        "sha256": sha256_file(artifact),
                        "metadata_sha256": sha256_hex(raw_metadata),
                    },
                    "target": "torque_residual",
                    "sample_dt_s": 0.002,
                    "mappings": [
                        {
                            "index": 0,
                            "input_joints": ["context", "elbow"],
                            "output_joints": ["elbow"],
                            "pd_baseline": {
                                "kp": {"context": 11.0, "elbow": 22.0},
                                "kd": {"context": 0.5, "elbow": 0.75},
                                "velocity_target": "zero",
                            },
                            "friction_baseline": {
                                "dry": {"elbow": 0.12},
                                "viscous": {"elbow": 0.03},
                            },
                            "armature": {"elbow": 0.02},
                            "effort_limits": {"elbow": 25.0},
                            "clamping": "residual_output",
                        }
                    ],
                }
            ],
            "pd": [
                {
                    "name": "shoulder_pd",
                    "joints": ["shoulder"],
                    "kp": {"shoulder": 30.0},
                    "kd": {"shoulder": 0.8},
                    "dry_friction": {"shoulder": 0.1},
                    "viscous_friction": {"shoulder": 0.02},
                    "armature": {"shoulder": 0.006},
                    "effort_limits": {"shoulder": 40.0},
                }
            ],
        },
        "source": {
            "profile": "default",
            "profile_sha256": "1" * 64,
            "sysid_revision": 4,
            "sysid_state_sha256": "2" * 64,
            "train_revision": 5,
            "train_state_sha256": "3" * 64,
        },
    }
    return FrozenDeployState.from_dict(payload, "deploy/default.yaml"), artifact


def test_builds_explicit_pd_and_mapped_neural_without_touching_omissions(
    tmp_path: Path,
) -> None:
    """Map two worlds, freeze physical values, clamps, and recurrent identity."""

    from anchor_bench.deploy.runtime import build_deployment_runtime

    frozen, _artifact = _frozen(tmp_path)
    model = _Model()
    solver = _Solver(model)
    requests = []
    neural = _NeuralActuator()

    def neural_factory(request):
        """Capture the authenticated mapping contract."""

        requests.append(request)
        neural.effort_indices = request.output_axes.flatten_velocities()
        assert model.joint_target_ke.numpy()[1] == 22.0
        assert model.joint_target_kd.numpy()[1] == 0.75
        return neural

    runtime = build_deployment_runtime(
        frozen,
        model,
        repository_root=tmp_path,
        solver=solver,
        neural_factory=neural_factory,
        bindings=_bindings(),
    )

    assert runtime.worlds == (0, 1)
    assert [(entry.kind, entry.name) for entry in runtime.entries] == [
        ("pd", "shoulder_pd"),
        ("neural", "arm_network"),
    ]
    pd = runtime.entries[0].actuator
    assert pd.kwargs["indices"].numpy().tolist() == [2, 6]
    assert pd.kwargs["pos_indices"].numpy().tolist() == [2, 6]
    assert pd.kwargs["target_pos_indices"].numpy().tolist() == [2, 6]
    assert pd.kwargs["controller"].kp.numpy().tolist() == [30.0, 30.0]
    assert pd.kwargs["controller"].kd.numpy().tolist() == [0.8, 0.8]
    assert pd.kwargs["clamping"][0].max_effort.numpy().tolist() == [40.0, 40.0]
    assert pd.kwargs["control_feedforward_attr"] is None

    request = requests[0]
    assert request.metadata.schema_version == 3
    assert request.input_axes.position_indices == ((0, 1), (4, 5))
    assert request.output_axes.velocity_indices == ((1,), (5,))
    assert request.external_feedforward_allowed is False
    assert request.external_delay_steps == 0
    assert request.solver_gains_disabled is False
    assert request.factory_output_semantics == "clamped_residual"

    np.testing.assert_allclose(model.joint_target_ke.numpy(), [100, 22, 0, 100, 100, 22, 0, 100])
    np.testing.assert_allclose(model.joint_target_kd.numpy(), [10, 0.75, 0, 10, 10, 0.75, 0, 10])
    np.testing.assert_allclose(model.joint_target_mode.numpy(), [9, 1, 4, 9, 9, 1, 4, 9])
    np.testing.assert_allclose(model.joint_friction.numpy(), [0.7, 0.12, 0.1, 0.7, 0.7, 0.12, 0.1, 0.7])
    np.testing.assert_allclose(model.joint_damping.numpy(), [0.8, 0.03, 0.02, 0.8, 0.8, 0.03, 0.02, 0.8])
    np.testing.assert_allclose(model.joint_armature.numpy(), [0.9, 0.02, 0.006, 0.9, 0.9, 0.02, 0.006, 0.9])
    assert solver.notifications == ["JOINT_DOF_PROPERTIES"]

    recurrent = runtime.state()
    recurrent.reset([True, False])
    assert [state.resets for state in neural.states] == [
        [[True, False]],
        [[True, False]],
    ]
    current_before = recurrent.current[1]
    control = _Control([7.0, 8.0, 9.0, 91.0, 17.0, 18.0, 19.0, 92.0])
    runtime.step(object(), control, recurrent, dt=0.002)
    assert recurrent.next[1] is current_before
    assert neural.steps[0][1]["dynamic_bias"] is None
    np.testing.assert_allclose(
        control.joint_f.numpy(),
        [7.0, 2.0, 1.0, 91.0, 17.0, 2.0, 1.0, 92.0],
    )
    runtime.step(object(), control, recurrent, dt=0.002)
    np.testing.assert_allclose(
        control.joint_f.numpy(),
        [7.0, 2.0, 1.0, 91.0, 17.0, 2.0, 1.0, 92.0],
    )
    with pytest.raises(ValueError, match="does not match frozen control timestep"):
        runtime.step(object(), control, recurrent, dt=0.004)


def test_dynamic_bias_requires_solver_and_reuses_eval_output(tmp_path: Path) -> None:
    """Evaluate pinned MuJoCo dynamic bias once per control step and reuse output."""

    from anchor_bench.deploy.runtime import build_deployment_runtime

    frozen, _artifact = _frozen(tmp_path, dynamic_bias=True)
    with pytest.raises(ValueError, match="SolverMuJoCo.eval_dynamic_bias"):
        build_deployment_runtime(
            frozen,
            _Model(),
            repository_root=tmp_path,
            neural_factory=lambda request: _NeuralActuator(),
            bindings=_bindings(),
        )

    model = _Model()
    solver = _Solver(model)
    neural = _NeuralActuator()
    runtime = build_deployment_runtime(
        frozen,
        model,
        repository_root=tmp_path,
        solver=solver,
        neural_factory=lambda request: neural,
        bindings=_bindings(),
    )
    recurrent = runtime.state()
    sim_state = object()
    control = _Control()
    runtime.step(sim_state, control, recurrent)
    first = solver.bias_calls[0][2]
    runtime.step(sim_state, control, recurrent)

    assert solver.bias_calls[0][1] is None
    assert solver.bias_calls[1][1] is first
    assert neural.steps[0][1]["dynamic_bias"] is first
    assert neural.steps[1][1]["dynamic_bias"] is first


def test_full_torque_neural_outputs_use_direct_effort_mode(tmp_path: Path) -> None:
    """Keep full-network outputs out of solver PD while preserving USD friction."""

    from anchor_bench.deploy import runtime

    frozen, _artifact = _frozen(tmp_path)
    payload = frozen.to_dict()
    neural = payload["composition"]["neural"][0]
    neural["target"] = "torque"
    mapping = neural["mappings"][0]
    mapping["pd_baseline"] = None
    mapping["friction_baseline"] = None
    mapping["clamping"] = "total_output"
    full_torque = FrozenDeployState.from_dict(payload, "deploy/default.yaml")
    model = _Model()
    instances = runtime._model_joint_instances(
        model,
        ("context", "elbow", "shoulder"),
    )
    updates = runtime._model_updates(full_torque, instances, _JointTargetMode)
    runtime._apply_model_updates(model, updates)

    np.testing.assert_allclose(
        model.joint_target_ke.numpy(),
        [100, 0, 0, 100, 100, 0, 0, 100],
    )
    np.testing.assert_allclose(
        model.joint_target_kd.numpy(),
        [10, 0, 0, 10, 10, 0, 0, 10],
    )
    np.testing.assert_allclose(
        model.joint_target_mode.numpy(),
        [9, 4, 4, 9, 9, 4, 4, 9],
    )
    np.testing.assert_allclose(
        model.joint_friction.numpy(),
        [0.7, 0.7, 0.1, 0.7, 0.7, 0.7, 0.1, 0.7],
    )
    np.testing.assert_allclose(
        model.joint_damping.numpy(),
        [0.8, 0.8, 0.02, 0.8, 0.8, 0.8, 0.02, 0.8],
    )


def test_neural_requires_explicit_schema_v3_factory_before_runtime_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail clearly instead of routing schema-v3 into Newton's schema-v1 GRU."""

    from anchor_bench.deploy import runtime

    frozen, _artifact = _frozen(tmp_path)
    monkeypatch.setattr(
        runtime,
        "_load_newton_bindings",
        lambda: (_ for _ in ()).throw(AssertionError("heavy runtime imported")),
    )
    with pytest.raises(ValueError, match="incompatible schema-version-1 metadata"):
        runtime.build_deployment_runtime(
            frozen,
            _Model(),
            repository_root=tmp_path,
        )


def test_rejects_artifact_and_frozen_mapping_drift(tmp_path: Path) -> None:
    """Authenticate both archive bytes and mapping-specific metadata values."""

    from anchor_bench.deploy.runtime import build_deployment_runtime

    frozen, artifact = _frozen(tmp_path)
    artifact.write_bytes(artifact.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        build_deployment_runtime(
            frozen,
            _Model(),
            repository_root=tmp_path,
            neural_factory=lambda request: _NeuralActuator(),
            bindings=_bindings(),
        )

    frozen, _artifact = _frozen(tmp_path)
    payload = frozen.to_dict()
    payload["composition"]["neural"][0]["mappings"][0]["effort_limits"]["elbow"] = 24.0
    drifted = FrozenDeployState.from_dict(payload, "deploy/default.yaml")
    with pytest.raises(ValueError, match="no longer matches frozen deployment"):
        build_deployment_runtime(
            drifted,
            _Model(),
            repository_root=tmp_path,
            neural_factory=lambda request: _NeuralActuator(),
            bindings=_bindings(),
        )


def test_default_loader_ignores_mutable_upstream_and_asset_mapping_fails(
    tmp_path: Path,
) -> None:
    """Load only deploy/default.yaml and reject a model missing an explicit input."""

    from anchor_bench.deploy.runtime import (
        build_deployment_runtime,
        load_frozen_deploy_state,
    )

    frozen, _artifact = _frozen(tmp_path)
    deploy_path = tmp_path / "configs/robots/demo/deploy/default.yaml"
    deploy_path.parent.mkdir(parents=True)
    deploy_path.write_text(
        yaml.safe_dump(frozen.to_dict(), sort_keys=False),
        encoding="utf-8",
    )
    for phase in ("sysid", "train"):
        path = tmp_path / f"configs/robots/demo/{phase}/default.yaml"
        path.parent.mkdir(parents=True)
        path.write_text("not: [valid", encoding="utf-8")
    loaded = load_frozen_deploy_state(deploy_path)
    assert loaded == frozen

    model = _Model()
    model.joint_label[0] = "env0/missing_context"
    model.joint_label[4] = "env1/missing_context"
    with pytest.raises(ValueError, match="missing deployment joints: context"):
        build_deployment_runtime(
            loaded,
            model,
            repository_root=tmp_path,
            neural_factory=lambda request: _NeuralActuator(),
            bindings=_bindings(),
        )
