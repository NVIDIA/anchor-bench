"""Schema-v3 mapped GRU execution for frozen Newton deployments.

Torch and Warp are imported only when a neural deployment is constructed.
Each instance owns one physical mapping and evaluates all replicated Newton
worlds as one recurrent batch.
"""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from anchor_bench.training.metadata import StatValue

if TYPE_CHECKING:
    from anchor_bench.deploy.runtime import MappedNeuralRequest


_SUPPORTED_FEATURES = frozenset(
    {
        "position",
        "position_error",
        "velocity",
        "solver_pd",
        "dynamic_bias",
        "previous_torque",
    }
)


def _stat_vector(value: StatValue, width: int, *, label: str) -> tuple[float, ...]:
    """Expand one validated scalar or vector statistic to its exact width."""

    values = value if isinstance(value, tuple) else (value,)
    if len(values) != width:
        raise ValueError(f"{label} must contain {width} values")
    result = tuple(float(item) for item in values)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{label} must contain finite values")
    return result


def validate_mapped_gru_request(request: MappedNeuralRequest) -> None:
    """Validate the host-safe schema-v3 contract for one physical mapping."""

    metadata = request.metadata
    if metadata.schema_version != 3:
        raise ValueError("mapped GRU deployment requires schema-version-3 metadata")
    if metadata.target_columns != (request.actuator.target,):
        raise ValueError("mapped GRU target semantics do not match frozen deployment")
    if not math.isclose(
        metadata.sample_dt_s,
        request.control_dt_s,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("mapped GRU timestep does not match frozen deployment")
    if request.external_feedforward_allowed or request.external_delay_steps != 0:
        raise ValueError("mapped GRU deployment does not permit external feedforward or delay")
    if request.mapping.index >= len(metadata.mappings):
        raise ValueError("mapped GRU metadata does not contain the frozen mapping index")

    mapping = metadata.mappings[request.mapping.index]
    if mapping.input_joints != request.input_axes.joints:
        raise ValueError("mapped GRU input axes do not match metadata")
    if mapping.output_joints != request.output_axes.joints:
        raise ValueError("mapped GRU output axes do not match metadata")
    if request.input_axes.worlds != request.output_axes.worlds:
        raise ValueError("mapped GRU input and output worlds must match")
    if not request.input_axes.worlds:
        raise ValueError("mapped GRU requires at least one Newton model world")
    if metadata.output_size != len(request.output_axes.joints):
        raise ValueError("mapped GRU output_size does not match its physical output axis")

    feature_width = 0
    for feature in metadata.input_features:
        if feature.name not in _SUPPORTED_FEATURES:
            raise ValueError(f"unsupported mapped GRU feature {feature.name!r}")
        if feature.components != ("value",):
            raise ValueError(
                f"mapped GRU feature {feature.name!r} has unsupported components "
                f"{feature.components!r}; deployment currently provides only 'value'"
            )
        feature_width += feature.width
    if feature_width != metadata.input_size:
        raise ValueError("mapped GRU feature widths do not match input_size")


def _load_runtime_modules() -> tuple[Any, Any]:
    """Import Torch and Warp only when production neural execution is requested."""

    try:
        torch = importlib.import_module("torch")
    except ImportError as error:  # pragma: no cover - full runtime dependency
        raise RuntimeError("schema-v3 neural deployment requires PyTorch") from error
    try:
        warp = importlib.import_module("warp")
    except ImportError as error:  # pragma: no cover - full runtime dependency
        raise RuntimeError("schema-v3 neural deployment requires Warp") from error
    return torch, warp


def _torch_device(model_device: Any, torch: Any) -> Any:
    """Translate one Warp-like device into the matching Torch device."""

    if isinstance(model_device, torch.device):
        return model_device
    if bool(getattr(model_device, "is_cuda", False)):
        return torch.device(f"cuda:{int(model_device.ordinal)}")
    if isinstance(model_device, str):
        return torch.device(model_device)
    return torch.device("cpu")


def _torch_view(value: Any, *, torch: Any, warp: Any, label: str) -> Any:
    """Return a zero-copy Torch view of a Torch or Warp runtime array."""

    if isinstance(value, torch.Tensor):
        return value
    try:
        result = warp.to_torch(value)
    except Exception as error:
        raise TypeError(f"{label} must be a Torch tensor or Warp array") from error
    if not isinstance(result, torch.Tensor):
        raise TypeError(f"{label} did not produce a Torch tensor view")
    return result


def _mask_tensor(mask: Any, *, reference: Any, torch: Any, warp: Any) -> Any:
    """Convert an environment reset mask without copying device-resident masks."""

    if isinstance(mask, torch.Tensor):
        result = mask
    else:
        try:
            result = warp.to_torch(mask)
        except Exception:
            result = torch.as_tensor(mask, device=reference.device)
    return result.to(device=reference.device, dtype=torch.bool).reshape(-1)


@dataclass
class MappedGRUState:
    """Double-bufferable hidden state and previous raw physical network torque."""

    hidden: Any
    previous_torque: Any
    _torch: Any
    _warp: Any

    def reset(self, mask: Any | None = None) -> None:
        """Restore physical-zero state for all or selected Newton worlds."""

        if mask is None:
            self.hidden.zero_()
            self.previous_torque.zero_()
            return
        selected = _mask_tensor(
            mask,
            reference=self.previous_torque,
            torch=self._torch,
            warp=self._warp,
        )
        world_count = int(self.previous_torque.shape[0])
        if selected.numel() != world_count:
            raise ValueError(f"mapped GRU reset mask must contain {world_count} environment values")
        self.hidden[:, selected, :] = 0.0
        self.previous_torque[selected, :] = 0.0


class MappedGRUActuator:
    """One schema-v3 GRU application over every world of one physical mapping."""

    def __init__(self, request: MappedNeuralRequest) -> None:
        """Load the authenticated archive and precompute mapped tensor constants."""

        validate_mapped_gru_request(request)
        torch, warp = _load_runtime_modules()
        self._request = request
        self._torch = torch
        self._warp = warp
        self._device = _torch_device(request.model.device, torch)
        try:
            self._network = torch.jit.load(
                str(request.artifact_path),
                map_location=self._device,
            ).eval()
        except Exception as error:  # pragma: no cover - archive/runtime specific
            raise RuntimeError(f"cannot load schema-v3 GRU archive {request.artifact_path}") from error

        self._world_count = len(request.input_axes.worlds)
        self._input_width = len(request.input_axes.joints)
        self._output_width = len(request.output_axes.joints)
        self._position_indices = self._indices(request.input_axes.position_indices)
        self._velocity_indices = self._indices(request.input_axes.velocity_indices)
        self._target_position_indices = self._indices(request.input_axes.target_position_indices)
        self._output_indices = self._indices(request.output_axes.velocity_indices)

        metadata = request.metadata
        input_mean: list[float] = []
        input_std: list[float] = []
        for feature in metadata.input_features:
            input_mean.extend(
                _stat_vector(
                    metadata.normalization.inputs.mean[feature.name],
                    feature.width,
                    label=f"normalization.inputs.mean.{feature.name}",
                )
            )
            input_std.extend(
                _stat_vector(
                    metadata.normalization.inputs.std[feature.name],
                    feature.width,
                    label=f"normalization.inputs.std.{feature.name}",
                )
            )
        self._input_mean = self._constant(input_mean)
        self._input_std = self._constant(input_std)
        target = metadata.target_columns[0]
        self._target_mean = self._constant(
            _stat_vector(
                metadata.normalization.targets.mean[target],
                self._output_width,
                label=f"normalization.targets.mean.{target}",
            )
        )
        self._target_std = self._constant(
            _stat_vector(
                metadata.normalization.targets.std[target],
                self._output_width,
                label=f"normalization.targets.std.{target}",
            )
        )
        self._effort_limits = self._constant(
            tuple(request.mapping.effort_limits[joint] for joint in request.output_axes.joints)
        )

        mapping = metadata.mappings[request.mapping.index]
        if mapping.pd_baseline is None:
            self._pd_kp = self._constant((0.0,) * self._input_width)
            self._pd_kd = self._constant((0.0,) * self._input_width)
        else:
            self._pd_kp = self._constant(
                _stat_vector(
                    mapping.pd_baseline.kp,
                    self._input_width,
                    label="pd_baseline.kp",
                )
            )
            self._pd_kd = self._constant(
                _stat_vector(
                    mapping.pd_baseline.kd,
                    self._input_width,
                    label="pd_baseline.kd",
                )
            )

    def _constant(self, values: Any) -> Any:
        """Create one immutable float32-shaped Torch runtime constant."""

        return self._torch.tensor(values, dtype=self._torch.float32, device=self._device)

    def _indices(self, rows: Any) -> Any:
        """Create one world-major int64 index tensor."""

        return self._torch.tensor(rows, dtype=self._torch.long, device=self._device)

    def _runtime_tensor(self, value: Any, *, label: str) -> Any:
        """Require one same-device, one-dimensional float32 runtime tensor."""

        tensor = _torch_view(value, torch=self._torch, warp=self._warp, label=label)
        if tensor.device != self._device:
            raise ValueError(f"{label} is on {tensor.device}, expected {self._device}")
        if tensor.dtype != self._torch.float32:
            raise TypeError(f"{label} must use float32")
        if tensor.ndim != 1:
            raise ValueError(f"{label} must be one-dimensional")
        return tensor

    def is_stateful(self) -> bool:
        """Declare independent hidden and autoregressive mapping state."""

        return True

    def is_graphable(self) -> bool:
        """Declare that TorchScript execution is not Warp-graph capture safe."""

        return False

    def state(self) -> MappedGRUState:
        """Allocate one physical-zero recurrent buffer for this mapping."""

        recurrent = self._request.metadata.recurrent
        return MappedGRUState(
            hidden=self._torch.zeros(
                recurrent.num_layers,
                self._world_count,
                recurrent.hidden_size,
                dtype=self._torch.float32,
                device=self._device,
            ),
            previous_torque=self._torch.zeros(
                self._world_count,
                self._output_width,
                dtype=self._torch.float32,
                device=self._device,
            ),
            _torch=self._torch,
            _warp=self._warp,
        )

    def step(
        self,
        sim_state: Any,
        sim_control: Any,
        current_state: MappedGRUState | None,
        next_state: MappedGRUState | None,
        dt: float | None = None,
        *,
        dynamic_bias: Any | None = None,
    ) -> None:
        """Assemble, infer, clamp, scatter, and advance one recurrent timestep."""

        if not isinstance(current_state, MappedGRUState) or not isinstance(next_state, MappedGRUState):
            raise TypeError("mapped GRU step requires current and next MappedGRUState")
        step_dt = self._request.control_dt_s if dt is None else float(dt)
        if not math.isfinite(step_dt) or not math.isclose(
            step_dt,
            self._request.control_dt_s,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("mapped GRU timestep does not match frozen control timestep")

        position_array = self._runtime_tensor(sim_state.joint_q, label="state.joint_q")
        velocity_array = self._runtime_tensor(sim_state.joint_qd, label="state.joint_qd")
        target_array = self._runtime_tensor(
            sim_control.joint_target_q,
            label="control.joint_target_q",
        )
        effort_array = self._runtime_tensor(sim_control.joint_f, label="control.joint_f")
        if self._request.actuator.target == "torque_residual":
            target_velocity = self._runtime_tensor(
                sim_control.joint_target_qd,
                label="control.joint_target_qd",
            )
            if self._torch.any(target_velocity[self._output_indices] != 0.0).item():
                raise ValueError("residual mapped GRU requires zero target velocity")
        position = position_array[self._position_indices]
        velocity = velocity_array[self._velocity_indices]
        position_error = target_array[self._target_position_indices] - position

        if dynamic_bias is None:
            bias = position.new_zeros(position.shape)
        else:
            bias_array = self._runtime_tensor(dynamic_bias, label="dynamic_bias")
            bias = bias_array[self._velocity_indices]
        if self._request.requires_dynamic_bias and dynamic_bias is None:
            raise ValueError("dynamic_bias feature requires SolverMuJoCo.eval_dynamic_bias output")

        feature_values = {
            "position": position,
            "position_error": position_error,
            "velocity": velocity,
            "solver_pd": self._pd_kp * position_error - self._pd_kd * velocity,
            "dynamic_bias": bias,
            "previous_torque": current_state.previous_torque,
        }
        physical = self._torch.cat(
            [feature_values[feature.name] for feature in self._request.metadata.input_features],
            dim=1,
        )
        expected_input = (self._world_count, self._request.metadata.input_size)
        if tuple(physical.shape) != expected_input:
            raise ValueError(f"mapped GRU input shape is {tuple(physical.shape)}, expected {expected_input}")
        if not self._torch.isfinite(physical).all().item():
            raise ValueError("mapped GRU physical input contains non-finite values")
        normalized = (physical - self._input_mean) / self._input_std

        with self._torch.inference_mode():
            result = self._network(normalized.unsqueeze(1), current_state.hidden)
            if not isinstance(result, (tuple, list)) or len(result) != 2:
                raise ValueError("mapped GRU network must return output and hidden tensors")
            output, hidden = result
            expected_output = (self._world_count, 1, self._output_width)
            expected_hidden = (
                self._request.metadata.recurrent.num_layers,
                self._world_count,
                self._request.metadata.recurrent.hidden_size,
            )
            if tuple(output.shape) != expected_output:
                raise ValueError(f"mapped GRU output shape is {tuple(output.shape)}, expected {expected_output}")
            if tuple(hidden.shape) != expected_hidden:
                raise ValueError(f"mapped GRU hidden shape is {tuple(hidden.shape)}, expected {expected_hidden}")
            raw_physical = output[:, 0, :] * self._target_std + self._target_mean
            if not self._torch.isfinite(raw_physical).all().item():
                raise ValueError("mapped GRU network returned non-finite output")
            if not self._torch.isfinite(hidden).all().item():
                raise ValueError("mapped GRU network returned non-finite hidden state")
            next_state.hidden.copy_(hidden)
            next_state.previous_torque.copy_(raw_physical)
            applied = self._torch.maximum(
                self._torch.minimum(raw_physical, self._effort_limits),
                -self._effort_limits,
            )
            effort_array.index_add_(
                0,
                self._output_indices.reshape(-1),
                applied.reshape(-1),
            )


def build_mapped_gru_actuator(request: MappedNeuralRequest) -> MappedGRUActuator:
    """Construct the production schema-v3 actuator for one frozen mapping."""

    return MappedGRUActuator(request)


__all__ = [
    "MappedGRUActuator",
    "MappedGRUState",
    "build_mapped_gru_actuator",
    "validate_mapped_gru_request",
]
