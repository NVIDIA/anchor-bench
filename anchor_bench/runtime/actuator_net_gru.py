from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_GRU_INPUT_COLUMNS = ("position", "position_error", "velocity")
SUPPORTED_GRU_INPUT_COLUMNS = {"position", "position_error", "velocity", "solver_pd"}


def _coerce_configclass(candidate):
    if callable(candidate):
        return candidate
    nested = getattr(candidate, "configclass", None)
    if callable(nested):
        return nested
    raise TypeError("isaaclab.utils.configclass is not a callable decorator")


try:  # pragma: no cover - exercised only in full Isaac Lab runtime
    from isaaclab.actuators import DCMotor, DCMotorCfg, ImplicitActuator, ImplicitActuatorCfg
    from isaaclab.utils import configclass as _configclass
    from isaaclab.utils.assets import read_file

    configclass = _coerce_configclass(_configclass)
    _ISAACLAB_AVAILABLE = True
except Exception:  # pragma: no cover - host-only fallback is covered by tests
    _ISAACLAB_AVAILABLE = False

    def configclass(cls):
        return dataclass(cls)

    @dataclass
    class DCMotorCfg:
        joint_names_expr: list[str] = field(default_factory=list)

    @dataclass
    class ImplicitActuatorCfg:
        joint_names_expr: list[str] = field(default_factory=list)

    class DCMotor:
        pass

    class ImplicitActuator:
        pass

    def read_file(path: str):
        return str(path)


def validate_gru_input_columns(columns: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    resolved = tuple(str(column) for column in columns)
    unsupported = sorted(set(resolved) - SUPPORTED_GRU_INPUT_COLUMNS)
    if unsupported:
        raise ValueError(f"unsupported GRU input columns: {', '.join(unsupported)}")
    if not resolved:
        raise ValueError("GRU input columns must be non-empty")
    return resolved


def _import_torch():
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("ActuatorNetGRU requires PyTorch in the full runtime.") from exc
    return torch


def _load_stats(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


class _GRURuntimeMixin:
    def _init_gru_runtime(self) -> None:
        if not _ISAACLAB_AVAILABLE:  # pragma: no cover - defensive runtime guard
            raise RuntimeError("ActuatorNetGRU requires Isaac Lab runtime imports.")
        self._torch = _import_torch()
        self._gru_input_columns = validate_gru_input_columns(tuple(self.cfg.input_columns))
        self._gru_output_column = str(self.cfg.output_column)
        self._gru_stats = _load_stats(getattr(self.cfg, "stats_file", None))

        file_bytes = read_file(self.cfg.network_file)
        self.network = self._torch.jit.load(file_bytes, map_location=self._device).eval()
        if not hasattr(self.network, "gru"):
            raise ValueError(f"{self.cfg.network_file} must expose a TorchScript .gru module")

        gru_state = self.network.gru.state_dict()
        input_dim = int(gru_state["weight_ih_l0"].shape[1])
        hidden_dim = int(gru_state["weight_hh_l0"].shape[1])
        num_layers = sum(1 for key in gru_state if key.startswith("weight_ih_l"))
        if input_dim != len(self._gru_input_columns):
            raise ValueError(
                f"{self.cfg.network_file} expects {input_dim} inputs but cfg.input_columns has "
                f"{len(self._gru_input_columns)} columns"
            )

        batch = self._num_envs * self.num_joints
        self.sea_input = self._torch.zeros(batch, 1, input_dim, device=self._device)
        self.sea_hidden_state = self._torch.zeros(num_layers, batch, hidden_dim, device=self._device)
        self.sea_hidden_state_per_env = self.sea_hidden_state.view(
            num_layers, self._num_envs, self.num_joints, hidden_dim
        )

    def _reset_gru_state(self, env_ids) -> None:
        with self._torch.no_grad():
            self.sea_hidden_state_per_env[:, env_ids] = 0.0

    def _input_stats(self, column: str) -> tuple[float, float] | None:
        inputs = self._gru_stats.get("normalization", {}).get("inputs", {})
        mean = inputs.get("mean", {}).get(column)
        std = inputs.get("std", {}).get(column)
        if mean is None or std is None:
            return None
        return float(mean), max(float(std), 1.0e-8)

    def _output_stats(self) -> tuple[float, float]:
        targets = self._gru_stats.get("normalization", {}).get("targets", {})
        mean = targets.get("mean", {}).get(self._gru_output_column, 0.0)
        std = targets.get("std", {}).get(self._gru_output_column, 1.0)
        return float(mean), max(float(std), 1.0e-8)

    def _normalize_input(self, column: str, values):
        stats = self._input_stats(column)
        if stats is None:
            return values
        mean, std = stats
        return (values - mean) / std

    def _position_error(self, control_action, joint_pos):
        if control_action.joint_positions is None:
            raise ValueError("GRU actuator input requires control_action.joint_positions")
        return (control_action.joint_positions - joint_pos).flatten()

    def _fill_gru_input(self, control_action, joint_pos, joint_vel) -> None:
        pos_error = None
        velocity = joint_vel.flatten()
        for index, column in enumerate(self._gru_input_columns):
            if column == "position":
                values = joint_pos.flatten()
            elif column == "position_error":
                pos_error = self._position_error(control_action, joint_pos) if pos_error is None else pos_error
                values = pos_error
            elif column == "velocity":
                values = velocity
            elif column == "solver_pd":
                pos_error = self._position_error(control_action, joint_pos) if pos_error is None else pos_error
                values = self.stiffness.flatten() * pos_error - self.damping.flatten() * velocity
            else:  # pragma: no cover - validate_gru_input_columns catches this
                raise ValueError(f"unsupported GRU input column: {column}")
            self.sea_input[:, 0, index] = self._normalize_input(column, values)

    def _predict_gru_effort(self, control_action, joint_pos, joint_vel):
        self._fill_gru_input(control_action, joint_pos, joint_vel)
        with self._torch.inference_mode():
            output, self.sea_hidden_state[:] = self.network(self.sea_input, self.sea_hidden_state)
        mean, std = self._output_stats()
        output = output * std + mean
        return output.reshape(self._num_envs, self.num_joints)


class ActuatorNetGRU(_GRURuntimeMixin, DCMotor):
    """Isaac Lab-style explicit GRU actuator that predicts total joint effort."""

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self._init_gru_runtime()

    def reset(self, env_ids):
        self._reset_gru_state(env_ids)

    def compute(self, control_action, joint_pos, joint_vel):
        self.computed_effort = self._predict_gru_effort(control_action, joint_pos, joint_vel)
        self._joint_vel[:] = joint_vel
        self.applied_effort = self._clip_effort(self.computed_effort)
        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action


class ActuatorNetGRUResidual(_GRURuntimeMixin, ImplicitActuator):
    """Implicit PD actuator that adds a GRU-predicted residual effort."""

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self._init_gru_runtime()

    def reset(self, env_ids):
        super().reset(env_ids)
        self._reset_gru_state(env_ids)

    def compute(self, control_action, joint_pos, joint_vel):
        residual = self._predict_gru_effort(control_action, joint_pos, joint_vel)
        if control_action.joint_efforts is None:
            control_action.joint_efforts = residual
        else:
            control_action.joint_efforts = control_action.joint_efforts + residual

        pos_error = self._position_error(control_action, joint_pos).reshape(self._num_envs, self.num_joints)
        velocity_error = -joint_vel
        if control_action.joint_velocities is not None:
            velocity_error = control_action.joint_velocities - joint_vel
        self.computed_effort = self.stiffness * pos_error + self.damping * velocity_error + control_action.joint_efforts
        self.applied_effort = self._clip_effort(self.computed_effort)
        return control_action


@configclass
class ActuatorNetGRUCfg(DCMotorCfg):
    """Configuration for explicit full-torque GRU actuator models."""

    class_type: type = ActuatorNetGRU
    stiffness = None
    damping = None

    network_file: str = ""
    stats_file: str | None = None
    input_columns: tuple[str, ...] = DEFAULT_GRU_INPUT_COLUMNS
    output_column: str = "torque"


@configclass
class ActuatorNetGRUResidualCfg(ImplicitActuatorCfg):
    """Configuration for implicit-PD actuators with GRU residual torque."""

    class_type: type = ActuatorNetGRUResidual

    network_file: str = ""
    stats_file: str | None = None
    input_columns: tuple[str, ...] = DEFAULT_GRU_INPUT_COLUMNS
    output_column: str = "torque_residual"


__all__ = [
    "ActuatorNetGRU",
    "ActuatorNetGRUCfg",
    "ActuatorNetGRUResidual",
    "ActuatorNetGRUResidualCfg",
    "DEFAULT_GRU_INPUT_COLUMNS",
    "SUPPORTED_GRU_INPUT_COLUMNS",
    "validate_gru_input_columns",
]
