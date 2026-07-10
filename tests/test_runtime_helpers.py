"""Focused host tests for runtime helpers used by authoritative phases."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest


def test_to_torch_accepts_isaac_lab_proxy_array() -> None:
    """Use an Isaac Lab ProxyArray's explicit Torch accessor."""

    from anchor_bench.runtime.robot_scene import to_torch

    class FakeTensor:
        """Stand in for torch.Tensor during a host-only test."""

    class FakeWarp:
        """Reject conversion through Warp when a proxy tensor exists."""

        class array:
            """Stand in for the Warp array type."""

        @staticmethod
        def to_torch(value):
            """Fail if the proxy takes the wrong conversion path."""

            raise AssertionError("ProxyArray should use its .torch accessor")

    tensor = FakeTensor()
    fake_torch = ModuleType("torch")
    fake_torch.Tensor = FakeTensor
    with patch.dict(sys.modules, {"torch": fake_torch}):
        assert to_torch(SimpleNamespace(torch=tensor), FakeWarp) is tensor


def test_runtime_unavailable_error_is_actionable() -> None:
    """Name the missing runtime and the supported container entrypoint."""

    from anchor_bench.runtime.environment import (
        RuntimeUnavailableError,
        require_full_runtime,
    )

    with pytest.raises(RuntimeUnavailableError) as error:
        require_full_runtime(
            ("definitely_missing_isaac_runtime_module",), workflow="sysid"
        )
    assert "sysid requires the full Anchor Bench Docker runtime" in str(error.value)
    assert "bash setup.sh --full" in str(error.value)
    assert "anchor-bench:full" in str(error.value)


def test_asset_resolution_error_names_supported_locations() -> None:
    """Report both configured and repository-local robot asset roots."""

    from anchor_bench.robots.registry import get_robot
    from anchor_bench.runtime.assets import AssetNotFoundError, resolve_robot_asset

    with patch.dict("os.environ", {}, clear=False) as environment:
        environment.pop("ANCHOR_BENCH_ASSET_DIR", None)
        with pytest.raises(AssetNotFoundError) as error:
            resolve_robot_asset(get_robot("so101"), "missing-so101.usd")
    assert str(error.value) == (
        "Missing asset 'missing-so101.usd' for robot so101.\n"
        "Looked in:\n"
        "- $ANCHOR_BENCH_ASSET_DIR\n"
        "- assets/robots/so101/files\n\n"
        "Set ANCHOR_BENCH_ASSET_DIR or place assets under "
        "assets/robots/so101/files/."
    )


def test_simulation_timing_uses_exact_profile_frequencies() -> None:
    """Derive exact step ratios and reject non-integral render cadence."""

    from anchor_bench.core.timing import resolve_simulation_timing
    from anchor_bench.sysid.schema import SysIdSimulationConfig

    def simulation(render_freq: int) -> SysIdSimulationConfig:
        """Build one complete authoritative simulation profile."""

        return SysIdSimulationConfig(
            physics_freq=500,
            control_freq=100,
            render_freq=render_freq,
            integrator="implicitfast",
            solver_iterations=10,
            line_search_iterations=10,
        )

    timing = resolve_simulation_timing(simulation(50), "sysid simulation")
    assert timing.control_divisor == 5
    assert timing.render_interval == 10
    assert timing.physics_dt == pytest.approx(0.002)
    assert timing.control_dt == pytest.approx(0.01)
    with pytest.raises(ValueError, match="physics_freq divisible by render_freq"):
        resolve_simulation_timing(simulation(333), "sysid simulation")


def test_scene_builder_passes_fixed_root_to_usd_spawn() -> None:
    """Forward the declarative robot root mode into Isaac Lab's USD spawn."""

    from anchor_bench.runtime.robot_scene import build_scene_class

    class FakeUsdFileCfg:
        """Capture authored USD spawn properties."""

        def __init__(self, **kwargs):
            """Store the received spawn values."""

            self.articulation_props = kwargs.get("articulation_props")

    class FakeArticulationRootPropertiesCfg:
        """Capture the fixed-root override."""

        def __init__(self, **kwargs):
            """Store the fixed-root value."""

            self.fix_root_link = kwargs.get("fix_root_link")

    class FakeArticulationCfg:
        """Capture the nested articulation spawn configuration."""

        class InitialStateCfg:
            """Accept the authored initial state."""

            def __init__(self, **kwargs):
                """Store initial-state values for completeness."""

                self.kwargs = kwargs

        def __init__(self, **kwargs):
            """Expose the captured USD spawn."""

            self.spawn = kwargs["spawn"]

    class FakeAssetBaseCfg:
        """Accept ground and light asset settings."""

        def __init__(self, **kwargs):
            """Store generic asset settings."""

            self.kwargs = kwargs

    fake_sim = SimpleNamespace(
        GroundPlaneCfg=lambda **kwargs: SimpleNamespace(kwargs=kwargs),
        UsdFileCfg=FakeUsdFileCfg,
        ArticulationRootPropertiesCfg=FakeArticulationRootPropertiesCfg,
        DomeLightCfg=lambda **kwargs: SimpleNamespace(kwargs=kwargs),
    )
    fake_isaaclab = ModuleType("isaaclab")
    fake_isaaclab.sim = fake_sim
    modules = {
        "isaaclab": fake_isaaclab,
        "isaaclab.sim": fake_sim,
        "isaaclab.actuators": SimpleNamespace(
            ImplicitActuatorCfg=lambda **kwargs: SimpleNamespace(kwargs=kwargs)
        ),
        "isaaclab.assets": SimpleNamespace(
            ArticulationCfg=FakeArticulationCfg,
            AssetBaseCfg=FakeAssetBaseCfg,
        ),
        "isaaclab.scene": SimpleNamespace(InteractiveSceneCfg=object),
        "isaaclab.utils": SimpleNamespace(configclass=lambda cls: cls),
    }
    with patch.dict(sys.modules, modules):
        scene = build_scene_class(
            Path("robot.usd"),
            {"stiffness": {"JointA": 1.0}, "damping": {"JointA": 1.0}},
            ["JointA"],
            fix_root=True,
        )
    assert scene.robot.spawn.articulation_props.fix_root_link is True


def test_viscous_friction_writes_newton_passive_damping() -> None:
    """Write mapped damping cells and notify Newton of a property change."""

    from anchor_bench.runtime.robot_scene import _write_viscous_friction

    class FakeTensor:
        """Expose tensor values through the Torch-like conversion chain."""

        def __init__(self, values):
            """Store numeric tensor values."""

            self._values = np.asarray(values, dtype=float)

        def detach(self):
            """Return an already detached tensor."""

            return self

        def cpu(self):
            """Return an already host-resident tensor."""

            return self

        def numpy(self):
            """Return the stored NumPy values."""

            return self._values

    class FakeDamping:
        """Mimic Newton's assignable passive-damping array."""

        def __init__(self):
            """Initialize two environments and three DOFs."""

            self.values = np.zeros((2, 3), dtype=float)

        def numpy(self):
            """Return a writable host copy."""

            return self.values.copy()

        def assign(self, values):
            """Install the modified damping values."""

            self.values = np.asarray(values, dtype=float)

    damping = FakeDamping()
    changes: list[str] = []

    class FakeNewtonManager:
        """Expose a live model and capture notifications."""

        @classmethod
        def get_model(cls):
            """Return the fake MuJoCo passive-damping array."""

            return SimpleNamespace(
                mujoco=SimpleNamespace(dof_passive_damping=damping)
            )

        @classmethod
        def add_model_change(cls, change):
            """Record one solver property notification."""

            changes.append(change)

    modules = {
        "isaaclab_newton": SimpleNamespace(),
        "isaaclab_newton.physics": SimpleNamespace(
            NewtonManager=FakeNewtonManager
        ),
        "newton": SimpleNamespace(),
        "newton.solvers": SimpleNamespace(
            SolverNotifyFlags=SimpleNamespace(
                JOINT_DOF_PROPERTIES="joint_dof_properties"
            )
        ),
    }
    robot = SimpleNamespace(joint_names=["JointA", "JointB", "JointC"])
    with patch.dict(sys.modules, modules):
        _write_viscous_friction(
            robot,
            FakeTensor([[0.2, 0.4], [0.6, 0.8]]),
            [0, 2],
        )
    np.testing.assert_allclose(
        damping.values,
        [[0.2, 0.0, 0.4], [0.6, 0.0, 0.8]],
    )
    assert changes == ["joint_dof_properties"]


def test_viscous_friction_requires_passive_damping() -> None:
    """Fail before mutation when Newton exposes no passive-damping array."""

    from anchor_bench.runtime.robot_scene import _write_viscous_friction

    class FakeNewtonManager:
        """Return a model without the required MuJoCo field."""

        @classmethod
        def get_model(cls):
            """Return an incomplete Newton model."""

            return SimpleNamespace(mujoco=SimpleNamespace())

    with patch.dict(
        sys.modules,
        {
            "isaaclab_newton": SimpleNamespace(),
            "isaaclab_newton.physics": SimpleNamespace(
                NewtonManager=FakeNewtonManager
            ),
        },
    ):
        with pytest.raises(RuntimeError, match="mujoco\\.dof_passive_damping"):
            _write_viscous_friction(
                SimpleNamespace(joint_names=["JointA"]),
                SimpleNamespace(),
                [0],
            )


def test_isaac_actuator_kwargs_maps_dry_friction() -> None:
    """Translate Anchor's dry-friction name to Isaac Lab's API keyword."""

    from anchor_bench.runtime.robot_scene import _isaac_implicit_actuator_kwargs

    values = _isaac_implicit_actuator_kwargs(
        {
            "stiffness": {"Rotation": 1.0},
            "damping": {"Rotation": 2.0},
            "dry_friction": {"Rotation": 0.3},
        }
    )
    assert values["dynamic_friction"] == {"Rotation": 0.3}
    assert "dry_friction" not in values
