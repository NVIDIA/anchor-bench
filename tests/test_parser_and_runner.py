from pathlib import Path

import pytest

from anchor_bench.core.task_runner import TaskRunner
from anchor_bench.errors import ConfigParseError, UnknownConfigTypeError
from anchor_bench.schema.actuator_sys_id import ActuatorSysIdConfig
from anchor_bench.schema.parse import parse
from anchor_bench.tasks.base_task import BaseTask

VALID_YAML = """\
type: actuator_sys_id
joint_parameters:
  - names: [hip]
    controller:
      type: pd
      clamps:
        effort_limit: 120.0
        velocity_limit: 30.0
      stiffness: {lower: 1.0, upper: 100.0, initial: 20.0}
      damping: {lower: 0.0, upper: 10.0}
    parameters:
      dry_friction: {lower: 0.0, upper: 1.0}
      viscous_friction: {lower: 0.0, upper: 1.0}
      armature: {lower: 0.0, upper: 0.1}
optimizer:
  type: python-cmaes
  sigma: 0.5
  max_iterations: 100
  epsilon: 0.001
simulation:
  type: newton
  physics_freq: 1000
  control_freq: 200
  render_freq: 60
  solver:
    type: warp_mujoco
    solver_iterations: 10
    line_search_iterations: 4
dataset:
  type: parquet
  commanded_state: {path: runs/commanded}
  measured_state: {path: runs/measured}
"""


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(VALID_YAML, encoding="utf-8")
    return path


def test_parse_selects_model_from_type(config_path):
    config = parse(config_path)

    assert isinstance(config, ActuatorSysIdConfig)


def test_parse_rejects_unknown_type(tmp_path):
    path = tmp_path / "unknown.yaml"
    path.write_text("type: unknown\n", encoding="utf-8")

    with pytest.raises(UnknownConfigTypeError, match="No parser is registered"):
        parse(path)


def test_parse_reports_missing_file(tmp_path):
    with pytest.raises(ConfigParseError, match="does not exist"):
        parse(tmp_path / "missing.yaml")


def test_runner_constructs_and_runs_registered_task(config_path):
    seen = []

    class RecordingTask(BaseTask[ActuatorSysIdConfig]):
        def __init__(self, config: ActuatorSysIdConfig) -> None:
            self.config = config

        @classmethod
        def name(cls) -> str:
            return "actuator_sys_id"

        def run(self):
            seen.append(self.config)
            return "complete"

    result = TaskRunner({"actuator_sys_id": RecordingTask}).run(str(config_path))

    assert result == "complete"
    assert len(seen) == 1
    assert isinstance(seen[0], ActuatorSysIdConfig)
