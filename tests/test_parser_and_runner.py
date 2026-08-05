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
  usd_path: assets/robots/test.usd
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
    path.write_text(
        f"{VALID_YAML}\noutput_dir: {tmp_path / 'output'}\n",
        encoding="utf-8",
    )
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
        def run(self) -> Path:
            seen.append(self.config)
            return Path(self.config.output_dir)

    assert RecordingTask.name() == "actuator_sys_id"
    result = TaskRunner({"actuator_sys_id": RecordingTask}).run(str(config_path))

    assert result.parent == config_path.parent / "output"
    assert (result / "config.yaml").is_file()
    assert (result / ".completed").is_file()
    assert len(seen) == 1
    assert isinstance(seen[0], ActuatorSysIdConfig)


def test_runner_skips_completed_task_unless_forced(config_path):
    run_count = 0

    class RecordingTask(BaseTask[ActuatorSysIdConfig]):
        def run(self) -> Path:
            nonlocal run_count
            run_count += 1
            return Path(self.config.output_dir)

    runner = TaskRunner({"actuator_sys_id": RecordingTask})
    first_output = runner.run(str(config_path))
    second_output = runner.run(str(config_path))

    assert second_output == first_output
    assert run_count == 1

    forced_output = runner.run(str(config_path), force=True)

    assert forced_output == first_output
    assert run_count == 2
    assert (forced_output / ".completed").is_file()


def test_forced_failure_removes_previous_completion_marker(config_path):
    should_fail = False

    class FailableTask(BaseTask[ActuatorSysIdConfig]):
        def run(self) -> Path:
            if should_fail:
                raise RuntimeError("task failed")
            return Path(self.config.output_dir)

    runner = TaskRunner({"actuator_sys_id": FailableTask})
    output_directory = runner.run(str(config_path))
    assert (output_directory / ".completed").is_file()

    should_fail = True
    with pytest.raises(RuntimeError, match="task failed"):
        runner.run(str(config_path), force=True)

    assert not (output_directory / ".completed").exists()


def test_base_task_rejects_multiple_inheritance():
    class TaskMixin:
        pass

    with pytest.raises(TypeError, match="directly inherit only from BaseTask"):

        class MultipleInheritanceTask(
            BaseTask[ActuatorSysIdConfig],
            TaskMixin,
        ):
            def run(self) -> Path:
                return Path("output")


def test_base_task_rejects_indirect_inheritance():
    class DirectTask(BaseTask[ActuatorSysIdConfig]):
        def run(self) -> Path:
            return Path("output")

    with pytest.raises(TypeError, match="directly inherit only from BaseTask"):

        class IndirectTask(DirectTask):
            pass
