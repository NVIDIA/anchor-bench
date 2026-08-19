# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import yaml

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
  source: runs/actuator.parquet
  resampling:
    type: uniform
    dt: 0.005
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
        def run(self, output_directory: Path) -> Path:
            assert output_directory.is_dir()
            assert (output_directory / "config.yaml").is_file()
            seen.append(self.config)
            return output_directory

    assert RecordingTask.name() == "actuator_sys_id"
    result = TaskRunner({"actuator_sys_id": RecordingTask}).run(str(config_path))

    assert result.parent == config_path.parent / "output"
    assert (result / "config.yaml").is_file()
    assert (result / ".completed").is_file()
    assert len(seen) == 1
    assert isinstance(seen[0], ActuatorSysIdConfig)


def test_runner_uses_explicit_output_folder(config_path, tmp_path):
    class RecordingTask(BaseTask[ActuatorSysIdConfig]):
        def run(self, output_directory: Path) -> Path:
            return output_directory

    requested_output = tmp_path / "named-output"
    result = TaskRunner({"actuator_sys_id": RecordingTask}).run(
        str(config_path),
        output=requested_output,
    )

    assert result == requested_output
    assert (result / "config.yaml").is_file()
    assert (result / ".completed").is_file()


def test_runner_skips_completed_task_without_rewriting_config_unless_forced(
    config_path,
    tmp_path,
):
    run_count = 0

    class RecordingTask(BaseTask[ActuatorSysIdConfig]):
        def run(self, output_directory: Path) -> Path:
            nonlocal run_count
            run_count += 1
            return output_directory

    runner = TaskRunner({"actuator_sys_id": RecordingTask})
    requested_output = tmp_path / "named-output"
    first_output = runner.run(str(config_path), output=requested_output)
    config_artifact = first_output / "config.yaml"
    config_artifact.write_text("stale: true\n", encoding="utf-8")

    second_output = runner.run(str(config_path), output=requested_output)

    assert second_output == first_output
    assert run_count == 1
    assert yaml.safe_load(config_artifact.read_text(encoding="utf-8")) == {
        "stale": True
    }

    forced_output = runner.run(str(config_path), force=True, output=requested_output)

    assert forced_output == first_output
    assert run_count == 2
    assert yaml.safe_load(config_artifact.read_text(encoding="utf-8")) == parse(
        config_path
    ).model_dump(mode="json")
    assert (forced_output / ".completed").is_file()


def test_forced_failure_removes_previous_completion_marker(config_path):
    should_fail = False

    class FailableTask(BaseTask[ActuatorSysIdConfig]):
        def run(self, output_directory: Path) -> Path:
            if should_fail:
                raise RuntimeError("task failed")
            return output_directory

    runner = TaskRunner({"actuator_sys_id": FailableTask})
    output_directory = runner.run(str(config_path))
    assert (output_directory / ".completed").is_file()

    should_fail = True
    with pytest.raises(RuntimeError, match="task failed"):
        runner.run(str(config_path), force=True)

    assert not (output_directory / ".completed").exists()


def test_forced_config_write_failure_removes_previous_completion_marker(
    config_path,
    monkeypatch,
    tmp_path,
):
    class RecordingTask(BaseTask[ActuatorSysIdConfig]):
        def run(self, output_directory: Path) -> Path:
            return output_directory

    runner = TaskRunner({"actuator_sys_id": RecordingTask})
    requested_output = tmp_path / "write-failure-output"
    output_directory = runner.run(str(config_path), output=requested_output)
    assert (output_directory / ".completed").is_file()

    def fail_write(_path: Path, _document: object) -> None:
        raise OSError("config write failed")

    monkeypatch.setattr("anchor_bench.core.task_runner.write_yaml", fail_write)

    with pytest.raises(OSError, match="config write failed"):
        runner.run(str(config_path), force=True, output=requested_output)

    assert not (output_directory / ".completed").exists()


def test_runner_rejects_a_task_returning_the_wrong_output_directory(config_path):
    class MisdirectedTask(BaseTask[ActuatorSysIdConfig]):
        def run(self, output_directory: Path) -> Path:
            return output_directory.parent

    with pytest.raises(RuntimeError, match="expected its assigned output directory"):
        TaskRunner({"actuator_sys_id": MisdirectedTask}).run(str(config_path))


def test_base_task_rejects_multiple_inheritance():
    class TaskMixin:
        pass

    with pytest.raises(TypeError, match="directly inherit only from BaseTask"):

        class MultipleInheritanceTask(
            BaseTask[ActuatorSysIdConfig],
            TaskMixin,
        ):
            def run(self, output_directory: Path) -> Path:
                return Path("output")


def test_base_task_rejects_indirect_inheritance():
    class DirectTask(BaseTask[ActuatorSysIdConfig]):
        def run(self, output_directory: Path) -> Path:
            return Path("output")

    with pytest.raises(TypeError, match="directly inherit only from BaseTask"):

        class IndirectTask(DirectTask):
            pass
