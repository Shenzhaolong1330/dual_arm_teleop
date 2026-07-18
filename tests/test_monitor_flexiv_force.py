from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.tools import monitor_flexiv_force as monitor


class FakeStateRobot:
    def __init__(self, wrench):
        self.state = SimpleNamespace(ext_wrench_in_tcp_raw=wrench)
        self.states_calls = 0

    def states(self):
        self.states_calls += 1
        return self.state


class FakeStateGripper:
    def __init__(self, force):
        self.state = SimpleNamespace(force=force)
        self.states_calls = 0

    def states(self):
        self.states_calls += 1
        return self.state


def _devices(
    left_wrench=(1, -2, 3, -4, 5, -6),
    right_wrench=(10, -20, 30, -40, 50, -60),
    left_force=-7,
    right_force=8,
):
    devices = monitor.ConnectedDevices(
        left_robot=FakeStateRobot(left_wrench),
        right_robot=FakeStateRobot(right_wrench),
        left_gripper=FakeStateGripper(left_force),
        right_gripper=FakeStateGripper(right_force),
    )
    return devices


class FakeRerun:
    class Scalars:
        def __init__(self, value):
            self.value = value

    def __init__(self):
        self.time_sequences = []
        self.logs = []

    def set_time_sequence(self, timeline, sequence):
        self.time_sequences.append((timeline, sequence))

    def log(self, entity_path, scalar):
        self.logs.append((entity_path, scalar.value))


def test_reads_exact_raw_wrench_order_and_signed_gripper_force():
    devices = _devices()

    sample = monitor.read_force_sample(devices, timestamp=123.5)

    assert sample.timestamp == 123.5
    assert sample.left_tcp_raw == (1.0, -2.0, 3.0, -4.0, 5.0, -6.0)
    assert sample.right_tcp_raw == (10.0, -20.0, 30.0, -40.0, 50.0, -60.0)
    assert sample.left_gripper_force == -7.0
    assert sample.right_gripper_force == 8.0
    assert devices.left_robot.states_calls == 1
    assert devices.right_robot.states_calls == 1
    assert devices.left_gripper.states_calls == 1
    assert devices.right_gripper.states_calls == 1


def test_sample_and_report_contains_complete_left_and_right_terminal_data(capsys):
    devices = _devices(left_force=-2.25, right_force=3.5)

    monitor.sample_and_report(
        devices,
        sequence=0,
        print_sample=True,
        rerun_logger=None,
    )

    output = capsys.readouterr().out
    assert "t=" in output
    assert "left_tcp_raw=[1.0, -2.0, 3.0, -4.0, 5.0, -6.0]" in output
    assert "left_gripper_force=-2.25" in output
    assert "right_tcp_raw=[10.0, -20.0, 30.0, -40.0, 50.0, -60.0]" in output
    assert "right_gripper_force=3.5" in output


def test_rerun_records_fourteen_scalars_with_fixed_paths_and_signs():
    devices = _devices(left_force=-7, right_force=8)
    fake_rerun = FakeRerun()
    logger = monitor.RerunLogger(fake_rerun)

    monitor.sample_and_report(
        devices,
        sequence=12,
        print_sample=False,
        rerun_logger=logger,
    )

    assert len(fake_rerun.logs) == 14
    assert fake_rerun.time_sequences == [("sample", 12)]
    paths = [path for path, _value in fake_rerun.logs]
    assert paths == [path for path, _value in monitor.read_force_sample(devices).rerun_scalars()]
    values = [value for _path, value in fake_rerun.logs]
    assert values == [1.0, -2.0, 3.0, -4.0, 5.0, -6.0, -7.0, 10.0, -20.0, 30.0, -40.0, 50.0, -60.0, 8.0]


def test_input_values_are_not_filtered_absed_or_clipped():
    left = [0.125, -9.5, 100.0, -0.75, 2.5, -30.0]
    right = [-0.25, 8.0, -200.0, 0.5, -4.5, 60.0]
    devices = _devices(left_wrench=left, right_wrench=right, left_force=-11.0, right_force=12.0)

    sample = monitor.read_force_sample(devices)

    assert sample.left_tcp_raw == tuple(left)
    assert sample.right_tcp_raw == tuple(right)
    assert sample.left_gripper_force == -11.0
    assert sample.right_gripper_force == 12.0
    assert left == [0.125, -9.5, 100.0, -0.75, 2.5, -30.0]
    assert right == [-0.25, 8.0, -200.0, 0.5, -4.5, 60.0]


@pytest.mark.parametrize(
    ("state_factory", "message"),
    [
        (lambda: SimpleNamespace(), "ext_wrench_in_tcp_raw.*missing"),
        (lambda: SimpleNamespace(ext_wrench_in_tcp_raw=[1, 2, 3]), "length 6"),
        (
            lambda: SimpleNamespace(ext_wrench_in_tcp_raw=[1, 2, 3, 4, 5, float("nan")]),
            "not finite",
        ),
        (
            lambda: SimpleNamespace(ext_wrench_in_tcp_raw=[1, 2, 3, 4, 5, float("inf")]),
            "not finite",
        ),
    ],
)
def test_invalid_wrench_fails_without_fabricating_a_sample(state_factory, message):
    devices = _devices()
    devices.left_robot.state = state_factory()

    with pytest.raises(monitor.SampleReadError, match=message):
        monitor.read_force_sample(devices)


@pytest.mark.parametrize(
    ("force", "message"),
    [(float("nan"), "not finite"), (float("inf"), "not finite")],
)
def test_invalid_gripper_force_fails_without_replacing_it(force, message):
    devices = _devices(left_force=force)

    with pytest.raises(monitor.SampleReadError, match=message):
        monitor.read_force_sample(devices)


def test_missing_gripper_force_is_reported():
    devices = _devices()
    devices.right_gripper.state = SimpleNamespace()

    with pytest.raises(monitor.SampleReadError, match="right gripper field 'force'.*missing"):
        monitor.read_force_sample(devices)


def test_help_does_not_connect_hardware(monkeypatch, capsys):
    called = False

    def fail_if_called(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("hardware connection must not happen during --help")

    monkeypatch.setattr(monitor, "connect_devices", fail_if_called)
    with pytest.raises(SystemExit) as exc_info:
        monitor.main(["--help"])

    assert exc_info.value.code == 0
    assert not called
    assert "--robot-config" in capsys.readouterr().out


def test_config_loader_reads_existing_field_names_without_production_defaults(tmp_path):
    config_path = tmp_path / "flexiv.yaml"
    config_path.write_text(
        "robot:\n"
        "  left_robot_sn: left-from-file\n"
        "  right_robot_sn: right-from-file\n"
        "  left_gripper_name: left-gripper-from-file\n"
        "  right_gripper_name: right-gripper-from-file\n",
        encoding="utf-8",
    )

    config = monitor.load_monitor_config(config_path)

    assert config.left_robot_sn == "left-from-file"
    assert config.right_robot_sn == "right-from-file"
    assert config.left_gripper_name == "left-gripper-from-file"
    assert config.right_gripper_name == "right-gripper-from-file"


def test_direct_connection_only_enables_configured_grippers_by_default():
    calls = []

    class FakeRobot:
        def __init__(self, serial):
            calls.append(("Robot", serial))

        def Enable(self):
            calls.append(("Robot.Enable",))

    class FakeGripper:
        def __init__(self, robot):
            calls.append(("Gripper", robot))

        def Enable(self, name):
            calls.append(("Gripper.Enable", name))

    fake_rdk = SimpleNamespace(Robot=FakeRobot, Gripper=FakeGripper)
    config = monitor.MonitorConfig(
        source_path=monitor.Path("fake.yaml"),
        left_robot_sn="left-sn",
        right_robot_sn="right-sn",
        left_gripper_name="left-device",
        right_gripper_name="right-device",
    )

    monitor.connect_devices(config, rdk_module=fake_rdk)

    assert calls[0:2] == [("Robot", "left-sn"), ("Robot", "right-sn")]
    assert [call for call in calls if call[0] == "Gripper.Enable"] == [
        ("Gripper.Enable", "left-device"),
        ("Gripper.Enable", "right-device"),
    ]
    assert not any(call[0] == "Robot.Enable" for call in calls)
