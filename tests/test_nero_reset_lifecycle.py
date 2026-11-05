"""Reset and transport cleanup must leave the NUC holding its target."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from robots.dual_agilex_nero.config_nero import NeroDualArmConfig
from robots.dual_agilex_nero.nero_interface_client import NeroDualArmClient
from robots.dual_agilex_nero.nero_dual_arm import NeroDualArm
import robots.dual_agilex_nero.nero_dual_arm as nero_module


def test_transport_close_does_not_stop_either_arm():
    transport = Mock()
    client = NeroDualArmClient.__new__(NeroDualArmClient)
    client.server = transport
    client.close()
    client.close()
    transport.robot_stop.assert_not_called()
    transport.close.assert_called_once()


def test_disconnected_cartesian_client_never_reports_success():
    client = NeroDualArmClient.__new__(NeroDualArmClient)
    client.server = None
    with pytest.raises(ConnectionError, match="not connected"):
        client.servo_p_OL("left_robot", [0.01, 0, 0, 0, 0, 0], delta=True)


def test_explicit_stop_on_close_remains_available():
    transport = Mock()
    client = NeroDualArmClient.__new__(NeroDualArmClient)
    client.server = transport
    client.close(stop=True)
    assert transport.robot_stop.call_args_list[0].args == ('left_robot',)
    assert transport.robot_stop.call_args_list[1].args == ('right_robot',)
    transport.close.assert_called_once()


def test_reset_homes_then_closes_and_opens_grippers(monkeypatch):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(nero_module, 'time', SimpleNamespace(
        monotonic=lambda: clock.now,
        sleep=lambda dt: setattr(clock, 'now', clock.now + dt),
    ))
    events = []
    widths = {'left': 0.094, 'right': 0.086}

    def move(side, width, force):
        events.append((side, width, force))
        widths[side] = max(min(width, 0.094 if side == 'left' else 0.086),
                           0.0015 if side == 'left' else 0.0001)
        return True

    rpc = SimpleNamespace(
        robot_go_home=lambda: events.append('home'),
        left_gripper_goto=lambda width, force: move('left', width, force),
        right_gripper_goto=lambda width, force: move('right', width, force),
        left_gripper_get_state=lambda: {'width': widths['left']},
        right_gripper_get_state=lambda: {'width': widths['right']},
        close=lambda: events.append('close_transport'),
    )
    robot = NeroDualArm(NeroDualArmConfig(cameras={}, use_gripper=True))
    robot._robot = rpc
    robot.is_connected = True
    robot.reset()
    robot.disconnect()
    assert events == ['home', ('left', 0.0, 2.0), ('right', 0.0, 2.0),
                      ('left', 0.1, 2.0), ('right', 0.1, 2.0), 'close_transport']
    assert robot._left_gripper_width == 0.094
    assert robot._right_gripper_width == 0.086


def test_failed_home_does_not_cycle_grippers():
    rpc = Mock()
    rpc.robot_go_home.side_effect = RuntimeError('home failed')
    robot = NeroDualArm(NeroDualArmConfig(cameras={}, use_gripper=True))
    robot._robot = rpc
    robot.is_connected = True
    with pytest.raises(RuntimeError, match='home failed'):
        robot.reset()
    rpc.left_gripper_goto.assert_not_called()
    rpc.right_gripper_goto.assert_not_called()


def test_failed_reset_cleanup_only_closes_transport():
    transport = Mock()
    transport.robot_go_home.side_effect = RuntimeError('home failed')
    client = NeroDualArmClient.__new__(NeroDualArmClient)
    client.server = transport
    robot = NeroDualArm(NeroDualArmConfig(cameras={}, use_gripper=False))
    robot._robot = client
    robot.is_connected = True
    with pytest.raises(RuntimeError, match='home failed'):
        try:
            robot.reset()
        finally:
            robot.disconnect()
    transport.robot_stop.assert_not_called()
    transport.close.assert_called_once()
