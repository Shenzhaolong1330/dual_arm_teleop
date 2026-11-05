"""One-shot stop must never go through connect/enable/home initialization."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from scripts.core import stop_robot
from scripts.core.robot_selection import load_selected_robot


def config_file(tmp_path, robot_type="nero_dual_arm", robot=None):
    path = tmp_path / "record.yaml"
    path.write_text(yaml.safe_dump({"record": {
        "robot_type": robot_type, "robot": robot or {},
    }}))
    return path


def test_dry_run_does_not_import_or_connect_hardware(tmp_path, monkeypatch, capsys):
    path = config_file(tmp_path, robot={"robot_ip": "192.0.2.12", "robot_port": 4444})
    monkeypatch.setitem(sys.modules, "zerorpc", None)
    monkeypatch.setitem(sys.modules, "flexivrdk", None)
    stop_robot.main(["--config", str(path), "--dry-run"])
    out = capsys.readouterr().out
    assert '"robot_ip": "192.0.2.12"' in out
    assert '"robot_port": 4444' in out
    assert "DRY_RUN_OK" in out


@pytest.mark.parametrize("robot_type,backend", [
    ("nero_dual_arm", "_stop_nero"), ("flexiv_dual_arm", "_stop_flexiv"),
])
def test_dispatch_follows_current_config(tmp_path, monkeypatch, robot_type, backend):
    path = config_file(tmp_path, robot_type)
    called = Mock(return_value={"right": {"ok": True}})
    for name in ("_stop_nero", "_stop_flexiv"):
        monkeypatch.setattr(stop_robot, name, Mock(side_effect=AssertionError("Wrong robot selected")))
    monkeypatch.setattr(stop_robot, backend, called)
    stop_robot.main(["--config", str(path), "--arm", "right"])
    assert called.call_args.args[1] == ("right",)


def test_explicit_hardware_file_and_inline_overrides(tmp_path):
    detail = tmp_path / "hardware.yaml"
    detail.write_text(yaml.safe_dump({"robot": {"robot_ip": "192.0.2.1", "robot_port": 9000}}))
    path = tmp_path / "record.yaml"
    path.write_text(yaml.safe_dump({"record": {
        "robot_type": "nero_dual_arm", "das_config_path": str(detail),
        "robot": {"robot_ip": "192.0.2.2"},
    }}))
    robot_type, config = load_selected_robot(path)
    assert robot_type == "nero_dual_arm"
    assert config == {"robot_ip": "192.0.2.2", "robot_port": 9000}


@pytest.mark.parametrize("robot_type", ["franka", "franka_dual_arm", "typo"])
def test_unsupported_robot_sends_no_commands(tmp_path, monkeypatch, robot_type):
    path = config_file(tmp_path, robot_type)
    monkeypatch.setitem(sys.modules, "zerorpc", None)
    monkeypatch.setitem(sys.modules, "flexivrdk", None)
    with pytest.raises(SystemExit) as error:
        stop_robot.main(["--config", str(path)])
    assert error.value.code == 2


def test_nero_attempts_both_arms_on_first_rpc_failure_and_closes(monkeypatch):
    transport = Mock()
    transport.robot_stop.side_effect = [RuntimeError("left unreachable"), True]
    transport.right_robot_get_arm_status.return_value = {"arm_status": 1}
    monkeypatch.setitem(sys.modules, "zerorpc", SimpleNamespace(Client=lambda **kw: transport))
    result = stop_robot._stop_nero({"robot_ip": "192.0.2.1", "robot_port": 4242}, ("left", "right"), 1)
    assert [call.args for call in transport.robot_stop.call_args_list] == [("left_robot",), ("right_robot",)]
    assert not result["left"]["ok"] and result["right"]["ok"]
    transport.close.assert_called_once()
    assert {call[0] for call in transport.method_calls} == {
        "connect", "robot_stop", "right_robot_get_arm_status", "close",
    }


def test_nero_requires_stop_status_not_just_rpc_ack(monkeypatch):
    transport = Mock()
    transport.robot_stop.return_value = True
    transport.left_robot_get_arm_status.return_value = {"arm_status": 0}
    monkeypatch.setitem(sys.modules, "zerorpc", SimpleNamespace(Client=lambda **kw: transport))
    result = stop_robot._stop_nero({"robot_ip": "192.0.2.1", "robot_port": 4242}, ("left",), 0)
    assert result["left"]["ok"] is False
    assert "not confirmed" in result["left"]["error"]
    transport.close.assert_called_once()


def test_flexiv_stops_without_enable_home_or_brake_and_attempts_other_arm(monkeypatch):
    left, right = Mock(), Mock()
    left.Stop.side_effect = RuntimeError("left stop failed")
    right.stopped.return_value = True
    right.mode.return_value = "IDLE"
    factory = Mock(side_effect=[left, right])
    monkeypatch.setitem(sys.modules, "flexivrdk", SimpleNamespace(
        Robot=factory, Mode=SimpleNamespace(IDLE="IDLE"),
    ))
    result = stop_robot._stop_flexiv({"left_robot_sn": "left", "right_robot_sn": "right"}, ("left", "right"), 1)
    assert not result["left"]["ok"] and result["right"]["ok"]
    assert result["right"]["brakes_engaged"] == "unverified"
    assert [call[0] for call in left.method_calls] == ["Stop"]
    assert [call[0] for call in right.method_calls] == ["Stop", "stopped", "mode"]


def test_partial_failure_returns_nonzero_with_both_results(tmp_path, monkeypatch, capsys):
    path = config_file(tmp_path)
    monkeypatch.setattr(stop_robot, "_stop_nero", lambda *args: {
        "left": {"ok": False, "error": "unreachable"}, "right": {"ok": True},
    })
    with pytest.raises(SystemExit) as error:
        stop_robot.main(["--config", str(path)])
    assert error.value.code == 1
    out = capsys.readouterr().out
    assert '"left"' in out and '"right"' in out and '"unreachable"' in out


def test_missing_selection_is_rejected(tmp_path):
    path = tmp_path / "record.yaml"
    path.write_text("record: {}")
    with pytest.raises(ValueError, match="explicitly select"):
        load_selected_robot(path)
