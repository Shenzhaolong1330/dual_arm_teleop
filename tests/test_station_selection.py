"""Exercise per-machine overrides and the shared three-embodiment CLI offline."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

from scripts.core import reset_robot, run_record, run_replay, stop_robot
from scripts.core.robot_selection import load_selected_robot, load_session_config


ROOT = Path(__file__).resolve().parents[1]
BODIES = ("franka_dual_arm", "nero_dual_arm", "flexiv_dual_arm")


def session(tmp_path):
    path = tmp_path / "record.yaml"
    path.write_text((ROOT / "scripts/config/record_cfg.yaml").read_text())
    station = {
        "default_robot_type": "franka_dual_arm",
        "record": {"repo_id": "{robot_type}/station_task", "task": {"num_episodes": 2}},
        "robots": {
            "franka_dual_arm": {"robot": {"robot_ip": "192.0.2.10"},
                               "cameras": {"head_cam_serial": "LOCAL-HEAD"}},
            "nero_dual_arm": {"robot": {"robot_ip": "192.0.2.20", "robot_port": 4444}},
        },
    }
    (tmp_path / "station.local.yaml").write_text(yaml.safe_dump(station))
    return path


def test_station_default_and_cli_select_only_matching_hardware(tmp_path):
    path = session(tmp_path)
    cfg = load_session_config(path)
    record = run_record.RecordConfig(cfg["record"])
    assert record.robot_type == cfg["replay"]["robot_type"] == "franka_dual_arm"
    assert record.repo_id == "franka_dual_arm/station_task"
    assert record.robot_ip == "192.0.2.10" and record.head_cam_serial == "LOCAL-HEAD"
    assert record.num_episodes == 2
    cfg = load_session_config(path, robot_type="nero_dual_arm")
    record = run_record.RecordConfig(cfg["record"])
    assert record.repo_id == "nero_dual_arm/station_task"
    assert record.robot_ip == "192.0.2.20" and record.robot_port == 4444
    assert record.head_cam_serial != "LOCAL-HEAD"
    selected, robot = load_selected_robot(path, robot_type="nero_dual_arm")
    replay = run_replay.ReplayConfig(run_replay._hydrate_replay_robot_details(cfg["replay"]))
    assert selected == replay.robot_type == "nero_dual_arm"
    assert robot["robot_ip"] == replay.robot_ip == "192.0.2.20"
    assert robot["robot_port"] == replay.robot_port == 4444
    assert robot["use_gripper"] == replay.robot_config["use_gripper"] is True


def test_explicit_station_and_opt_out(tmp_path):
    path = session(tmp_path)
    alternate = tmp_path / "alternate.yaml"
    alternate.write_text("default_robot_type: flexiv_dual_arm\n")
    cfg = load_session_config(path, station_path=alternate)
    assert cfg["record"]["robot_type"] == "flexiv_dual_arm"
    assert cfg["record"]["repo_id"] == "{robot_type}/B-CARE"
    cfg = load_session_config(path, use_station=False)
    assert cfg["record"]["robot_type"] == "franka_dual_arm"
    with pytest.raises(FileNotFoundError):
        load_session_config(path, station_path=tmp_path / "missing.yaml")


def test_switch_rejects_hardware_belonging_to_previous_robot(tmp_path):
    path = tmp_path / "record.yaml"
    path.write_text(yaml.safe_dump({"record": {
        "robot_type": "nero_dual_arm", "robot": {"robot_ip": "192.0.2.20"},
    }}))
    with pytest.raises(ValueError, match="inline hardware"):
        load_session_config(path, robot_type="franka_dual_arm")


@pytest.mark.parametrize("robot_type", BODIES)
def test_record_dry_run_validates_each_body_without_hardware(tmp_path, monkeypatch, capsys, robot_type):
    path = session(tmp_path)
    monkeypatch.setattr(run_record, "run_record", Mock(side_effect=AssertionError("Hardware run")))
    monkeypatch.setattr(run_record, "create_robot", Mock(side_effect=AssertionError("Robot constructed")))
    run_record.main(["--config", str(path), "--robot-type", robot_type,
                     "--repo-id", "{robot_type}/shared_task", "--dry-run"])
    result = json.loads(capsys.readouterr().out)
    assert result["robot_type"] == robot_type
    assert result["repo_id"] == f"{robot_type}/shared_task"
    assert result["fps"] == 30
    assert result["quest_processing"]["profile"] == "cross_embodiment_v1"
    assert len(result["action_names"]) == len(result["observation_state_names"]) == 14
    assert result["robot"]["observation_pose_frame"] == "world"


def test_reset_and_replay_use_same_station_target(tmp_path, monkeypatch):
    path = session(tmp_path)
    adapter = Mock()
    monkeypatch.setattr(reset_robot, "create_robot", Mock(return_value=adapter))
    config_factory = Mock(side_effect=lambda **kwargs: kwargs)
    monkeypatch.setattr(reset_robot, "create_robot_config", config_factory)
    reset_robot.main(["--config", str(path), "--robot-type", "nero_dual_arm"])
    assert config_factory.call_args.kwargs["robot_type"] == "nero_dual_arm"
    assert config_factory.call_args.kwargs["robot_ip"] == "192.0.2.20"
    assert config_factory.call_args.kwargs["use_gripper"] is True
    adapter.reset.assert_called_once()
    adapter.disconnect.assert_called_once()
    replay = Mock()
    monkeypatch.setattr(run_replay, "run_replay", replay)
    run_replay.main(["--config", str(path), "--robot-type", "nero_dual_arm"])
    assert replay.call_args.args[0].robot_ip == "192.0.2.20"


def test_stop_resolves_same_override_without_importing_hardware(tmp_path, monkeypatch, capsys):
    import sys
    path = session(tmp_path)
    monkeypatch.setitem(sys.modules, "zerorpc", None)
    stop_robot.main(["--config", str(path), "--robot-type", "nero_dual_arm", "--dry-run"])
    output = capsys.readouterr().out
    assert '"robot_ip": "192.0.2.20"' in output and '"robot_port": 4444' in output
