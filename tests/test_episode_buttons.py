"""A/B must persist/discard before resetting, then advance without keyboard input."""

from contextlib import ExitStack
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from robots.dual_arm_schema import action_feature_names, observation_state_feature_names
from scripts.core import run_record as recording
from teleoperators.oculus_teleoperator.config_oculus_teleop import OculusTeleopConfig


def test_button_edges_are_shared_across_episodes_and_discard_wins():
    events = {}
    both = {"save_episode_requested": True, "discard_episode_requested": True}
    assert recording._episode_control_boundary(both, events)
    assert events.pop("episode_control") == "discard"
    events["rerecord_episode"] = False
    events["reset_episode"] = False
    # Holding either button, or releasing just one, cannot trigger a second action.
    for action in (both, {"save_episode_requested": True}, {"discard_episode_requested": True}):
        assert recording._episode_control_boundary(action, events)
        assert "episode_control" not in events
        assert events["reset_episode"] is False
    assert not recording._episode_control_boundary({}, events)
    assert recording._episode_control_boundary({"save_episode_requested": True}, events)
    assert events["episode_control"] == "save"


def run_session(root, mode, *, save_fails=False, reset_fails_after_save=False):
    action = dict.fromkeys(action_feature_names(), 0.0)
    action.update(left_gripper_width=0.02, right_gripper_width=0.03)
    obs = dict.fromkeys(observation_state_feature_names(), 0.0)
    obs.update(left_gripper_width=0.02, right_gripper_width=0.03)
    save = dict(action, save_episode_requested=True, reset_requested=True)
    discard = dict(action, discard_episode_requested=True)
    frame1 = dict(action, **{"left_delta_ee_pose.x": 0.001})
    frame2 = dict(action, **{"left_delta_ee_pose.x": 0.002})
    frame3 = dict(action, **{"left_delta_ee_pose.x": 0.003})
    # Empty A, held A, saved segment, discarded segment, saved segment.
    sequence = [save, save, frame1, save, save, frame2, discard, discard, frame3, save]
    events = dict(exit_early=False, stop_recording=False, rerecord_episode=False)
    robot = Mock()
    robot.name = robot.robot_type = "mock"
    robot.cameras = {}
    robot.action_features = dict.fromkeys(action_feature_names(), float)
    robot.observation_features = dict.fromkeys(observation_state_feature_names(), float)
    robot.get_observation.return_value = obs
    robot.send_action.side_effect = lambda value: dict(value)
    teleop = Mock()
    teleop.get_action.side_effect = sequence
    cfg = SimpleNamespace(
        dataset_name="button_test", dataset_root=root, run_mode=mode, fps=30,
        left_wrist_cam_serial="fake", right_wrist_cam_serial="fake", head_cam_serial="fake",
        cam_width=64, cam_height=64, robot_ip="none", robot_port=0,
        debug=True, use_gripper=True, gripper_max_open=0.085, gripper_force=10,
        gripper_speed=0.1, close_threshold=0.5, gripper_reverse=False,
        control_mode="oculus", robot_extra_config={}, robot_type="mock", resume=False,
        create_teleop_config=lambda: OculusTeleopConfig(processing_profile="cross_embodiment_v1"),
        save_meta_period=1, display=False, num_episodes=2, episode_time_sec=10, reset_time_sec=10,
        task_description="episode buttons", reset_on_finish=True, disconnect_on_finish=True,
        push_to_hub=False, success_policy=recording.SUCCESS_POLICY_RECORDED_IS_SUCCESS,
        annotate_success=False, dual_arm=True,
        policy=SimpleNamespace(device="cpu", use_amp=False, pretrained_path=None),
    )
    policy = Mock(config=cfg.policy)
    patch_values = {
        "create_robot": robot, "create_robot_config": None, "OculusTeleop": teleop,
        "init_keyboard_listener": (None, events),
        "make_default_processors": (lambda pair: dict(pair[0]), lambda pair: dict(pair[0]), dict),
        "make_policy": policy, "make_pre_post_processors": (Mock(), Mock()),
    }
    saved_before_reset = []

    def reset_robot():
        info = json.loads((root / "meta/info.json").read_text())
        saved_before_reset.append(info["total_episodes"])
        if reset_fails_after_save and info["total_episodes"] == 1:
            raise RuntimeError("reset failed after saving")

    robot.reset.side_effect = reset_robot
    with ExitStack() as stack:
        for name, value in patch_values.items():
            stack.enter_context(patch.object(recording, name, return_value=value))
        stack.enter_context(patch.object(recording, "update_dataset_info"))
        cleanup = stack.enter_context(patch.object(recording, "handle_incomplete_dataset"))
        wait = stack.enter_context(patch.object(recording, "_wait_for_next_episode_with_teleop",
                                               side_effect=AssertionError("A/B must auto-advance")))
        stack.enter_context(patch.object(recording, "busy_wait"))
        stack.enter_context(patch.object(recording, "make_robot_action", side_effect=lambda a, f: dict(a)))
        # Mix uses the same command stream as its policy, for an observable marker.
        predicted = iter((frame1, frame2, frame3))
        predict = stack.enter_context(patch.object(recording, "predict_action",
                                                  side_effect=lambda **kw: next(predicted)))
        if save_fails:
            stack.enter_context(patch.object(LeRobotDataset, "save_episode", side_effect=OSError("disk full")))
        if save_fails or reset_fails_after_save:
            with pytest.raises(SystemExit):
                recording.run_record(cfg)
        else:
            recording.run_record(cfg)
        wait.assert_not_called()
        if reset_fails_after_save:
            cleanup.assert_not_called()
    return robot, saved_before_reset, predict


@pytest.mark.parametrize("mode", [recording.RUN_MODE_RECORD, recording.RUN_MODE_MIX])
def test_save_discard_reset_auto_next_and_saved_episodes_survive(tmp_path, mode):
    root = tmp_path / mode
    robot, saved_before_reset, predict = run_session(root, mode)
    assert saved_before_reset == [0, 1, 1, 2]
    assert robot.send_action.call_count == 3  # No button frames/reset trajectory recorded.
    dataset = LeRobotDataset("button_test", root=root)
    assert dataset.num_episodes == 2
    assert len(dataset) == 2
    x = dataset.features["action"]["names"].index("left_delta_ee_pose.x")
    np.testing.assert_allclose([float(dataset[i]["action"][x]) for i in range(2)], [0.001, 0.003])
    assert [int(dataset[i]["episode_index"]) for i in range(2)] == [0, 1]
    if mode == recording.RUN_MODE_MIX:
        assert predict.call_count == 3  # No inference on the A/B transition frames.
        assert all(bool(dataset[i]["success"]) for i in range(2))


def test_save_failure_does_not_reset_after_failed_persistence(tmp_path):
    robot, saved_before_reset, _ = run_session(tmp_path / "save_fails", recording.RUN_MODE_RECORD,
                                              save_fails=True)
    assert saved_before_reset == [0]  # Only the initial empty-A reset occurred.
    assert robot.reset.call_count == 1


def test_reset_failure_preserves_already_saved_episode_without_delete_prompt(tmp_path):
    root = tmp_path / "reset_fails"
    _, saved_before_reset, _ = run_session(root, recording.RUN_MODE_RECORD,
                                         reset_fails_after_save=True)
    assert saved_before_reset == [0, 1]
    dataset = LeRobotDataset("button_test", root=root)
    assert dataset.num_episodes == 1
    assert len(dataset) == 1
