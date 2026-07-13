from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    from scripts.core.run_record import (
        EPISODE_CONTROL_OCULUS,
        EPISODE_PHASE_RECORDING,
        EPISODE_PHASE_RESETTING,
        EPISODE_PHASE_WAITING,
        EpisodeButtonController,
        ResetHomeOnRequestRobot,
        _new_episode_events,
        _request_episode_discard,
        _request_episode_start_or_save,
        _reset_home_after_episode,
        _set_episode_phase,
        _wait_for_episode_start,
        handle_incomplete_dataset,
    )
    from teleoperators.oculus_teleoperator.oculus_teleop import OculusTeleop
except ModuleNotFoundError as exc:
    if exc.name != "zerorpc":
        raise
    # The DP3 validation environment intentionally lacks robot RPC dependencies.
    # Run this hardware-orchestration test directly in the dual_arm_teleop env.
    import pytest

    pytest.skip("dual_arm_teleop runtime dependency zerorpc is unavailable", allow_module_level=True)


class FakeRobot:
    def __init__(self):
        self.reset_calls = 0
        self.observation_calls = 0
        self.sent_actions: list[dict] = []

    def reset(self):
        self.reset_calls += 1

    def get_observation(self):
        self.observation_calls += 1
        return {"robot_state": self.observation_calls}

    def send_action(self, action):
        self.sent_actions.append(dict(action))
        return action


class FakeWaitingTeleop:
    def __init__(self, controller: EpisodeButtonController, *, start_after: int = 1):
        self.controller = controller
        self.start_after = start_after
        self.calls = 0

    def get_action(self):
        self.calls += 1
        return self.controller.consume(
            {
                "x_button_pressed": self.calls >= self.start_after,
                "y_button_pressed": False,
                "left_delta_ee_pose.x": 0.001 * self.calls,
                "left_gripper_release_requested": False,
            }
        )


class FakeOculusRobot:
    def get_observations(self):
        values = {
            "x_button_pressed": True,
            "y_button_pressed": True,
            "reset_requested": False,
        }
        for arm in ("left", "right"):
            for axis in ("x", "y", "z", "rx", "ry", "rz"):
                values[f"{arm}_delta_ee_pose.{axis}"] = 0.0
        return values


class EpisodeControlTest(unittest.TestCase):
    def test_keyboard_style_commands_are_phase_sensitive(self):
        events = _new_episode_events()
        self.assertEqual(events["episode_phase"], EPISODE_PHASE_WAITING)
        self.assertEqual(_request_episode_start_or_save(events, source="test"), "start")
        self.assertTrue(events["start_recording"])

        _set_episode_phase(events, EPISODE_PHASE_RECORDING)
        self.assertEqual(_request_episode_start_or_save(events, source="test"), "save")
        self.assertTrue(events["exit_early"])

        _set_episode_phase(events, EPISODE_PHASE_RESETTING)
        self.assertEqual(_request_episode_discard(events, source="test"), "ignored")
        self.assertFalse(events["rerecord_episode"])

    def test_quest_buttons_use_rising_edges_and_y_consumes_gripper_release(self):
        events = _new_episode_events()
        controller = EpisodeButtonController(events)

        action = controller.consume(
            {
                "x_button_pressed": True,
                "y_button_pressed": False,
                "left_gripper_release_requested": False,
            }
        )
        self.assertTrue(events["start_recording"])
        self.assertFalse(action["left_gripper_release_requested"])

        _set_episode_phase(events, EPISODE_PHASE_RECORDING)
        controller.consume({"x_button_pressed": True, "y_button_pressed": False})
        self.assertFalse(events["exit_early"], "held X must not immediately stop recording")
        controller.consume({"x_button_pressed": False, "y_button_pressed": False})
        controller.consume({"x_button_pressed": True, "y_button_pressed": False})
        self.assertTrue(events["exit_early"])

        _set_episode_phase(events, EPISODE_PHASE_RECORDING)
        controller.consume({"x_button_pressed": False, "y_button_pressed": False})
        action = controller.consume(
            {
                "x_button_pressed": False,
                "y_button_pressed": True,
                "left_gripper_release_requested": True,
                "right_gripper_release_requested": True,
            }
        )
        self.assertTrue(events["rerecord_episode"])
        self.assertTrue(events["exit_early"])
        self.assertNotIn("left_gripper_release_requested", action)
        self.assertNotIn("right_gripper_release_requested", action)

    def test_wait_keeps_quest_robot_control_live_without_a_dataset(self):
        events = _new_episode_events()
        teleop = FakeWaitingTeleop(EpisodeButtonController(events), start_after=2)
        robot = FakeRobot()
        processor_inputs = []

        def identity_processor(value):
            processor_inputs.append(value)
            return value[0]

        with patch("scripts.core.run_record.busy_wait"):
            started = _wait_for_episode_start(
                events=events,
                robot=robot,
                teleop=teleop,
                teleop_action_processor=identity_processor,
                robot_action_processor=identity_processor,
                episode_index=0,
                num_episodes=2,
                fps=30,
                control_mode=EPISODE_CONTROL_OCULUS,
            )
        self.assertTrue(started)
        self.assertEqual(teleop.calls, 2)
        self.assertEqual(robot.observation_calls, 2)
        self.assertEqual(
            robot.sent_actions,
            [
                {"left_delta_ee_pose.x": 0.001, "left_gripper_release_requested": False},
                {"left_delta_ee_pose.x": 0.002, "left_gripper_release_requested": False},
            ],
        )
        self.assertEqual(len(processor_inputs), 4)
        self.assertEqual(events["episode_phase"], EPISODE_PHASE_RECORDING)

    def test_auto_reset_home_and_manual_a_reset_are_rising_edge_only(self):
        robot = FakeRobot()
        _reset_home_after_episode(robot, outcome="saved")
        self.assertEqual(robot.reset_calls, 1)

        wrapped = ResetHomeOnRequestRobot(robot)
        wrapped.send_action({"reset_requested": True})
        wrapped.send_action({"reset_requested": True})
        wrapped.send_action({"reset_requested": False, "action": 1})
        self.assertEqual(robot.reset_calls, 2)
        self.assertEqual(robot.sent_actions, [{"reset_requested": False, "action": 1}])

    def test_oculus_teleop_forwards_x_y_button_state(self):
        teleop = OculusTeleop.__new__(OculusTeleop)
        teleop.oculus_robot = FakeOculusRobot()
        teleop.cfg = SimpleNamespace(use_gripper=False)
        teleop._log_timing_debug = lambda _elapsed_ms: None
        action = OculusTeleop._get_action_impl(teleop)
        self.assertTrue(action["x_button_pressed"])
        self.assertTrue(action["y_button_pressed"])

    def test_incomplete_dataset_delete_confirmation_is_restored(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "dataset"
            dataset.mkdir()
            (dataset / "sentinel").write_text("keep until confirmed", encoding="utf-8")
            with (
                patch("scripts.core.run_record.termios.tcflush"),
                patch("builtins.input", return_value="y"),
            ):
                handle_incomplete_dataset(dataset, preserve=False)
            self.assertFalse(dataset.exists())

    def test_manifest_preservation_still_skips_delete_prompt_for_non_ctrl_c_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "dataset"
            dataset.mkdir()
            with patch("builtins.input") as prompt:
                handle_incomplete_dataset(dataset, preserve=True)
            prompt.assert_not_called()
            self.assertTrue(dataset.exists())


if __name__ == "__main__":
    unittest.main()
