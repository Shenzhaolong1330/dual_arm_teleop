"""Regression coverage for the shared dual-arm mirror teleoperation path."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
import yaml

from scripts.core.run_record import RecordConfig
from teleoperators.oculus_teleoperator.oculus.oculus_dual_arm_robot import OculusDualArmRobot
from teleoperators.oculus_teleoperator.config_oculus_teleop import OculusTeleopConfig
from teleoperators.oculus_teleoperator.oculus_teleop import OculusTeleop


class _FakeOculusReader:
    def get_transformations_and_buttons(self):
        return {
            "l": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            "r": np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0]),
        }, {
            "LG": True,
            "RG": True,
            "A": False,
            "Y": True,
            "B": False,
            "leftTrig": (0.2,),
            "rightTrig": (0.7,),
            "LTr": True,
            "RTr": True,
        }


def _make_mirrored_oculus_robot() -> OculusDualArmRobot:
    """Create the core mapper without opening an ADB/Oculus connection."""

    robot = object.__new__(OculusDualArmRobot)
    robot._oculus_reader = _FakeOculusReader()
    robot._use_gripper = True
    robot._mirror_teleop = True
    robot._gripper_trigger_deadzone = 0.0
    robot._gripper_trigger_gamma = 1.0
    robot._left_pose_scaler = [1.0, 1.0]
    robot._right_pose_scaler = [1.0, 1.0]
    robot._left_channel_signs = [1, 1, 1, 1, 1, 1]
    robot._right_channel_signs = [1, 1, 1, 1, 1, 1]
    robot._position_axis_order = (0, 1, 2)
    robot._rotation_axis_order = (0, 1, 2)
    robot._action_smoothing_method = "none"
    robot._action_smoothing_alpha = 1.0
    robot._left_smoothed_delta = None
    robot._right_smoothed_delta = None
    robot._left_missing_hold_count = 0
    robot._right_missing_hold_count = 0
    robot._action_missing_hold_frames = 0
    robot._action_missing_decay = 0.0
    robot._action_deadband_translation = 0.0
    robot._action_deadband_rotation = 0.0
    robot._action_spike_translation = None
    robot._action_spike_rotation = None
    robot._delta_filter_warn_counts = {"left": 0, "right": 0}
    robot._left_prev_transform = np.zeros(6)
    robot._right_prev_transform = np.zeros(6)
    robot._left_last_gripper_position = 1.0
    robot._right_last_gripper_position = 1.0
    robot._left_one_euro = None
    robot._right_one_euro = None
    robot._compute_delta_pose = lambda current, _previous: current.copy()
    return robot


class OculusMirrorTeleopTest(unittest.TestCase):
    def test_episode_buttons_are_global_and_do_not_release_a_gripper_or_move(self):
        for mirror in (False, True):
            for button in ("A", "B"):
                with self.subTest(mirror=mirror, button=button):
                    core = _make_mirrored_oculus_robot()
                    core._mirror_teleop = mirror
                    core._left_one_euro = Mock()
                    core._right_one_euro = Mock()
                    poses, buttons = _FakeOculusReader().get_transformations_and_buttons()
                    buttons.update(A=button == "A", B=button == "B", Y=False)
                    core._oculus_reader = SimpleNamespace(get_transformations_and_buttons=lambda: (poses, buttons))
                    teleop = OculusTeleop(OculusTeleopConfig(mirror_teleop=mirror))
                    teleop._is_connected = True
                    teleop.oculus_robot = core
                    action = teleop.get_action()
                    self.assertEqual(action["save_episode_requested"], button == "A")
                    self.assertEqual(action["discard_episode_requested"], button == "B")
                    self.assertFalse(action["left_gripper_release_requested"])
                    self.assertFalse(action["right_gripper_release_requested"])
                    self.assertFalse(action["left_grip_pressed"])
                    self.assertFalse(action["right_grip_pressed"])
                    for side in ("left", "right"):
                        for axis in ("x", "y", "z", "rx", "ry", "rz"):
                            self.assertEqual(action[f"{side}_delta_ee_pose.{axis}"], 0)

    def test_mirror_swaps_arms_grippers_and_release_buttons(self):
        robot = _make_mirrored_oculus_robot()

        action = robot.get_action()

        expected = np.array(
            [-10.0, -20.0, 30.0, -40.0, -50.0, 60.0, 0.3,
             -1.0, -2.0, 3.0, -4.0, -5.0, 6.0, 0.8]
        )
        np.testing.assert_allclose(action, expected)
        self.assertTrue(robot._left_grip_pressed)
        self.assertTrue(robot._right_grip_pressed)
        self.assertFalse(robot._left_gripper_release_requested)
        self.assertTrue(robot._right_gripper_release_requested)

    def test_robot_configs_use_their_selected_mirror_mode(self):
        root = Path(__file__).resolve().parents[1]
        record = yaml.safe_load((root / "scripts/config/record_cfg.yaml").read_text())["record"]

        for robot_type in ("nero_dual_arm", "arx_dual_arm", "franka_dual_arm", "flexiv_dual_arm"):
            cfg = deepcopy(record)
            cfg["robot_type"] = robot_type
            teleop_cfg = RecordConfig(cfg).create_teleop_config()
            self.assertTrue(teleop_cfg.mirror_teleop, robot_type)

    def test_flexiv_face_to_face_controls_preserve_installation_axis_signs(self):
        root = Path(__file__).resolve().parents[1]
        config = yaml.safe_load((root / "scripts/config/robots/flexiv_config.yaml").read_text())
        oculus = config["teleop"]["oculus_config"]
        self.assertTrue(oculus["mirror_teleop"])
        self.assertEqual(oculus["left_channel_signs"], [1, 1, 1, -1, -1, 1])
        self.assertEqual(oculus["right_channel_signs"], oculus["left_channel_signs"])


if __name__ == "__main__":
    unittest.main()
