"""Regression coverage for the shared dual-arm mirror teleoperation path."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import unittest

import numpy as np
import yaml

from scripts.core.run_record import RecordConfig
from teleoperators.oculus_teleoperator.oculus.oculus_dual_arm_robot import OculusDualArmRobot


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

    def test_x_embodiment_robot_configs_enable_mirror(self):
        root = Path(__file__).resolve().parents[1]
        record = yaml.safe_load((root / "scripts/config/record_cfg.yaml").read_text())["record"]

        for robot_type in ("nero_dual_arm", "arx_dual_arm", "franka_dual_arm", "flexiv_dual_arm"):
            cfg = deepcopy(record)
            cfg["robot_type"] = robot_type
            teleop_cfg = RecordConfig(cfg).create_teleop_config()
            self.assertTrue(teleop_cfg.mirror_teleop, robot_type)

    def test_flexiv_opposite_controller_signs_match_mirror_mapping(self):
        mirror_signs = OculusDualArmRobot.MIRROR_ACTION_SIGNS
        left_signs = np.array([-1, 1, 1, 1, -1, 1])
        right_signs = np.array([1, -1, 1, -1, 1, 1])

        np.testing.assert_array_equal(mirror_signs * right_signs, left_signs)
        np.testing.assert_array_equal(mirror_signs * left_signs, right_signs)


if __name__ == "__main__":
    unittest.main()
