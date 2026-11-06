"""Avoid redundant GN01 calibration while retaining a power-on fallback."""

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import yaml

from robots.dual_flexiv_rizon4s.config_flexiv import FlexivDualArmConfig
from robots.dual_flexiv_rizon4s.flexiv_dual_arm import FlexivDualArm


class Gripper:
    def __init__(self, calibrated=True, width=0.03):
        self.calibrated = calibrated
        self.width = width
        self.Init = Mock(side_effect=self.initialize)
        self.Move = Mock(side_effect=self.move)

    def initialize(self):
        self.calibrated = True
        self.width = 0.03

    def move(self, width, velocity, force):
        if self.calibrated:
            self.width = width

    def states(self):
        return SimpleNamespace(width=self.width, is_moving=False, force=0.0)

    def params(self):
        return SimpleNamespace(min_width=0.0, max_width=0.1, min_vel=0.001,
                               max_vel=0.2, min_force=-80.0, max_force=80.0)


class GripperReadinessTests(unittest.TestCase):
    def setUp(self):
        self.robot = FlexivDualArm(FlexivDualArmConfig(
            use_gripper=True, debug=False, initialize_gripper_on_connect=True,
            gripper_init_strategy="if_needed", gripper_max_open=0.1,
        ))
        self.robot._left_gripper = Gripper()
        self.robot._right_gripper = Gripper()
        self.robot._wait_grippers_idle_after_init = Mock()

    def test_calibrated_reconnect_opens_without_init(self):
        self.robot._initialize_grippers()
        for gripper in (self.robot._left_gripper, self.robot._right_gripper):
            gripper.Init.assert_not_called()
            gripper.Move.assert_called_once_with(0.1, 0.2, 10.0)

    def test_already_open_returns_without_moves_or_waits(self):
        for gripper in (self.robot._left_gripper, self.robot._right_gripper):
            gripper.width = 0.0995
        with patch("robots.dual_flexiv_rizon4s.flexiv_dual_arm.time.sleep") as sleep:
            self.robot._initialize_grippers()
        sleep.assert_not_called()
        for gripper in (self.robot._left_gripper, self.robot._right_gripper):
            gripper.Move.assert_not_called()
            gripper.Init.assert_not_called()

    def test_uncalibrated_side_falls_back_without_recalibrating_ready_side(self):
        self.robot._left_gripper.calibrated = False
        # A stale target cache is not evidence that the hardware is ready.
        self.robot._left_gripper_width = 0.1
        now = [0.0]
        with patch("robots.dual_flexiv_rizon4s.flexiv_dual_arm.time.monotonic", lambda: now[0]), \
             patch("robots.dual_flexiv_rizon4s.flexiv_dual_arm.time.sleep",
                   lambda seconds: now.__setitem__(0, now[0] + seconds)):
            self.robot._initialize_grippers()
        self.assertLessEqual(now[0], 2.03)
        self.robot._left_gripper.Init.assert_called_once()
        self.robot._right_gripper.Init.assert_not_called()
        self.robot._wait_grippers_idle_after_init.assert_called_once_with(
            {"left": self.robot._left_gripper},
        )

    def test_delivery_error_is_not_hidden_by_calibration_retry(self):
        self.robot._left_gripper.Move.side_effect = RuntimeError("delivery failed")
        with self.assertRaises(RuntimeError):
            self.robot._initialize_grippers()
        self.robot._left_gripper.Init.assert_not_called()
        self.robot._right_gripper.Init.assert_not_called()

    def test_record_and_readonly_camera_mappings_match(self):
        root = Path(__file__).resolve().parents[3]
        from robots.dual_flexiv_rizon4s.hardware_config import load_hardware_config
        station = load_hardware_config(readonly=True)
        default = yaml.safe_load((root / "scripts/config/robots/flexiv_config.yaml").read_text())
        expected = {"left_wrist_cam_serial": "347622072061",
                    "right_wrist_cam_serial": "347622074732", "head_cam_serial": "344522070241"}
        for config in (station, default):
            for key, serial in expected.items():
                self.assertEqual(config["cameras"][key], serial)
            self.assertEqual(config["robot"]["gripper_init_strategy"], "if_needed")
            self.assertEqual(config["robot"]["gripper_force"], 20.0)
            self.assertEqual(config["robot"]["gripper_control_mode"], "force_feedback")
            self.assertEqual(config["robot"]["gripper_feedback_force_limit"], 5.0)


if __name__ == "__main__":
    unittest.main()
