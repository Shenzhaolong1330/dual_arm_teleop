"""Reset regressions without hardware or real sleeps."""

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import yaml

from robots.dual_flexiv_rizon4s.config_flexiv import FlexivDualArmConfig
from robots.dual_flexiv_rizon4s.flexiv_dual_arm import FlexivDualArm
from robots.dual_flexiv_rizon4s.reset_station import build_reset_config


class FakeGripper:
    def __init__(self, width=0.0, move_works=True):
        self.width = width
        self.move_works = move_works
        self.initialized = False
        self.moves = []

    def Init(self):
        self.initialized = True

    def Move(self, width, velocity, force):
        if not self.initialized:
            raise AssertionError("GN01 Move occurred before Init")
        self.moves.append((width, velocity, force))
        if self.move_works:
            self.width = width

    def states(self):
        return SimpleNamespace(width=self.width, is_moving=False, force=0.0)

    def params(self):
        return SimpleNamespace(min_width=0.0, max_width=0.1, min_vel=0.001,
                               max_vel=0.2, min_force=-80.0, max_force=80.0)


class Clock:
    now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class ResetGripperTests(unittest.TestCase):
    def setUp(self):
        self.station = Path(__file__).resolve().parents[1] / "station_deepcybo_5090.yaml"

    def test_home_is_half_original_speed_without_extended_timeout(self):
        robot = build_reset_config(self.station)["record"]["robot"]
        self.assertEqual(robot["home_joint_max_vel"], 0.5)
        self.assertEqual(robot["home_joint_max_acc"], 2.0)
        self.assertEqual(robot["home_joint_timeout_sec"], 20.0)
        self.assertEqual(robot["gripper_init_timeout_sec"], 10.0)

    def test_gn01_init_enabled_only_for_explicit_reset(self):
        recording = yaml.safe_load(self.station.read_text())["robot"]
        reset = build_reset_config(self.station)["record"]["robot"]
        self.assertFalse(recording["initialize_gripper_on_connect"])
        self.assertTrue(reset["initialize_gripper_on_connect"])
        self.assertFalse(reset["go_home_on_connect"])
        self.assertTrue(reset["reset_go_home"])
        self.assertFalse(reset["use_cartesian_servo_thread"])
        self.assertFalse(reset["switch_cartesian_mode_on_connect"])

    def test_reset_initializes_both_grippers_before_open(self):
        config = build_reset_config(self.station)["record"]["robot"]
        robot = FlexivDualArm(FlexivDualArmConfig(**config, cameras={}, debug=False))
        robot.is_connected = True
        robot._left_gripper = FakeGripper()
        robot._right_gripper = FakeGripper()
        robot._execute_home_for_sides = Mock()
        robot._stop_arms_for_sides = Mock()
        robot._prepare_grippers = Mock()
        robot._wait_grippers_idle_after_init = Mock()
        robot._refresh_cached_poses = Mock()
        with patch("robots.dual_flexiv_rizon4s.flexiv_dual_arm.time.sleep") as sleep:
            robot.reset()
        for gripper in (robot._left_gripper, robot._right_gripper):
            self.assertTrue(gripper.initialized)
            self.assertEqual(len(gripper.moves), 1)
            self.assertAlmostEqual(gripper.moves[0][0], 0.1)
        sleep.assert_not_called()

    def test_gripper_at_target_returns_without_extra_settle(self):
        robot = FlexivDualArm(FlexivDualArmConfig(use_gripper=True, gripper_init_settle_sec=0.5))
        gripper = FakeGripper(width=0.1)
        with patch("robots.dual_flexiv_rizon4s.flexiv_dual_arm.time.sleep") as sleep:
            robot._wait_grippers_width({"left": (gripper, 0.1, 0.2, 10.0)})
        sleep.assert_not_called()
        self.assertEqual(gripper.moves, [])

    def test_stalled_gripper_does_not_blindly_resend_move(self):
        robot = FlexivDualArm(FlexivDualArmConfig(use_gripper=True, gripper_init_timeout_sec=0.1))
        gripper = FakeGripper(move_works=False)
        clock = Clock()
        with patch("robots.dual_flexiv_rizon4s.flexiv_dual_arm.time.monotonic", clock.monotonic), \
             patch("robots.dual_flexiv_rizon4s.flexiv_dual_arm.time.sleep", clock.sleep):
            with self.assertRaises(TimeoutError):
                robot._wait_grippers_width({"left": (gripper, 0.1, 0.2, 10.0)})
        self.assertEqual(gripper.moves, [])


if __name__ == "__main__":
    unittest.main()
