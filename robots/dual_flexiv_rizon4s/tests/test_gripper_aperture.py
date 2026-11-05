"""Behavioural checks for aperture feedback, endpoint grasp and input loss."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from robots.dual_flexiv_rizon4s.gripper_aperture import GripperApertureController
from robots.dual_flexiv_rizon4s.config_flexiv import FlexivDualArmConfig
from robots.dual_flexiv_rizon4s.flexiv_dual_arm import FlexivDualArm


class ApertureFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.gripper = Mock()
        self.state = SimpleNamespace(width=0.05)
        self.gripper.states.return_value = self.state
        self.controller = GripperApertureController(
            self.gripper, min_width=0, max_width=0.1, speed=0.2, force_limit=20,
        )
        self.addCleanup(self.controller.close)

    def test_intermediate_targets_use_signed_bounded_force_and_accept_aperture(self):
        self.assertEqual(self.controller.update(0.03), 0.03)
        self.assertGreater(self.gripper.Grasp.call_args.args[0], 0)
        self.assertLessEqual(self.gripper.Grasp.call_args.args[0], 5)
        self.controller.update(0.07)
        self.assertLess(self.gripper.Grasp.call_args.args[0], 0)
        self.assertGreaterEqual(self.gripper.Grasp.call_args.args[0], -5)
        self.gripper.Move.assert_not_called()

    def test_unchanged_target_still_updates_feedback_as_width_changes(self):
        with patch("robots.dual_flexiv_rizon4s.gripper_aperture.time.monotonic", return_value=10):
            self.controller.update(0.04)
        first = self.gripper.Grasp.call_args.args[0]
        self.state.width = 0.044
        with patch("robots.dual_flexiv_rizon4s.gripper_aperture.time.monotonic", return_value=10.1):
            self.controller.update(0.04)
        self.assertEqual(self.gripper.Grasp.call_count, 2)
        self.assertLess(self.gripper.Grasp.call_args.args[0], first)

    def test_endpoint_is_single_move_with_full_speed_and_force(self):
        self.controller.update(0.0)
        self.controller.update(0.0)
        self.gripper.Move.assert_called_once_with(0.0, 0.2, 20)
        self.gripper.Grasp.assert_not_called()
        self.controller.update(0.1)
        self.assertEqual(self.gripper.Move.call_count, 2)

    def test_settled_intermediate_aperture_holds_without_force_chatter(self):
        self.controller.update(0.05)
        self.controller.update(0.05)
        self.gripper.Stop.assert_called_once()
        self.gripper.Grasp.assert_not_called()

    def test_input_loss_stops_and_resuming_restarts_from_current_width(self):
        self.controller.update(0.03)
        deadline = self.controller._deadline
        self.controller._check_watchdog(deadline - 0.01)
        self.gripper.Stop.assert_not_called()
        self.controller._check_watchdog(deadline + 0.01)
        self.gripper.Stop.assert_called_once()
        self.controller._check_watchdog(deadline + 1)
        self.gripper.Stop.assert_called_once()
        self.state.width = 0.06
        self.controller.update(0.07)
        self.assertLess(self.gripper.Grasp.call_args.args[0], 0)

    def test_failed_delivery_does_not_accept_goal_and_stops(self):
        self.gripper.Grasp.side_effect = RuntimeError("delivery failed")
        with self.assertRaisesRegex(RuntimeError, "delivery failed"):
            self.controller.update(0.03)
        self.assertIsNone(self.controller._target)
        self.assertIsNone(self.controller._deadline)
        self.gripper.Stop.assert_called_once()

    def test_invalid_measured_width_stops_instead_of_producing_force(self):
        self.state.width = float("nan")
        with self.assertRaisesRegex(RuntimeError, "width feedback"):
            self.controller.update(0.03)
        self.gripper.Grasp.assert_not_called()
        self.gripper.Stop.assert_called_once()

    def test_watchdog_stop_error_prevents_future_commands(self):
        self.controller.update(0.03)
        self.gripper.Stop.side_effect = RuntimeError("stop failed")
        self.controller._check_watchdog(self.controller._deadline + 1)
        with self.assertRaisesRegex(RuntimeError, "watchdog"):
            self.controller.update(0.04)
        self.gripper.Stop.side_effect = None

    def test_adapter_preserves_width_actions_and_manual_move_closes_feedback(self):
        robot = FlexivDualArm(FlexivDualArmConfig(
            use_gripper=True, gripper_control_mode="force_feedback", gripper_max_open=0.1,
            gripper_force=20, gripper_min_command_interval_sec=1.0,
        ))
        self.addCleanup(robot._close_gripper_feedback)
        robot._left_gripper = self.gripper
        self.gripper.params.return_value = SimpleNamespace(
            min_width=0, max_width=0.1, min_vel=0.001, max_vel=0.2,
            min_force=-80, max_force=80,
        )
        action = {"left_gripper_width": 0.03}
        robot._update_gripper_cache(action)
        self.assertEqual(action, {"left_gripper_width": 0.03})
        self.assertIn("left", robot._gripper_feedback)
        robot.move_gripper_width(0.1, side="left", wait=False)
        self.assertFalse(robot._gripper_feedback)
        self.gripper.Move.assert_called_once_with(0.1, 0.2, 20)


if __name__ == "__main__":
    unittest.main()
