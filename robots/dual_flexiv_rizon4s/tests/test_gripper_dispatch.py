"""Low-latency delivery must preserve acknowledged gripper action labels."""

import threading
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

from robots.dual_flexiv_rizon4s.config_flexiv import FlexivDualArmConfig
from robots.dual_flexiv_rizon4s.flexiv_dual_arm import FlexivDualArm


class GripperDispatchTests(unittest.TestCase):
    def test_both_rpc_calls_overlap_and_labels_wait_for_both_results(self):
        robot = FlexivDualArm(FlexivDualArmConfig(use_gripper=True, send_arms_parallel=True))
        barrier = threading.Barrier(2)
        delivered = set()
        def send(side, width):
            barrier.wait(timeout=1)
            delivered.add(side)
            return min(width, 0.05)
        robot._move_gripper_to_width_if_needed = send
        action = {"left_gripper_width": 0.08, "right_gripper_width": 0.03}
        robot._update_gripper_cache(action)
        self.assertEqual(delivered, {"left", "right"})
        self.assertEqual(action, {"left_gripper_width": 0.05, "right_gripper_width": 0.03})

    def test_failed_rpc_does_not_look_like_a_successful_target(self):
        robot = FlexivDualArm(FlexivDualArmConfig(use_gripper=True, send_arms_parallel=True))
        robot._move_gripper_to_width_if_needed = Mock(side_effect=RuntimeError("delivery failed"))
        action = {"left_gripper_width": 0.04}
        with self.assertRaisesRegex(RuntimeError, "delivery failed"):
            robot._update_gripper_cache(action)
        self.assertEqual(robot._left_gripper_cmd, 1.0)

    def test_rate_limit_uses_latest_input_and_returns_only_delivered_targets(self):
        robot = FlexivDualArm(FlexivDualArmConfig(
            use_gripper=True, gripper_max_open=0.1, gripper_min_command_interval_sec=0.3,
        ))
        gripper = Mock()
        gripper.params.return_value = SimpleNamespace(
            min_width=0, max_width=0.1, min_vel=0.001, max_vel=0.2, min_force=-80, max_force=80,
        )
        robot._left_gripper = gripper
        for now, requested, expected in [(0, 0.08, 0.08), (0.1, 0.07, 0.08),
                                         (0.2, 0.06, 0.08), (0.31, 0.04, 0.04)]:
            with patch("robots.dual_flexiv_rizon4s.flexiv_dual_arm.time.monotonic", return_value=now):
                action = {"left_gripper_width": requested}
                robot._update_gripper_cache(action)
            self.assertEqual(action["left_gripper_width"], expected)
        self.assertEqual([c.args[0] for c in gripper.Move.call_args_list], [0.08, 0.04])

    def test_intermediate_widths_are_smoother_but_endpoints_keep_full_speed(self):
        robot = FlexivDualArm(FlexivDualArmConfig(
            use_gripper=True, gripper_max_open=0.1, gripper_continuous_speed=0.05,
        ))
        gripper = Mock()
        gripper.params.return_value = SimpleNamespace(
            min_width=0, max_width=0.1, min_vel=0.001, max_vel=0.2, min_force=-80, max_force=80,
        )
        robot._left_gripper = gripper
        for target in (0.04, 0.06, 0.0, 0.1):
            robot._move_gripper_to_width_if_needed("left", target)
        self.assertEqual([c.args[1] for c in gripper.Move.call_args_list], [0.05, 0.05, 0.2, 0.2])


if __name__ == "__main__":
    unittest.main()
