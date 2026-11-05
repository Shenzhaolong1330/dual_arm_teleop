"""A stalled arm sender must not remain hidden from the recording loop."""
import unittest
from unittest.mock import Mock

from robots.dual_flexiv_rizon4s.config_flexiv import FlexivDualArmConfig
from robots.dual_flexiv_rizon4s.flexiv_dual_arm import FlexivDualArm


class ServoFailureTests(unittest.TestCase):
    def test_right_send_failure_preserves_cause_and_rejects_further_goals(self):
        robot = FlexivDualArm(FlexivDualArmConfig(
            use_cartesian_servo_thread=True, send_arms_parallel=True,
        ))
        robot.is_connected = True
        robot._left_robot, robot._right_robot = Mock(), Mock()
        robot._right_robot.SendCartesianMotionForce.side_effect = RuntimeError("controller not in Cartesian mode")
        with self.assertLogs("robots.dual_flexiv_rizon4s.flexiv_dual_arm", level="ERROR") as logs:
            robot._cartesian_servo_loop()
        self.assertTrue(robot._servo_stop_event.is_set())
        self.assertIn("right", str(robot._servo_error))
        self.assertIn("controller not in Cartesian mode", "\n".join(logs.output))
        robot._right_robot.SendCartesianMotionForce.assert_called_once()
        for operation in (lambda: robot.send_action({"left_delta_ee_pose.x": 0.001}), robot.get_observation):
            with self.assertRaisesRegex(RuntimeError, "right.*controller not in Cartesian mode"):
                operation()
        robot._right_robot.SendCartesianMotionForce.assert_called_once()


if __name__ == "__main__":
    unittest.main()
