"""Connection failure regressions using fake robots; no hardware is accessed."""
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from robots.dual_flexiv_rizon4s.config_flexiv import FlexivDualArmConfig
from robots.dual_flexiv_rizon4s.flexiv_dual_arm import FlexivDualArm


class Clock:
    now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class ConnectionTests(unittest.TestCase):
    def setUp(self):
        self.adapter = FlexivDualArm(FlexivDualArmConfig(
            clear_fault_on_connect=False, enable_timeout_sec=0.5,
        ))
        self.robot = SimpleNamespace(
            operational=lambda: False, connected=lambda: True,
            fault=lambda: False, operational_status=lambda: "NOT_ENABLED",
            Enable=Mock(),
        )

    def test_already_operational_does_not_resend_enable(self):
        self.robot.operational = lambda: True
        self.adapter._prepare_robot("left", self.robot)
        self.robot.Enable.assert_not_called()

    def test_lost_reply_preserves_cause_and_never_retries_enable(self):
        failure = RuntimeError("Failed to deliver the request")
        self.robot.Enable.side_effect = failure
        with self.assertRaisesRegex(RuntimeError, "No automatic retry") as caught:
            self.adapter._prepare_robot("left", self.robot)
        self.assertIs(caught.exception.__cause__, failure)
        self.assertIn("NOT_ENABLED", str(caught.exception))
        self.robot.Enable.assert_called_once()

    def test_non_operational_robot_times_out(self):
        clock = Clock()
        with patch("robots.dual_flexiv_rizon4s.flexiv_dual_arm.time.monotonic", clock.monotonic), \
             patch("robots.dual_flexiv_rizon4s.flexiv_dual_arm.time.sleep", clock.sleep):
            with self.assertRaisesRegex(TimeoutError, "left.*0.5s"):
                self.adapter._prepare_robot("left", self.robot)
        self.robot.Enable.assert_called_once()

    def test_disconnection_aborts_wait_without_resending_enable(self):
        self.robot.connected = lambda: False
        with self.assertRaisesRegex(RuntimeError, "lost readiness"):
            self.adapter._prepare_robot("right", self.robot)
        self.robot.Enable.assert_called_once()

    def test_ready_after_brake_release_completes(self):
        clock = Clock()
        self.robot.operational = lambda: clock.now >= 0.2
        with patch("robots.dual_flexiv_rizon4s.flexiv_dual_arm.time.monotonic", clock.monotonic), \
             patch("robots.dual_flexiv_rizon4s.flexiv_dual_arm.time.sleep", clock.sleep):
            self.adapter._prepare_robot("right", self.robot)
        self.robot.Enable.assert_called_once()

    def test_status_read_failure_does_not_hide_enable_error(self):
        self.robot.connected = Mock(side_effect=RuntimeError("status unavailable"))
        self.robot.Enable.side_effect = RuntimeError("request delivery failed")
        with self.assertRaisesRegex(RuntimeError, "request delivery failed"):
            self.adapter._prepare_robot("left", self.robot)

    def test_both_constructors_receive_local_interface_whitelist(self):
        config = FlexivDualArmConfig(
            left_robot_sn="Rizon4s-063173", right_robot_sn="Rizon4s-063278",
            network_interface_whitelist=["192.168.2.100"],
        )
        adapter = FlexivDualArm(config)
        for name in ("_prepare_robot", "_finish_prepare_robot", "_refresh_cached_poses",
                     "_connect_cameras", "_start_cartesian_servo_thread"):
            setattr(adapter, name, Mock())
        factory = Mock()
        with patch.dict(sys.modules, {"flexivrdk": SimpleNamespace(Robot=factory)}):
            adapter.connect()
        self.assertEqual(factory.call_args_list[0].args, ("Rizon4s-063173", ["192.168.2.100"]))
        self.assertEqual(factory.call_args_list[1].args, ("Rizon4s-063278", ["192.168.2.100"]))


if __name__ == '__main__':
    unittest.main()
