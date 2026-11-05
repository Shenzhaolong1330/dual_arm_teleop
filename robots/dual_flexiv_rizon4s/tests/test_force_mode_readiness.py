"""Force-mode prerequisites with fake controllers; no hardware or real waits."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from robots.dual_flexiv_rizon4s.config_flexiv import FlexivDualArmConfig
from robots.dual_flexiv_rizon4s.flexiv_dual_arm import FlexivDualArm


class ForceModeReadinessTests(unittest.TestCase):
    def make_adapter(self, **overrides):
        config = dict(debug=False, switch_cartesian_mode_on_connect=True,
                      zero_ft_sensor_on_connect=False)
        config.update(overrides)
        adapter = FlexivDualArm(FlexivDualArmConfig(**config))
        adapter._flexivrdk = SimpleNamespace(Mode=SimpleNamespace(
            NRT_PRIMITIVE_EXECUTION="primitive",
            NRT_CARTESIAN_MOTION_FORCE="cartesian",
        ))
        return adapter

    def test_zero_finishes_before_cartesian_mode_even_with_flag_disabled(self):
        adapter = self.make_adapter()
        controller = Mock()
        controller.primitive_states.return_value = {"terminated": True}
        adapter._finish_prepare_robot("left", controller)
        self.assertEqual(controller.method_calls, [
            unittest.mock.call.SwitchMode("primitive"),
            unittest.mock.call.ExecutePrimitive("ZeroFTSensor", {}),
            unittest.mock.call.primitive_states(),
            unittest.mock.call.SwitchMode("cartesian"),
            unittest.mock.call.SetForceControlAxis([False] * 6),
        ])

    def test_zero_failure_prevents_entering_cartesian_mode(self):
        adapter = self.make_adapter()
        adapter._zero_ft_sensor = Mock(side_effect=RuntimeError("zero failed"))
        controller = Mock()
        with self.assertRaisesRegex(RuntimeError, "zero failed"):
            adapter._finish_prepare_robot("left", controller)
        controller.SwitchMode.assert_not_called()

    def test_readonly_and_debug_do_not_zero_or_switch_modes(self):
        for options in ({"switch_cartesian_mode_on_connect": False},
                        {"debug": True, "zero_ft_sensor_on_connect": True}):
            with self.subTest(options=options):
                adapter = self.make_adapter(**options)
                controller = Mock()
                adapter._finish_prepare_robot("left", controller)
                self.assertEqual(controller.method_calls, [])

    def test_reset_zeroes_both_arms_after_gripper_setup(self):
        adapter = self.make_adapter(reset_go_home=True)
        adapter.is_connected = True
        adapter._left_robot = Mock()
        adapter._right_robot = Mock()
        sequence = Mock()
        for method in ("_stop_cartesian_servo_thread", "_execute_home_for_sides",
                       "_stop_arms_for_sides", "_prepare_grippers", "_initialize_grippers",
                       "move_gripper_width", "_finish_prepare_robot",
                       "_refresh_cached_poses", "_start_cartesian_servo_thread"):
            stub = Mock()
            setattr(adapter, method, stub)
            sequence.attach_mock(stub, method)
        adapter.reset()
        names = [call[0] for call in sequence.mock_calls]
        self.assertLess(names.index("move_gripper_width"), names.index("_finish_prepare_robot"))
        self.assertEqual(adapter._finish_prepare_robot.call_args_list, [
            unittest.mock.call("left", adapter._left_robot),
            unittest.mock.call("right", adapter._right_robot),
        ])


if __name__ == "__main__":
    unittest.main()
