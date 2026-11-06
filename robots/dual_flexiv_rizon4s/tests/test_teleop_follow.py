"""Exercise configured Grip -> action processors -> Flexiv servo with fake I/O."""

from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import numpy as np
import yaml

from lerobot.processor import make_default_processors
from robots import create_robot_config
from robots.dual_flexiv_rizon4s.flexiv_dual_arm import FlexivDualArm
from scripts.core.run_record import RecordConfig
from teleoperators.oculus_teleoperator import OculusTeleop


class Reader:
    def __init__(self):
        self.poses = {side: np.eye(4) for side in ("l", "r")}
        self.buttons = {"LG": False, "RG": False}

    def get_transformations_and_buttons(self):
        return self.poses, self.buttons

    def stop(self):
        pass


class TeleopFollowTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[3]
        record = yaml.safe_load((root / "scripts/config/record_cfg.yaml").read_text())["record"]
        record["robot_type"] = "flexiv_dual_arm"
        self.rc = RecordConfig(record)
        self.reader = Reader()
        self.teleop = OculusTeleop(self.rc.create_teleop_config())
        with patch("teleoperators.oculus_teleoperator.oculus.oculus_dual_arm_robot.OculusReader",
                   return_value=self.reader):
            self.teleop.connect()
        self.addCleanup(self.teleop.disconnect)

    def test_left_grip_moves_only_right_servo_and_reaches_controller_call(self):
        self.assertTrue(self.teleop.cfg.mirror_teleop)
        self.reader.buttons["LG"] = True
        self.teleop.get_action()  # Capture the clutch reference on press.
        self.reader.poses["l"][0, 3] += 0.01
        raw = self.teleop.get_action()
        left = [raw[f"left_delta_ee_pose.{axis}"] for axis in ("x", "y", "z")]
        right = [raw[f"right_delta_ee_pose.{axis}"] for axis in ("x", "y", "z")]
        np.testing.assert_array_equal(left, 0)
        self.assertGreater(np.linalg.norm(right), 0)

        teleop_processor, robot_processor, _ = make_default_processors()
        action = robot_processor((teleop_processor((raw, {})), {}))
        config = create_robot_config("flexiv_dual_arm", **{
            **self.rc.robot_extra_config, "cameras": {}, "use_gripper": False, "debug": False,
        })
        adapter = FlexivDualArm(config)
        adapter.is_connected = True
        adapter._left_robot, adapter._right_robot = Mock(), Mock()
        pose = np.array([0.4, 0, 0.3, 1, 0, 0, 0], dtype=float)
        adapter._cached_left_pose7 = pose.copy()
        adapter._cached_right_pose7 = pose.copy()
        adapter._reset_servo_targets_from_cached()
        adapter.send_action(action)
        np.testing.assert_array_equal(adapter._servo_left_target_pose7, pose)
        self.assertGreater(np.linalg.norm(adapter._servo_right_target_pose7[:3] - pose[:3]), 0)

        original_send = adapter._send_cartesian_pose_targets
        def send_once(*args):
            original_send(*args)
            adapter._servo_stop_event.set()
        adapter._send_cartesian_pose_targets = send_once
        adapter._cartesian_servo_loop()
        command = adapter._right_robot.SendCartesianMotionForce.call_args.args[0]
        self.assertGreater(np.linalg.norm(np.array(command[:3]) - pose[:3]), 0)
        adapter._left_robot.SendCartesianMotionForce.assert_called_once()

    def test_trigger_alone_does_not_enable_arm_following(self):
        self.teleop.get_action()
        self.reader.buttons.update({"LTr": True, "leftTrig": (1.0,)})
        self.reader.poses["l"][0, 3] += 0.01
        action = self.teleop.get_action()
        self.assertFalse(action["left_grip_pressed"])
        self.assertFalse(action["right_grip_pressed"])
        self.assertEqual(action["right_gripper_width"], 0.0)
        for side in ("left", "right"):
            for axis in ("x", "y", "z", "rx", "ry", "rz"):
                self.assertEqual(action[f"{side}_delta_ee_pose.{axis}"], 0.0)


if __name__ == "__main__":
    unittest.main()
