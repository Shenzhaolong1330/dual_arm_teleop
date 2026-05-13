#!/usr/bin/env python3
"""Unit-style smoke tests for robots.dual_franka.franka_dual_arm.

Run from the repository root:
    python robots/dual_franka/test_franka_dual_arm.py

These tests do not connect to ROS, cameras, or a real ZeroRPC server. They
patch FrankaDualArmClient with a fake in-memory client and exercise the LeRobot
robot wrapper's behavior.
"""

from __future__ import annotations

import unittest

import numpy as np

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from robots.dual_franka.config_franka import FrankaDualArmConfig
from robots.dual_franka import franka_dual_arm as franka_mod
from robots.dual_franka.franka_dual_arm import FrankaDualArm


class FakeCamera:
    def __init__(self, height: int = 2, width: int = 3) -> None:
        self.height = height
        self.width = width
        self.connected = False
        self.disconnected = False
        self.read_count = 0

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.disconnected = True
        self.connected = False

    def read(self) -> np.ndarray:
        self.read_count += 1
        return np.full((self.height, self.width, 3), self.read_count, dtype=np.uint8)


class FakeFrankaClient:
    instances: list["FakeFrankaClient"] = []

    def __init__(self, ip: str, port: int, timeout: float) -> None:
        self.ip = ip
        self.port = port
        self.timeout = timeout
        self.calls: list[tuple] = []
        self.closed = False
        self.fail_get_full_state = False
        FakeFrankaClient.instances.append(self)

    def ping(self) -> dict:
        self.calls.append(("ping",))
        return {"ok": True}

    def gripper_initialize(self) -> dict:
        self.calls.append(("gripper_initialize",))
        return {"left": {"ok": True}, "right": {"ok": True}}

    def left_gripper_goto(self, **kwargs) -> dict:
        self.calls.append(("left_gripper_goto", kwargs))
        return {"ok": True}

    def right_gripper_goto(self, **kwargs) -> dict:
        self.calls.append(("right_gripper_goto", kwargs))
        return {"ok": True}

    def reset(self) -> dict:
        self.calls.append(("reset",))
        return {}

    def go_home(self, side: str, duration_sec: float | None, rate_hz: float | None) -> dict:
        self.calls.append(("go_home", side, duration_sec, rate_hz))
        return {"ok": True}

    def step(self, action: dict | None = None) -> dict:
        self.calls.append(("step", action))
        return {"ok": True, "observation": self._state()}

    def dual_robot_move_to_ee_pose(self, left_delta, right_delta, delta: bool, wait: bool) -> dict:
        self.calls.append(("dual_robot_move_to_ee_pose", left_delta, right_delta, delta, wait))
        return {"ok": True}

    def get_full_state(self) -> dict:
        self.calls.append(("get_full_state",))
        if self.fail_get_full_state:
            raise RuntimeError("simulated state failure")
        return self._state()

    def close(self) -> None:
        self.calls.append(("close",))
        self.closed = True

    @staticmethod
    def _state() -> dict:
        return {
            "left_arm": {
                "joint_positions": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                "joint_velocities": [0.0] * 7,
                "joint_efforts": [0.0] * 7,
                "end_pose": [0.1, 0.2, 0.3, 0.01, 0.02, 0.03],
                "gripper": 0.25,
            },
            "right_arm": {
                "joint_positions": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
                "joint_velocities": [0.0] * 7,
                "joint_efforts": [0.0] * 7,
                "end_pose": [0.4, 0.5, 0.6, 0.04, 0.05, 0.06],
                "gripper": 0.75,
            },
        }


def make_robot(**kwargs) -> FrankaDualArm:
    config = FrankaDualArmConfig(
        cameras={},
        robot_ip="10.0.0.2",
        robot_port=4242,
        rpc_timeout_sec=3.0,
        **kwargs,
    )
    return FrankaDualArm(config)


def connected_robot(**kwargs) -> tuple[FrankaDualArm, FakeFrankaClient]:
    robot = make_robot(**kwargs)
    fake = FakeFrankaClient("10.0.0.2", 4242, 3.0)
    robot._robot = fake
    robot._is_connected = True
    return robot, fake


class FrankaDualArmTest(unittest.TestCase):
    def setUp(self) -> None:
        FakeFrankaClient.instances.clear()
        self._real_client = franka_mod.FrankaDualArmClient
        franka_mod.FrankaDualArmClient = FakeFrankaClient

    def tearDown(self) -> None:
        franka_mod.FrankaDualArmClient = self._real_client

    def test_connect_disconnect_and_camera_lifecycle(self) -> None:
        robot = make_robot(use_gripper=True, open_grippers_on_connect=True)
        camera = FakeCamera()
        robot.cameras = {"front_image": camera}

        robot.connect()

        fake = FakeFrankaClient.instances[-1]
        self.assertTrue(robot.is_connected)
        self.assertEqual((fake.ip, fake.port, fake.timeout), ("10.0.0.2", 4242, 3.0))
        self.assertIn(("ping",), fake.calls)
        self.assertIn(("gripper_initialize",), fake.calls)
        self.assertIn(
            ("left_gripper_goto", {"width": 0.085, "speed": 0.1, "force": 10.0, "blocking": True}),
            fake.calls,
        )
        self.assertTrue(camera.connected)
        with self.assertRaises(DeviceAlreadyConnectedError):
            robot.connect()

        robot.disconnect()

        self.assertFalse(robot.is_connected)
        self.assertTrue(fake.closed)
        self.assertTrue(camera.disconnected)
        robot.disconnect()

    def test_reset_can_anchor_or_go_home(self) -> None:
        robot, fake = connected_robot(
            use_gripper=False,
            reset_go_home=False,
            go_home_duration_sec=5.0,
            go_home_rate_hz=50.0,
        )
        robot.reset()
        self.assertIn(("reset",), fake.calls)

        robot, fake = connected_robot(
            use_gripper=False,
            reset_go_home=True,
            go_home_duration_sec=5.0,
            go_home_rate_hz=50.0,
        )
        robot.reset()
        self.assertIn(("go_home", "both", 5.0, 50.0), fake.calls)

        disconnected = make_robot(use_gripper=False)
        with self.assertRaises(DeviceNotConnectedError):
            disconnected.reset()

    def test_go_home_moves_both_arms(self) -> None:
        robot, fake = connected_robot(
            use_gripper=False,
            go_home_duration_sec=5.0,
            go_home_rate_hz=50.0,
        )

        robot.go_home()

        self.assertIn(("go_home", "both", 5.0, 50.0), fake.calls)

    def test_reset_requested_routes_through_reset(self) -> None:
        robot, fake = connected_robot(use_gripper=False, reset_go_home=False)
        returned = robot.send_action({"reset_requested": True})
        self.assertTrue(returned["reset_requested"])
        self.assertIn(("reset",), fake.calls)
        self.assertFalse(any(call[0] == "step" for call in fake.calls))

    def test_send_action_builds_single_server_step(self) -> None:
        robot, fake = connected_robot(
            debug=False,
            use_gripper=True,
            max_cartesian_delta=0.04,
            max_rotation_delta=0.25,
        )

        sent_action = robot.send_action(
            {
                "left_delta_ee_pose.x": 0.1,
                "left_delta_ee_pose.y": -0.02,
                "left_delta_ee_pose.z": 0.0,
                "left_delta_ee_pose.rx": 0.3,
                "left_delta_ee_pose.ry": 0.0,
                "left_delta_ee_pose.rz": 0.0,
                "right_delta_ee_pose.x": 0.0,
                "right_delta_ee_pose.y": 0.06,
                "right_delta_ee_pose.z": 0.0,
                "right_delta_ee_pose.rx": 0.0,
                "right_delta_ee_pose.ry": -0.4,
                "right_delta_ee_pose.rz": 0.0,
                "left_gripper_cmd_bin": 0.5,
                "right_gripper_cmd_bin": 1.2,
            }
        )

        step_call = fake.calls[-1]
        self.assertEqual(step_call[0], "step")
        server_action = step_call[1]
        self.assertEqual(server_action["left_arm"]["motion"]["translation"], [0.04, -0.02, 0.0])
        self.assertEqual(server_action["left_arm"]["motion"]["rotation_rotvec"], [0.25, 0.0, 0.0])
        self.assertEqual(server_action["right_arm"]["motion"]["translation"], [0.0, 0.04, 0.0])
        self.assertEqual(server_action["right_arm"]["motion"]["rotation_rotvec"], [0.0, -0.25, 0.0])
        self.assertAlmostEqual(server_action["left_arm"]["gripper"]["width"], 0.0425)
        self.assertAlmostEqual(server_action["right_arm"]["gripper"]["width"], 0.085)
        self.assertEqual(sent_action["left_delta_ee_pose.x"], 0.04)
        self.assertEqual(sent_action["left_delta_ee_pose.rx"], 0.25)
        self.assertEqual(sent_action["right_delta_ee_pose.y"], 0.04)
        self.assertEqual(sent_action["right_delta_ee_pose.ry"], -0.25)
        self.assertEqual(sent_action["left_gripper_cmd_bin"], 0.5)
        self.assertEqual(sent_action["right_gripper_cmd_bin"], 1.0)

    def test_duplicate_gripper_command_is_suppressed(self) -> None:
        robot, fake = connected_robot(debug=False, use_gripper=True)

        action = {"left_gripper_cmd_bin": 0.5, "right_gripper_cmd_bin": 1.0}
        robot.send_action(action)
        robot.send_action(action)

        step_calls = [call for call in fake.calls if call[0] == "step"]
        self.assertEqual(len(step_calls), 1)

    def test_debug_mode_skips_motion_but_keeps_gripper(self) -> None:
        robot, fake = connected_robot(debug=True, use_gripper=True)

        sent_action = robot.send_action(
            {
                "left_delta_ee_pose.x": 0.02,
                "left_gripper_cmd_bin": 0.0,
            }
        )

        step_call = fake.calls[-1]
        self.assertEqual(step_call[0], "step")
        self.assertNotIn("motion", step_call[1]["left_arm"])
        self.assertEqual(step_call[1]["left_arm"]["gripper"]["width"], 0.0)
        self.assertEqual(sent_action["left_delta_ee_pose.x"], 0.0)
        self.assertEqual(sent_action["left_gripper_cmd_bin"], 0.0)

    def test_private_cartesian_compatibility_path(self) -> None:
        robot, fake = connected_robot(debug=False, use_gripper=False)

        robot._send_action_cartesian({"left_delta_ee_pose.x": 0.01})

        self.assertEqual(fake.calls[-1][0], "dual_robot_move_to_ee_pose")
        np.testing.assert_allclose(fake.calls[-1][1], [0.01, 0.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(fake.calls[-1][2], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.assertTrue(fake.calls[-1][3])
        self.assertFalse(fake.calls[-1][4])

    def test_joint_action_is_ignored_with_single_warning_latch(self) -> None:
        robot, fake = connected_robot(debug=False, use_gripper=False)
        action = {f"left_joint_{i + 1}.pos": 0.0 for i in range(7)}

        robot.send_action(action)
        robot.send_action(action)

        self.assertTrue(robot._warned_joint_control)
        self.assertFalse(any(call[0] == "step" for call in fake.calls))

    def test_observation_flattening_camera_and_fallback(self) -> None:
        robot, fake = connected_robot(use_gripper=True)
        camera = FakeCamera()
        robot.cameras = {"front_image": camera}

        obs = robot.get_observation()

        self.assertEqual(obs["left_joint_2.pos"], 0.1)
        self.assertEqual(obs["right_joint_1.pos"], 1.0)
        self.assertEqual(obs["left_ee_pose.x"], 0.1)
        self.assertEqual(obs["right_ee_pose.rz"], 0.06)
        self.assertEqual(obs["left_gripper_state_norm"], 0.25)
        self.assertEqual(obs["right_gripper_state_norm"], 0.75)
        self.assertEqual(obs["left_gripper_cmd_bin"], 0.25)
        self.assertEqual(obs["front_image"].shape, (2, 3, 3))

        robot._cached_rpc_state = None
        fake.fail_get_full_state = True
        fallback = robot.get_observation()
        self.assertIs(fallback, obs)

    def test_observation_reuses_cached_step_state_and_cached_camera_frame(self) -> None:
        robot, fake = connected_robot(use_gripper=True, debug=False)
        camera = FakeCamera()
        robot.cameras = {"front_image": camera}
        cached_frame = np.full((2, 3, 3), 7, dtype=np.uint8)
        robot._latest_frames["front_image"] = cached_frame

        robot.send_action({"left_delta_ee_pose.x": 0.01})
        fake.calls.clear()

        obs = robot.get_observation()

        self.assertEqual(obs["left_joint_2.pos"], 0.1)
        self.assertIs(obs["front_image"], cached_frame)
        self.assertEqual(camera.read_count, 0)
        self.assertFalse(any(call[0] == "get_full_state" for call in fake.calls))

    def test_observation_accepts_nested_rpc_server_state(self) -> None:
        robot, fake = connected_robot(use_gripper=True)
        fake._state = lambda: {  # type: ignore[method-assign]
            "left_arm": {
                "robot_state": {
                    "joint_positions": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                    "eef_pose": {
                        "position": [0.1, 0.2, 0.3],
                        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    },
                },
                "gripper": {
                    "position": 0.0,
                    "open_position": 0.0,
                    "closed_position": 0.8,
                },
            },
            "right_arm": {
                "robot_state": {
                    "joint_positions": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
                    "eef_pose": {
                        "position": [0.4, 0.5, 0.6],
                        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    },
                },
                "gripper": {
                    "position": 0.4,
                    "open_position": 0.0,
                    "closed_position": 0.8,
                },
            },
        }

        obs = robot.get_observation()

        self.assertEqual(obs["left_joint_2.pos"], 0.1)
        self.assertEqual(obs["right_joint_1.pos"], 1.0)
        self.assertEqual(obs["left_ee_pose.x"], 0.1)
        self.assertEqual(obs["right_ee_pose.z"], 0.6)
        self.assertEqual(obs["left_gripper_state_norm"], 1.0)
        self.assertEqual(obs["right_gripper_state_norm"], 0.5)

    def test_features_and_noop_methods(self) -> None:
        robot = make_robot(use_gripper=True, control_mode="oculus")
        robot.cameras = {"front_image": FakeCamera(height=4, width=5)}

        self.assertIn("left_delta_ee_pose.x", robot.action_features)
        self.assertIn("right_gripper_cmd_bin", robot.action_features)
        self.assertEqual(robot.observation_features["front_image"], (4, 5, 3))

        joint_robot = make_robot(use_gripper=False, control_mode="joint")
        self.assertIn("left_joint_1.pos", joint_robot.action_features)
        self.assertNotIn("left_gripper_cmd_bin", joint_robot.action_features)

        robot.is_connected = True
        self.assertTrue(robot.is_calibrated())
        self.assertIsNone(robot.calibrate())
        self.assertIsNone(robot.configure())

    def test_reverse_gripper_converts_open_fraction(self) -> None:
        robot, fake = connected_robot(debug=True, use_gripper=True, gripper_reverse=True)

        robot.send_action({"left_gripper_cmd_bin": 1.0})

        step_call = fake.calls[-1]
        self.assertEqual(step_call[1]["left_arm"]["gripper"]["width"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
