"""Contract tests for the shared Nero/Franka/Flexiv X-embodiment schema."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from robots.dual_agilex_nero.config_nero import NeroDualArmConfig
from robots.dual_agilex_nero.nero_dual_arm import NeroDualArm
from robots.dual_arx_r5.config_arx import ArxDualArmConfig
from robots.dual_arx_r5.arx_dual_arm import ArxDualArm
from robots.dual_flexiv_rizon4s.config_flexiv import FlexivDualArmConfig
from robots.dual_flexiv_rizon4s.flexiv_dual_arm import FlexivDualArm
from robots.dual_franka.config_franka import FrankaDualArmConfig
from robots.dual_franka.franka_dual_arm import FrankaDualArm
from robots.dual_arm_schema import action_feature_names, observation_state_feature_names
from scripts.core.run_record import label_gripper_actions_from_next_observation
from teleoperators.oculus_teleoperator.config_oculus_teleop import OculusTeleopConfig
from teleoperators.oculus_teleoperator.oculus_teleop import OculusTeleop


class FakeCamera:
    height = 2
    width = 3

    def read(self, **_kwargs):
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)


class FakeNeroClient:
    def __init__(self):
        self.widths = {"left": 0.04, "right": 0.07}
        self.commands: list[tuple[str, float]] = []

    def left_robot_get_ee_pose(self):
        return [0.1, 0.2, 0.3, 0.6, 0.5, 0.4]

    def right_robot_get_ee_pose(self):
        return [0.7, 0.8, 0.9, 1.2, 1.1, 1.0]

    def left_gripper_get_state(self):
        return {"width": self.widths["left"]}

    def right_gripper_get_state(self):
        return {"width": self.widths["right"]}

    def left_gripper_goto(self, width, force):
        self.widths["left"] = width
        self.commands.append(("left", width))

    def right_gripper_goto(self, width, force):
        self.widths["right"] = width
        self.commands.append(("right", width))


class FakeArxClient:
    def __init__(self):
        self.grippers = {"left": 0.25, "right": 0.75}
        self.commands: list[tuple[str, float]] = []

    def get_full_state(self):
        return {
            "left_arm": {
                "joint_positions": np.zeros(7),
                "end_pose": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
                "gripper": self.grippers["left"],
            },
            "right_arm": {
                "joint_positions": np.zeros(7),
                "end_pose": np.array([0.7, 0.8, 0.9, 1.0, 1.1, 1.2]),
                "gripper": self.grippers["right"],
            },
        }

    def set_left_gripper(self, command):
        self.grippers["left"] = command
        self.commands.append(("left", command))

    def set_right_gripper(self, command):
        self.grippers["right"] = command
        self.commands.append(("right", command))


class FakeFlexivRobot:
    def states(self):
        return SimpleNamespace(q=[0.0] * 7, tcp_pose=[0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0])


class FakeFlexivGripper:
    def __init__(self, width: float):
        self.width = width
        self.moves: list[tuple[float, float, float]] = []

    def states(self):
        return SimpleNamespace(width=self.width, is_moving=False, force=0.0)

    def params(self):
        return SimpleNamespace(
            min_width=0.0,
            max_width=0.1,
            min_vel=0.01,
            max_vel=0.2,
            min_force=0.1,
            max_force=20.0,
        )

    def Move(self, width, velocity, force):
        self.width = width
        self.moves.append((width, velocity, force))


class FakeFrankaClient:
    def __init__(self):
        self.actions: list[dict] = []

    def step(self, action):
        self.actions.append(action)
        return {"ok": True, "observation": self.get_full_state()}

    @staticmethod
    def get_full_state():
        return {
            "left_arm": {
                "end_pose": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                "gripper": {"open_fraction": 0.25},
            },
            "right_arm": {
                "end_pose": [0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
                "gripper": {"open_fraction": 0.75},
            },
        }


class FakeOculusRobot:
    def get_observations(self):
        return {
            "left_delta_ee_pose.x": 0.01,
            "right_delta_ee_pose.rz": -0.02,
            "left_gripper_cmd": 0.25,
            "right_gripper_cmd": 0.75,
        }


class XEmbodimentSchemaTest(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.cameras = {
            "left_wrist_image": FakeCamera(),
            "right_wrist_image": FakeCamera(),
            "head_image": FakeCamera(),
        }
        self.expected_action = list(action_feature_names())
        self.expected_state = list(observation_state_feature_names())

    def tearDown(self):
        self.tmp_dir.cleanup()

    def _assert_features(self, robot):
        self.assertEqual(list(robot.action_features), self.expected_action)
        self.assertEqual(
            list(robot.observation_features),
            self.expected_state + list(self.cameras),
        )

    def test_nero_schema_observation_and_physical_width_action(self):
        robot = NeroDualArm(NeroDualArmConfig(cameras={}, debug=True, use_gripper=True))
        robot.cameras = self.cameras
        robot._robot = FakeNeroClient()
        robot.is_connected = True

        self._assert_features(robot)
        obs = robot.get_observation()
        self.assertEqual(set(obs), set(robot.observation_features))
        self.assertAlmostEqual(obs["left_ee_pose.rx"], 0.4)
        self.assertAlmostEqual(obs["left_ee_pose.rz"], 0.6)
        self.assertAlmostEqual(obs["right_gripper_width"], 0.07)

        sent = robot.send_action({"left_gripper_width": 0.03, "right_gripper_width": 0.2})
        self.assertAlmostEqual(sent["left_gripper_width"], 0.03)
        self.assertAlmostEqual(sent["right_gripper_width"], 0.1)
        self.assertEqual(robot._robot.commands[-1], ("right", 0.1))

    def test_arx_schema_observation_and_calibrated_width_action(self):
        robot = ArxDualArm(
            ArxDualArmConfig(
                cameras={},
                debug=True,
                use_gripper=True,
                calibration_dir=Path(self.tmp_dir.name),
                gripper_min_width=0.01,
                gripper_max_open=0.09,
            )
        )
        robot.cameras = self.cameras
        robot._client = FakeArxClient()
        robot.is_connected = True

        self._assert_features(robot)
        obs = robot.get_observation()
        self.assertEqual(set(obs), set(robot.observation_features))
        self.assertAlmostEqual(obs["left_gripper_width"], 0.07)
        self.assertAlmostEqual(obs["right_gripper_width"], 0.03)

        sent = robot.send_action({"left_gripper_width": 0.02, "right_gripper_width": 0.2})
        self.assertAlmostEqual(sent["left_gripper_width"], 0.02)
        self.assertAlmostEqual(sent["right_gripper_width"], 0.09)
        self.assertEqual(robot._client.commands[-1], ("right", 0.0))
        self.assertAlmostEqual(robot._client.commands[0][1], 0.875)

    def test_flexiv_schema_observation_and_physical_width_action(self):
        robot = FlexivDualArm(
            FlexivDualArmConfig(
                cameras={},
                debug=True,
                use_gripper=True,
                calibration_dir=Path(self.tmp_dir.name),
            )
        )
        robot.cameras = self.cameras
        robot._left_robot = FakeFlexivRobot()
        robot._right_robot = FakeFlexivRobot()
        robot._left_gripper = FakeFlexivGripper(0.03)
        robot._right_gripper = FakeFlexivGripper(0.08)
        robot.is_connected = True

        self._assert_features(robot)
        obs = robot.get_observation()
        self.assertEqual(set(obs), set(robot.observation_features))
        self.assertAlmostEqual(obs["left_gripper_width"], 0.03)

        sent = robot.send_action({"left_gripper_width": -0.1, "right_gripper_width": 0.06})
        self.assertAlmostEqual(sent["left_gripper_width"], 0.0)
        self.assertAlmostEqual(sent["right_gripper_width"], 0.06)
        self.assertEqual(robot._right_gripper.moves[-1][0], 0.06)

    def test_franka_schema_observation_and_physical_width_action(self):
        robot = FrankaDualArm(FrankaDualArmConfig(cameras={}, debug=True, use_gripper=True))
        robot.cameras = self.cameras
        robot._robot = FakeFrankaClient()
        robot.is_connected = True

        self._assert_features(robot)
        obs = robot.get_observation()
        self.assertEqual(set(obs), set(robot.observation_features))
        self.assertAlmostEqual(obs["left_gripper_width"], 0.25 * 0.085)

        sent = robot.send_action({"left_gripper_width": 0.02, "right_gripper_width": 1.0})
        self.assertAlmostEqual(sent["left_gripper_width"], 0.02)
        self.assertAlmostEqual(sent["right_gripper_width"], 0.085)
        gripper_action = robot._robot.actions[-1]
        self.assertAlmostEqual(gripper_action["right_arm"]["gripper"]["width"], 0.085)

    def test_recording_labels_width_from_the_next_observation(self):
        dataset = SimpleNamespace(
            features={
                "action": {"names": self.expected_action},
                "observation.state": {"names": self.expected_state},
            },
            episode_buffer={
                "size": 2,
                "action": [np.zeros(14, dtype=np.float32), np.zeros(14, dtype=np.float32)],
                "observation.state": [
                    np.array([0.0] * 12 + [0.01, 0.02], dtype=np.float32),
                    np.array([0.0] * 12 + [0.03, 0.04], dtype=np.float32),
                ],
            },
        )

        label_gripper_actions_from_next_observation(dataset)

        np.testing.assert_allclose(dataset.episode_buffer["action"][0][-2:], [0.03, 0.04])
        np.testing.assert_allclose(dataset.episode_buffer["action"][1][-2:], [0.03, 0.04])

    def test_oculus_maps_trigger_to_physical_widths(self):
        teleop = OculusTeleop(
            OculusTeleopConfig(
                use_gripper=True,
                gripper_min_width=0.01,
                gripper_max_open=0.09,
            )
        )
        teleop.oculus_robot = FakeOculusRobot()
        teleop._is_connected = True

        action = teleop.get_action()

        self.assertEqual(list(teleop.action_features), self.expected_action)
        self.assertAlmostEqual(action["left_gripper_width"], 0.03)
        self.assertAlmostEqual(action["right_gripper_width"], 0.07)


if __name__ == "__main__":
    unittest.main()
