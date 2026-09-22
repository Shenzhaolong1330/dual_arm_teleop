"""Contract tests for the shared Nero/Franka/Flexiv X-embodiment schema."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch
from contextlib import ExitStack
import copy
import json
import sys

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
from scripts.core import run_record as recording
from robots.dual_arm_schema import (
    GRIPPER_ACTION_SEMANTICS, GRIPPER_ACTION_SEMANTICS_KEY,
    validate_recording_semantics, width_from_normalized_command,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import write_info

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "tools"))
from scripts.tools import merge_lerobot_datasets as merging
from scripts.tools import preprocess_dataset as cleaning

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
        robot._robot.widths["left"] = 0.015  # Obstruction does not change target.
        robot.get_observation()
        count = len(robot._robot.commands)
        sent = robot.send_action({"left_gripper_width": 0.030000001})
        self.assertEqual(sent["left_gripper_width"], 0.03)
        self.assertEqual(len(robot._robot.commands), count)
        robot._robot.left_gripper_goto = Mock(side_effect=RuntimeError("RPC failed"))
        with self.assertRaisesRegex(RuntimeError, "RPC failed"):
            robot.send_action({"left_gripper_width": 0.0})

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
        robot._client.grippers["left"] = 0.7  # Feedback differs from held target.
        robot.get_observation()
        count = len(robot._client.commands)
        sent = robot.send_action({"left_gripper_width": 0.020000001})
        self.assertEqual(sent["left_gripper_width"], 0.02)
        self.assertEqual(len(robot._client.commands), count)
        robot.config.gripper_reverse = True
        robot.send_action({"left_gripper_cmd": 0.25})
        self.assertAlmostEqual(robot._client.commands[-1][1], 0.75)

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
        robot._left_gripper.width = 0.015
        robot.get_observation()
        sent = robot.send_action({"left_gripper_width": 0.00000001})
        self.assertEqual(sent["left_gripper_width"], 0.0)
        self.assertEqual(len(robot._left_gripper.moves), 1)
        robot._right_gripper_params["max_width"] = 0.05
        sent = robot.send_action({"right_gripper_width": 0.09})
        self.assertEqual(sent["right_gripper_width"], 0.05)
        self.assertEqual(robot._right_gripper.moves[-1][0], 0.05)
        robot._left_gripper.Move = Mock(side_effect=RuntimeError("Move failed"))
        with self.assertRaisesRegex(RuntimeError, "Move failed"):
            robot.send_action({"left_gripper_width": 0.04})

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
        count = len(robot._robot.actions)
        sent = robot.send_action({"left_gripper_width": 0.020000001})
        self.assertAlmostEqual(sent["left_gripper_width"], 0.02)
        self.assertEqual(len(robot._robot.actions), count)
        robot._robot.step = Mock(side_effect=RuntimeError("step failed"))
        with self.assertRaisesRegex(RuntimeError, "step failed"):
            robot.send_action({"left_gripper_width": 0.0})

    def test_all_adapters_convert_legacy_commands_and_reject_nonfinite(self):
        cases = (
            (NeroDualArm, NeroDualArmConfig),
            (ArxDualArm, ArxDualArmConfig),
            (FlexivDualArm, FlexivDualArmConfig),
            (FrankaDualArm, FrankaDualArmConfig),
        )
        for robot_cls, cfg_cls in cases:
            with self.subTest(robot=robot_cls.__name__):
                robot = robot_cls(cfg_cls(cameras={}, debug=True, use_gripper=True,
                                          gripper_reverse=True, gripper_max_open=0.085,
                                          calibration_dir=Path(self.tmp_dir.name)))
                robot.is_connected = True
                if robot_cls is NeroDualArm:
                    robot._robot = FakeNeroClient()
                elif robot_cls is ArxDualArm:
                    robot._client = FakeArxClient()
                elif robot_cls is FlexivDualArm:
                    robot._left_robot = FakeFlexivRobot()
                    robot._right_robot = FakeFlexivRobot()
                    robot._left_gripper = FakeFlexivGripper(0.015)
                    robot._right_gripper = FakeFlexivGripper(0.025)
                else:
                    robot._robot = FakeFrankaClient()
                sent = robot.send_action({"left_gripper_cmd": 1.0, "right_gripper_cmd": 0.0})
                self.assertAlmostEqual(sent["left_gripper_width"], 0.0)
                self.assertAlmostEqual(sent["right_gripper_width"], 0.085)
                for value in (float("nan"), float("inf"), -float("inf")):
                    with self.assertRaises(ValueError):
                        robot.send_action({"left_gripper_width": value})
                    with self.assertRaises(ValueError):
                        robot.send_action({"left_gripper_cmd": value})
                if robot_cls is ArxDualArm:
                    robot.config.debug = False
                    with patch.object(robot, "_smooth_reset_arms") as home:
                        robot.reset()
                        home.assert_called_once_with(True, True)
                    robot._client.set_left_gripper = Mock(side_effect=RuntimeError("ARX command failed"))
                    with self.assertRaisesRegex(RuntimeError, "ARX command failed"):
                        robot.send_action({"left_gripper_width": 0.0})
                elif robot_cls is FrankaDualArm:
                    robot._robot.step = Mock(return_value={"ok": False})
                    with self.assertRaisesRegex(RuntimeError, "rejected"):
                        robot.send_action({"left_gripper_width": 0.04})

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


class RecordingCommandTargetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.features = {
            "action": {"dtype": "float32", "shape": (14,), "names": list(action_feature_names())},
            "observation.state": {"dtype": "float32", "shape": (14,), "names": list(observation_state_feature_names())},
        }
        self.action = dict.fromkeys(action_feature_names(), 0.0)
        self.action["left_delta_ee_pose.x"] = 0.123
        self.action["right_gripper_width"] = 0.07
        self.obs = dict.fromkeys(observation_state_feature_names(), 0.0)
        self.obs.update(left_gripper_width=0.015, right_gripper_width=0.025)
        self.events = dict(exit_early=False, stop_recording=False, rerecord_episode=False)

    def dataset(self, name, marked=True, mixed=False):
        features = copy.deepcopy(self.features)
        if mixed:
            for key in ("policy_action", "expert_action", "sent_action"):
                features[key] = copy.deepcopy(features["action"])
            for key in ("action_source", "frame_role", "expert_action_missing", "success_policy"):
                features[key] = {"dtype": "string", "shape": (1,), "names": None}
            for key in ("is_expert", "expert_label_complete", "success", "success_inferred_from_recorded_episode"):
                features[key] = {"dtype": "bool", "shape": (1,), "names": None}
            features["intervention_segment_id"] = {"dtype": "int64", "shape": (1,), "names": None}
        ds = LeRobotDataset.create(name, fps=10, features=features, root=self.root / name,
                                  robot_type="mock", use_videos=False)
        if marked:
            ds.meta.info[GRIPPER_ACTION_SEMANTICS_KEY] = GRIPPER_ACTION_SEMANTICS
            write_info(ds.meta.info, ds.root)
        return ds

    def robot(self, frames=2, returned=None):
        robot = Mock()
        robot.robot_type = "mock"
        robot.get_observation.return_value = self.obs
        def send(action):
            if robot.send_action.call_count >= frames:
                self.events["exit_early"] = True
            result = dict(action) if returned is None else returned
            if returned is None:
                result["right_gripper_width"] = 0.06  # Hardware clamp.
                result["left_delta_ee_pose.x"] = 0.0
            return result
        robot.send_action.side_effect = send
        return robot

    def loop(self, ds, robot, mode="teleop", actions=None):
        teleop = Mock()
        teleop.get_action.side_effect = actions
        teleop.get_action.return_value = (dict(self.action, **{"left_delta_ee_pose.x": 0.0})
                                          if mode == "mixed" else self.action)
        policy = Mock(config=SimpleNamespace(device="cpu", use_amp=False))
        common = dict(robot=robot, events=self.events, dataset=ds, fps=10,
                      teleop_action_processor=lambda pair: dict(pair[0]),
                      robot_action_processor=lambda pair: dict(pair[0]),
                      robot_observation_processor=lambda obs: dict(obs),
                      control_time_s=10, single_task="blocked gripper", display_data=False)
        with patch.object(recording, "busy_wait"), patch.object(recording, "predict_action", return_value=self.action), \
             patch.object(recording, "make_robot_action", side_effect=lambda action, features: dict(action)):
            if mode == "mixed":
                return recording.run_mix_record_loop(**common, teleop=teleop, policy=policy,
                                                      preprocessor=Mock(), postprocessor=Mock())
            return recording.record_loop_with_recorded_resets(
                **common, teleop=teleop if mode == "teleop" else None,
                policy=policy if mode == "policy" else None,
                preprocessor=Mock(), postprocessor=Mock())

    def test_saved_teleop_policy_and_mixed_targets_including_terminal_frame(self):
        for mode in ("teleop", "policy", "mixed"):
            with self.subTest(mode=mode):
                self.events["exit_early"] = False
                ds = self.dataset(mode, mixed=mode == "mixed")
                self.loop(ds, self.robot(), mode)
                ds.save_episode()
                ds.finalize()
                loaded = LeRobotDataset(mode, root=ds.root)
                self.assertEqual(len(loaded), 2)
                for i in range(2):
                    row = loaded[i]
                    np.testing.assert_allclose(row["action"][-2:], [0.0, 0.06])
                    np.testing.assert_allclose(row["observation.state"][-2:], [0.015, 0.025])
                    if mode == "mixed":
                        np.testing.assert_array_equal(row["action"][-2:], row["sent_action"][-2:])
                        self.assertAlmostEqual(float(row["policy_action"][-1]), 0.07)
                        self.assertAlmostEqual(float(row["expert_action"][-1]), 0.07)
                    else:
                        self.assertAlmostEqual(float(row["action"][0]), 0.123)

    def test_missing_or_nonfinite_return_aborts_without_recording(self):
        for returned in ({}, {"left_gripper_width": float("nan"), "right_gripper_width": 0.05}):
            for mode in ("teleop", "policy", "mixed"):
                self.events["exit_early"] = False
                ds = SimpleNamespace(features=self.features, fps=10, add_frame=Mock())
                with self.assertRaises(ValueError):
                    self.loop(ds, self.robot(returned=returned), mode)
                ds.add_frame.assert_not_called()
        for bad in (float("nan"), float("inf"), -float("inf")):
            with self.assertRaises(ValueError):
                width_from_normalized_command(bad, 0, 0.085)
        self.assertAlmostEqual(width_from_normalized_command(0.25, 0.01, 0.09, reverse=True), 0.07)

    def test_reset_stops_before_motion_and_held_press_is_not_repeated(self):
        for mode in ("teleop", "mixed"):
            self.events = dict(exit_early=False, stop_recording=False, rerecord_episode=False)
            ds = self.dataset("reset_" + mode, mixed=mode == "mixed")
            robot = self.robot(frames=100)
            reset = dict(self.action, reset_requested=True)
            self.loop(ds, robot, mode, actions=[self.action, reset])
            self.assertTrue(self.events["reset_episode"])
            self.assertEqual(ds.episode_buffer["size"], 1)
            self.assertEqual(robot.send_action.call_count, 1)
            robot.reset.assert_not_called()
            ds.save_episode()
            self.events["reset_episode"] = False
            self.loop(ds, robot, mode, actions=[reset, self.action, reset])
            self.assertEqual(ds.episode_buffer["size"], 1)
            self.assertEqual(robot.send_action.call_count, 2)
            ds.save_episode()
            ds.finalize()
            loaded = LeRobotDataset("reset_" + mode, root=ds.root)
            self.assertEqual([int(loaded[i]["episode_index"]) for i in range(2)], [0, 1])


    def test_full_recording_reset_boundaries_empty_segments_and_success(self):
        for mode in (recording.RUN_MODE_RECORD, recording.RUN_MODE_POLICY, recording.RUN_MODE_MIX):
            with self.subTest(mode=mode), ExitStack() as stack:
                root = self.root / mode
                self.events = dict(exit_early=False, stop_recording=False, rerecord_episode=False)
                robot = Mock()
                robot.name = robot.robot_type = "mock"
                robot.cameras = {}
                robot.action_features = dict.fromkeys(action_feature_names(), float)
                robot.observation_features = dict.fromkeys(observation_state_feature_names(), float)
                robot.get_observation.return_value = self.obs
                robot.send_action.side_effect = lambda action: dict(action)
                reset = dict(self.action, reset_requested=True)
                sequence = [reset, reset, self.action, reset, reset, self.action, reset]
                teleop = Mock()
                teleop.get_action.side_effect = sequence
                cfg = SimpleNamespace(
                    dataset_name=mode, dataset_root=root, run_mode=mode, fps=10,
                    left_wrist_cam_serial="fake", right_wrist_cam_serial="fake", head_cam_serial="fake",
                    cam_width=64, cam_height=64, robot_ip="none", robot_port=0,
                    debug=True, use_gripper=True, gripper_max_open=0.085, gripper_force=10,
                    gripper_speed=0.1, close_threshold=0.5, gripper_reverse=False,
                    control_mode="oculus", robot_extra_config={}, robot_type="mock", resume=False,
                    create_teleop_config=lambda: None, save_meta_period=1, display=False,
                    num_episodes=2, episode_time_sec=10, reset_time_sec=10, task_description="test",
                    reset_on_finish=True, disconnect_on_finish=True, push_to_hub=False,
                    success_policy=recording.SUCCESS_POLICY_RECORDED_IS_SUCCESS,
                    annotate_success=False, dual_arm=True,
                    policy=SimpleNamespace(device="cpu", use_amp=False, pretrained_path=None),
                )
                policy = Mock(config=cfg.policy)
                patch_values = {
                    "create_robot": robot, "create_robot_config": None, "OculusTeleop": teleop,
                    "init_keyboard_listener": (None, self.events),
                    "make_default_processors": (lambda pair: dict(pair[0]), lambda pair: dict(pair[0]), dict),
                    "make_policy": policy, "make_pre_post_processors": (Mock(), Mock()),
                    "_wait_for_next_episode_with_teleop": False,
                }
                for name, value in patch_values.items():
                    stack.enter_context(patch.object(recording, name, return_value=value))
                stack.enter_context(patch.object(recording, "update_dataset_info"))
                cleanup = stack.enter_context(patch.object(recording, "handle_incomplete_dataset"))
                stack.enter_context(patch.object(recording, "busy_wait"))
                stack.enter_context(patch.object(recording, "make_robot_action", side_effect=lambda a, f: dict(a)))
                stack.enter_context(patch.object(recording, "predict_action",
                                                side_effect=sequence if mode == recording.RUN_MODE_POLICY else None,
                                                return_value=self.action))
                saved_before_reset = []
                def reset_robot():
                    # Recording must persist the segment before any reset motion starts.
                    info = json.loads((root / "meta/info.json").read_text())
                    saved_before_reset.append(info["total_episodes"])
                robot.reset.side_effect = reset_robot
                recording.run_record(cfg)
                cleanup.assert_not_called()
                self.assertEqual(saved_before_reset, [0, 1, 2])
                self.assertEqual(robot.send_action.call_count, 2)
                recording._wait_for_next_episode_with_teleop.assert_called()
                self.assertEqual(recording._wait_for_next_episode_with_teleop.call_count, 2)
                loaded = LeRobotDataset(mode, root=root)
                self.assertEqual(loaded.num_episodes, 2)
                self.assertEqual(len(loaded), 2)
                self.assertEqual(loaded.meta.info[GRIPPER_ACTION_SEMANTICS_KEY], GRIPPER_ACTION_SEMANTICS)
                for row in loaded:
                    if mode == recording.RUN_MODE_MIX:
                        self.assertFalse(bool(row["success"]))
                        self.assertFalse(bool(row["success_inferred_from_recorded_episode"]))
                    else:
                        self.assertEqual(float(row["action"][-2]), 0.0)
                # Resume guard runs before connecting hardware and never retags old data.
                info = dict(loaded.meta.info)
                info.pop(GRIPPER_ACTION_SEMANTICS_KEY)
                write_info(info, root)
                before = (root / "meta/info.json").read_bytes()
                cfg.resume = True
                robot.connect.reset_mock()
                with self.assertRaises(SystemExit):
                    recording.run_record(cfg)
                robot.connect.assert_not_called()
                cleanup.assert_not_called()
                self.assertEqual((root / "meta/info.json").read_bytes(), before)

    def test_wait_ignores_held_reset_until_right_arrow(self):
        robot = self.robot()
        robot.action_features = dict.fromkeys(action_feature_names(), float)
        teleop = Mock()
        teleop.get_action.return_value = dict(self.action, reset_requested=True)
        def right_arrow(_):
            self.events["exit_early"] = True
        with patch.object(recording, "busy_wait", side_effect=right_arrow):
            moved = recording._wait_for_next_episode_with_teleop(
                robot=robot, teleop=teleop, events=self.events, fps=10,
                teleop_action_processor=lambda pair: pair[0],
                robot_action_processor=lambda pair: pair[0],
                robot_observation_processor=dict, display_data=False)
        self.assertFalse(moved)
        self.assertTrue(self.events["metric_reset_held"])
        robot.send_action.assert_not_called()
        robot.reset.assert_not_called()

    def fill(self, ds):
        for _ in range(3):
            ds.add_frame({"action": np.array(list(self.action.values()), dtype=np.float32),
                          "observation.state": np.array(list(self.obs.values()), dtype=np.float32), "task": "test"})
        ds.save_episode()
        ds.finalize()
        return ds

    def merge_config(self, sources, name="merged", fast=True):
        return {"source": {"parent_dir": str(self.root), "datasets": sources},
                "output": {"repo_id": name, "root": str(self.root / name), "overwrite": True},
                "fast_merge": fast}

    def test_resume_rejects_unmarked_and_preserves_it(self):
        old = self.fill(self.dataset("old", marked=False))
        before = (old.root / "meta/info.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "Cannot resume"):
            validate_recording_semantics(old.meta.info, old.features)
        self.assertEqual((old.root / "meta/info.json").read_bytes(), before)
        validate_recording_semantics({GRIPPER_ACTION_SEMANTICS_KEY: GRIPPER_ACTION_SEMANTICS}, self.features)

    def test_both_merge_modes_reject_before_overwrite_or_create(self):
        self.fill(self.dataset("new"))
        self.fill(self.dataset("old", marked=False))
        for fast in (False, True):
            output = self.root / "merged"
            output.mkdir(exist_ok=True)
            sentinel = output / "keep.txt"
            sentinel.write_text("keep")
            cfg = self.merge_config(["new", "old"], fast=fast)
            with self.assertRaisesRegex(ValueError, "gripper_action_semantics"):
                merging.merge_lerobot_datasets(cfg)
            self.assertEqual(sentinel.read_text(), "keep")
            cfg["output"]["root"] = str(self.root / "absent")
            with self.assertRaises(ValueError):
                merging.merge_lerobot_datasets(cfg)
            self.assertFalse((self.root / "absent").exists())

    def test_fast_clean_preserves_video_dataset_labels_and_marker(self):
        from scripts.tools import fast_preprocess_dataset as fast_cleaning
        features = copy.deepcopy(self.features)
        features["observation.images.head_image"] = {
            "dtype": "video", "shape": (64, 64, 3), "names": ["height", "width", "channels"]}
        source = LeRobotDataset.create("video", fps=10, features=features, root=self.root / "video",
                                       robot_type="mock", use_videos=True, image_writer_threads=1)
        source.meta.info[GRIPPER_ACTION_SEMANTICS_KEY] = GRIPPER_ACTION_SEMANTICS
        write_info(source.meta.info, source.root)
        targets = []
        for i in range(4):
            action = np.array(list(self.action.values()), dtype=np.float32)
            action[-2] = 0.08 if i == 0 else 0.0
            targets.append(action[-2:].copy())
            source.add_frame({"action": action, "observation.state": np.array(list(self.obs.values()), dtype=np.float32),
                              "observation.images.head_image": np.full((64, 64, 3), i * 30, dtype=np.uint8), "task": "test"})
        source.save_episode()
        source.finalize()
        cfg = {"source": {"root": str(source.root)},
               "output": {"root": str(self.root / "fast_clean")},
               "static_trim": {"enabled": False},
               "action_smoothing": {"enabled": True, "smooth_gripper": False},
               "fast_stream": {"video_workers": 1, "ffmpeg_threads": 1}}
        fast_cleaning._materialize_job(cfg, max_episodes=None, dry_run=False)
        info_path = self.root / "fast_clean/meta/info.json"
        before = info_path.read_bytes()
        self.assertEqual(json.loads(before)[GRIPPER_ACTION_SEMANTICS_KEY], GRIPPER_ACTION_SEMANTICS)
        import pyarrow.parquet as pq
        tables = list((self.root / "fast_clean/data").rglob("*.parquet"))
        actions = np.asarray(pq.read_table(tables[0])["action"].to_pylist())
        np.testing.assert_allclose(actions[:, -2:], targets)
        cfg["action_smoothing"]["smooth_gripper"] = True
        cfg["output"]["overwrite"] = True
        with self.assertRaisesRegex(ValueError, "incompatible"):
            fast_cleaning._materialize_job(cfg, max_episodes=None, dry_run=False)
        self.assertEqual(info_path.read_bytes(), before)
        self.assertFalse((self.root / "fast_clean.fast-staging").exists())

    def test_clean_and_both_merges_keep_marker_and_targets(self):
        source = self.fill(self.dataset("source"))
        cfg = {"source": {"repo_id": "source", "root": str(source.root)},
               "output": {"repo_id": "clean", "root": str(self.root / "clean")},
               "static_trim": {"enabled": False},
               "action_smoothing": {"enabled": True, "smooth_cartesian": True, "smooth_gripper": False}}
        cleaning.preprocess_dataset(cfg)
        for fast in (False, True):
            name = "merge_" + str(fast)
            summary = merging.merge_lerobot_datasets(self.merge_config(["source", "clean"], name, fast))
            loaded = LeRobotDataset(name, root=self.root / name)
            self.assertEqual(loaded.meta.info[GRIPPER_ACTION_SEMANTICS_KEY], GRIPPER_ACTION_SEMANTICS)
            self.assertEqual(len(loaded), 6)
            self.assertEqual(summary["merge_mode"], "fast_aggregate" if fast else "frame_rewrite")
            for row in loaded:
                np.testing.assert_allclose(row["action"][-2:], [0.0, 0.07])
        cfg["action_smoothing"]["smooth_gripper"] = True
        cfg["output"]["overwrite"] = True
        before = (self.root / "clean/meta/info.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "incompatible"):
            cleaning.preprocess_dataset(cfg)
        self.assertEqual((self.root / "clean/meta/info.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
