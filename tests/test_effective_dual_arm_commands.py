"""Hardware-free regression tests for world poses and recorded commands."""

from __future__ import annotations

import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from robots.dual_arm_schema import (
    ACTION_SEMANTICS, ACTION_SEMANTICS_KEY, AXES,
    action_feature_names, observation_state_feature_names,
    command_target_action, recording_semantics, validate_recording_semantics,
)
from robots.dual_agilex_nero.config_nero import NeroDualArmConfig
from robots.dual_agilex_nero.nero_dual_arm import NeroDualArm
from robots.dual_franka.config_franka import FrankaDualArmConfig
from robots.dual_franka.franka_dual_arm import FrankaDualArm
from robots.dual_flexiv_rizon4s.config_flexiv import FlexivDualArmConfig
from robots.dual_flexiv_rizon4s.flexiv_dual_arm import FlexivDualArm
from scripts.core import run_record as recording
from scripts.utils.dataset_utils import validate_merge_semantics, validate_gripper_smoothing


def vector(action, side="left"):
    return np.array([action[f"{side}_delta_ee_pose.{axis}"] for axis in AXES])


def action(left=None, right=None):
    result = dict.fromkeys(action_feature_names(), 0.0)
    for side, delta in (("left", left), ("right", right)):
        if delta is not None:
            result.update({f"{side}_delta_ee_pose.{axis}": float(value) for axis, value in zip(AXES, delta, strict=True)})
    result.update(left_gripper_width=0.04, right_gripper_width=0.04)
    return result


class Gripper:
    def __init__(self):
        self.moves = []

    def params(self):
        return SimpleNamespace(min_width=0.0, max_width=0.1, min_vel=0.01,
                               max_vel=0.2, min_force=0.1, max_force=20.0)

    def Move(self, width, speed, force):
        self.moves.append((width, speed, force))


class EffectiveCommandsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.features = {
            **hw_to_dataset_features(dict.fromkeys(action_feature_names(), float), "action"),
            **hw_to_dataset_features(dict.fromkeys(observation_state_feature_names(), float), "observation"),
        }

    def nero(self, **overrides):
        values = dict(cameras={}, use_gripper=True, debug=False, calibration_dir=self.root,
                      observation_pose_frame="world",
                      left_base_pose_in_world=[0, 0.27, 0, 1, 0, 0, 0],
                      right_base_pose_in_world=[0, -0.27, 0, 1, 0, 0, 0])
        values.update(overrides)
        robot = NeroDualArm(NeroDualArmConfig(**values))
        robot._robot = Mock()
        robot._robot.servo_p_OL.return_value = True
        robot.is_connected = True
        return robot

    def franka(self, **overrides):
        values = dict(cameras={}, debug=False, use_gripper=True, calibration_dir=self.root,
                      max_cartesian_delta=0.06, max_rotation_delta=0.08)
        values.update(overrides)
        robot = FrankaDualArm(FrankaDualArmConfig(**values))
        robot._robot = Mock()
        robot._robot.step.return_value = {"ok": True}
        robot.is_connected = True
        return robot

    def flexiv(self, threaded=False, **overrides):
        values = dict(cameras={}, debug=False, use_gripper=True, calibration_dir=self.root,
                      max_cartesian_delta=0.02, max_rotation_delta=0.04,
                      left_mount_yaw_deg=45, right_mount_yaw_deg=-45,
                      action_debug=False, use_cartesian_servo_thread=threaded)
        values.update(overrides)
        robot = FlexivDualArm(FlexivDualArmConfig(**values))
        robot._left_robot, robot._right_robot = Mock(), Mock()
        robot._left_gripper, robot._right_gripper = Gripper(), Gripper()
        robot.is_connected = True
        return robot

    def test_nero_station_config_and_midpoint_world_observations(self):
        project = Path(__file__).resolve().parents[1]
        cfg = yaml.safe_load((project / "scripts/config/robots/nero_cofig.yaml").read_text())["robot"]
        self.assertEqual(cfg["observation_pose_frame"], "world")
        np.testing.assert_allclose(cfg["left_base_pose_in_world"], [0, 0.27, 0, 1, 0, 0, 0])
        np.testing.assert_allclose(cfg["right_base_pose_in_world"], [0, -0.27, 0, 1, 0, 0, 0])
        robot = self.nero(use_gripper=False)
        robot._robot.left_robot_get_ee_pose.return_value = [0, 0, 0, 0, 0, 0]
        robot._robot.right_robot_get_ee_pose.return_value = [0, 0, 0, 0, 0, 0]
        obs = robot.get_observation()
        for side, y in (("left", 0.27), ("right", -0.27)):
            np.testing.assert_allclose([obs[f"{side}_ee_pose.{axis}"] for axis in AXES], [0, y, 0, 0, 0, 0])
        # A shared point must have the same world position from either base.
        robot._robot.left_robot_get_ee_pose.return_value = [0.4, -0.1, 0.2, 0, 0, 0]
        robot._robot.right_robot_get_ee_pose.return_value = [0.4, 0.44, 0.2, 0, 0, 0]
        obs = robot.get_observation()
        for side in ("left", "right"):
            np.testing.assert_allclose([obs[f"{side}_ee_pose.{axis}"] for axis in AXES[:3]], [0.4, 0.17, 0.2])

    def test_nero_rotated_extrinsic_composes_pose_and_maps_commands_once(self):
        base_rotation = Rotation.from_euler("z", 90, degrees=True)
        quat = base_rotation.as_quat()[[3, 0, 1, 2]].tolist()
        robot = self.nero(use_gripper=False, left_base_pose_in_world=[0, 0.27, 0] + quat)
        local_rotation = Rotation.from_rotvec([0.2, -0.3, 0.4])
        robot._robot.left_robot_get_ee_pose.return_value = [0.4, 0.1, 0.2, 0.4, -0.3, 0.2]
        robot._robot.right_robot_get_ee_pose.return_value = [0, 0, 0, 0, 0, 0]
        obs = robot.get_observation()
        np.testing.assert_allclose([obs[f"left_ee_pose.{axis}"] for axis in AXES[:3]], [-0.1, 0.67, 0.2], atol=1e-12)
        observed_rotation = Rotation.from_rotvec([obs[f"left_ee_pose.{axis}"] for axis in AXES[3:]])
        np.testing.assert_allclose(observed_rotation.as_matrix(), (base_rotation * local_rotation).as_matrix(), atol=1e-12)
        requested = action([0.01, 0.02, 0.003, 0.02, 0.01, -0.005])
        returned = robot.send_action(requested)
        arm, wire, = robot._robot.servo_p_OL.call_args.args
        self.assertEqual(arm, "left_robot")
        np.testing.assert_allclose(wire[:3], base_rotation.inv().apply(vector(requested)[:3]), atol=1e-12)
        np.testing.assert_allclose(wire[3:], base_rotation.inv().apply(vector(requested)[3:]), atol=1e-12)
        np.testing.assert_allclose(vector(returned), vector(requested), atol=1e-12)

    def test_invalid_nero_frame_configuration_fails_before_io(self):
        for extra in ({"observation_pose_frame": "unknown"}, {"left_base_pose_in_world": None},
                      {"right_base_pose_in_world": [0, 0, 0, 0, 0, 0, 0]},
                      {"left_base_pose_in_world": [float("nan"), 0, 0, 1, 0, 0, 0]}):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                self.nero(**extra)

    def test_nero_deadband_rate_limit_debug_and_rejection(self):
        requested = action([0.0005, 0, 0, 0, 0, 0], [0.01, 0, 0, 0, 0, 0])
        robot = self.nero()
        returned = robot.send_action(requested)
        np.testing.assert_array_equal(vector(returned), 0)
        np.testing.assert_allclose(vector(returned, "right"), vector(requested, "right"))
        self.assertEqual(robot._robot.servo_p_OL.call_count, 1)
        self.assertEqual(robot._robot.servo_p_OL.call_args.args[0], "right_robot")
        with patch.object(robot, "_should_send_action", return_value=False):
            returned = robot.send_action(requested)
        np.testing.assert_array_equal(vector(returned, "right"), 0)
        self.assertEqual(robot._robot.servo_p_OL.call_count, 1)
        robot.config.debug = True
        returned = robot.send_action(requested)
        np.testing.assert_array_equal(vector(returned, "right"), 0)
        robot.config.debug = False
        robot._robot.servo_p_OL.return_value = False
        with patch.object(robot, "_should_send_action", return_value=True), self.assertRaisesRegex(RuntimeError, "rejected"):
            robot.send_action(requested)

    def test_nero_healthy_rejected_frame_is_zero_and_other_arm_continues(self):
        robot = self.nero(use_gripper=False)
        requested = action([0.01, 0, 0, 0, 0, 0], [0.02, 0, 0, 0, 0, 0])
        robot._robot.left_robot_get_arm_status.return_value = {
            "arm_status": 0, "all_joints_enabled": True, "joint_enabled": [True] * 7,
        }
        robot._robot.servo_p_OL.side_effect = [False, True, True, True]
        with patch.object(robot, "_should_send_action", return_value=True):
            returned = robot.send_action(requested)
            np.testing.assert_array_equal(vector(returned), 0)
            np.testing.assert_allclose(vector(returned, "right"), vector(requested, "right"))
            # Next accepted frame must contain only its new delta, never retry
            # or accumulate the discarded target from the rejected frame.
            next_action = action([0.003, 0, 0, 0, 0, 0], [0.004, 0, 0, 0, 0, 0])
            returned = robot.send_action(next_action)
        np.testing.assert_allclose(vector(returned), vector(next_action))
        np.testing.assert_allclose(vector(returned, "right"), vector(next_action, "right"))
        self.assertEqual([c.args[0] for c in robot._robot.servo_p_OL.call_args_list],
                         ["left_robot", "right_robot", "left_robot", "right_robot"])
        np.testing.assert_allclose(robot._robot.servo_p_OL.call_args_list[2].args[1], vector(next_action))

    def test_nero_right_rejection_preserves_accepted_left_delta(self):
        robot = self.nero(use_gripper=False)
        requested = action([0.01, 0, 0, 0, 0, 0], [0.02, 0, 0, 0, 0, 0])
        robot._robot.right_robot_get_arm_status.return_value = {
            "arm_status": 0, "all_joints_enabled": True, "joint_enabled": [True] * 7,
        }
        robot._robot.servo_p_OL.side_effect = [True, False]
        returned = robot.send_action(requested)
        np.testing.assert_allclose(vector(returned), vector(requested))
        np.testing.assert_array_equal(vector(returned, "right"), 0)

    def test_nero_fault_disabled_and_unknown_status_remain_fatal(self):
        for status in (
            {"arm_status": 1, "all_joints_enabled": True, "joint_enabled": [True] * 7},
            {"arm_status": 7, "all_joints_enabled": True, "joint_enabled": [True] * 7},
            {"arm_status": 0, "all_joints_enabled": False, "joint_enabled": [False] * 7},
            {"arm_status": 0, "all_joints_enabled": True, "joint_enabled": [True] * 6 + [False]},
            {"arm_status": 0}, None,
        ):
            with self.subTest(status=status):
                robot = self.nero(use_gripper=False)
                robot._robot.servo_p_OL.return_value = False
                robot._robot.left_robot_get_arm_status.return_value = status
                with self.assertRaisesRegex(RuntimeError, "not confirmed healthy"):
                    robot.send_action(action([0.01, 0, 0, 0, 0, 0]))

    def test_nero_rpc_errors_remain_fatal(self):
        for fail_during in ("servo", "status"):
            robot = self.nero(use_gripper=False)
            if fail_during == "servo":
                robot._robot.servo_p_OL.side_effect = ConnectionError("RPC lost")
            else:
                robot._robot.servo_p_OL.return_value = False
                robot._robot.left_robot_get_arm_status.side_effect = ConnectionError("RPC lost")
            with self.subTest(fail_during=fail_during), self.assertRaisesRegex(ConnectionError, "RPC lost"):
                robot.send_action(action([0.01, 0, 0, 0, 0, 0]))

    def test_franka_returned_targets_match_rpc_and_noop_packets(self):
        robot = self.franka()
        returned = robot.send_action(action([0.12, -0.12, 0, 0.2, -0.2, 0]))
        motion = robot._robot.step.call_args.args[0]["left_arm"]["motion"]
        np.testing.assert_allclose(vector(returned), motion["translation"] + motion["rotation_rotvec"])
        np.testing.assert_allclose(vector(returned), [0.06, -0.06, 0, 0.08, -0.08, 0])
        returned = robot.send_action(action([1e-12, 0, 0, 0, 0, 0]))
        np.testing.assert_array_equal(vector(returned), 0)
        robot._robot.step.return_value = {"ok": False}
        with self.assertRaisesRegex(RuntimeError, "rejected"):
            robot.send_action(action())

    def test_flexiv_clipped_targets_return_in_public_frame_for_both_paths(self):
        requested = action([0.03, 0.04, 0, 0.03, 0.04, 0], [-0.03, 0.04, 0, -0.03, 0.04, 0])
        original = copy.deepcopy(requested)
        for threaded in (False, True):
            with self.subTest(threaded=threaded):
                robot = self.flexiv(threaded=threaded)
                returned = robot.send_action(requested)
                self.assertEqual(requested, original)
                for side in ("left", "right"):
                    delta = vector(returned, side)
                    expected = vector(requested, side) * np.array([0.4] * 3 + [0.8] * 3)
                    np.testing.assert_allclose(delta, expected, atol=1e-12)
                    if threaded:
                        target = getattr(robot, f"_servo_{side}_target_pose7")
                    else:
                        target = np.array(getattr(robot, f"_{side}_robot").SendCartesianMotionForce.call_args.args[0])
                    basis = Rotation.from_euler("z", getattr(robot.config, f"{side}_mount_yaw_deg"), degrees=True)
                    np.testing.assert_allclose(target[:3], basis.apply(delta[:3]), atol=1e-12)
                    target_rotation = Rotation.from_quat(target[[4, 5, 6, 3]])
                    np.testing.assert_allclose(target_rotation.as_rotvec(), basis.apply(delta[3:]), atol=1e-12)

    def test_flexiv_debug_and_right_only_actions(self):
        robot = self.flexiv(debug=True)
        returned = robot.send_action(action([0.01, 0, 0, 0, 0, 0]))
        np.testing.assert_array_equal(vector(returned), 0)
        robot._left_robot.SendCartesianMotionForce.assert_not_called()
        robot.config.debug = False
        returned = robot.send_action({"right_delta_ee_pose.y": 0.01})
        np.testing.assert_array_equal(vector(returned), 0)
        np.testing.assert_allclose(vector(returned, "right"), [0, 0.01, 0, 0, 0, 0], atol=1e-12)
        robot._right_robot.SendCartesianMotionForce.assert_called_once()

    def test_nonfinite_nero_and_flexiv_actions_fail_before_arm_io(self):
        for factory in (self.nero, self.flexiv):
            for bad in (float("nan"), float("inf")):
                robot = factory()
                with self.subTest(factory=factory.__name__, bad=bad), self.assertRaises(ValueError):
                    robot.send_action(action([bad, 0, 0, 0, 0, 0]))
                if isinstance(robot, NeroDualArm):
                    robot._robot.servo_p_OL.assert_not_called()
                else:
                    robot._left_robot.SendCartesianMotionForce.assert_not_called()

    def test_saved_teleop_policy_and_mix_rows_match_all_three_adapters(self):
        cases = (
            (self.franka, [0.12, 0.12, 0, 0.2, 0, 0], [0.06, 0.06, 0, 0.08, 0, 0]),
            (self.nero, [0.0005, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]),
            (self.flexiv, [0.03, 0.04, 0, 0.03, 0.04, 0], [0.012, 0.016, 0, 0.024, 0.032, 0]),
        )
        for factory, raw, expected in cases:
            for mode in ("teleop", "policy", "mix"):
                name = factory.__name__ + "_" + mode
                with self.subTest(name=name):
                    robot = factory()
                    robot.get_observation = Mock(return_value=dict.fromkeys(observation_state_feature_names(), 0.0))
                    events = dict(exit_early=False, stop_recording=False, rerecord_episode=False)
                    send = robot.send_action
                    returned = []
                    def send_once(values):
                        result = send(values)
                        returned.append(result)
                        events["exit_early"] = True
                        return result
                    robot.send_action = send_once
                    requested = action(raw)
                    teleop = Mock()
                    teleop.get_action.return_value = dict(requested, left_grip_pressed=True, right_grip_pressed=True)
                    policy = Mock(config=SimpleNamespace(device="cpu", use_amp=False))
                    features = copy.deepcopy(self.features)
                    if mode == "mix":
                        for key in ("policy_action", "expert_action", "sent_action"):
                            features[key] = copy.deepcopy(features["action"])
                        for key in ("action_source", "frame_role", "expert_action_missing"):
                            features[key] = {"dtype": "string", "shape": (1,), "names": None}
                        for key in ("is_expert", "expert_label_complete"):
                            features[key] = {"dtype": "bool", "shape": (1,), "names": None}
                        features["intervention_segment_id"] = {"dtype": "int64", "shape": (1,), "names": None}
                    ds = LeRobotDataset.create(name, fps=30, features=features, root=self.root / name,
                                               robot_type=robot.name, use_videos=False)
                    with patch.object(recording, "busy_wait"), \
                         patch.object(recording, "predict_action", return_value=requested), \
                         patch.object(recording, "make_robot_action", side_effect=lambda value, _: dict(value)):
                        common = dict(
                            robot=robot, events=events, dataset=ds, fps=30,
                            teleop_action_processor=lambda pair: dict(pair[0]),
                            robot_action_processor=lambda pair: dict(pair[0]),
                            robot_observation_processor=dict, control_time_s=5, single_task="test", display_data=False,
                        )
                        if mode == "mix":
                            recording.run_mix_record_loop(**common, teleop=teleop, policy=policy,
                                                          preprocessor=Mock(), postprocessor=Mock())
                        else:
                            recording.record_loop_with_recorded_resets(
                                **common, teleop=teleop if mode == "teleop" else None,
                                policy=policy if mode == "policy" else None,
                                preprocessor=Mock(), postprocessor=Mock(),
                            )
                    ds.save_episode()
                    ds.finalize()
                    loaded = LeRobotDataset(name, root=ds.root)
                    self.assertEqual(len(loaded), 1)
                    saved = loaded[0]["action"].numpy()
                    np.testing.assert_allclose(saved[:6], expected, atol=1e-8)
                    np.testing.assert_array_equal(saved, np.array([returned[0][key] for key in action_feature_names()], dtype=np.float32))
                    if mode == "mix":
                        np.testing.assert_array_equal(saved, loaded[0]["sent_action"].numpy())

    def test_missing_arm_return_aborts_before_recording_and_semantics_are_versioned(self):
        requested = action()
        incomplete = dict(requested)
        del incomplete["left_delta_ee_pose.rx"]
        with self.assertRaisesRegex(ValueError, "left_delta_ee_pose.rx"):
            command_target_action(requested, incomplete, self.features)
        meta = recording_semantics(self.features, self.nero().config)
        self.assertEqual(meta[ACTION_SEMANTICS_KEY], ACTION_SEMANTICS)
        self.assertEqual(meta["observation_pose_frame"], "world")
        validate_recording_semantics(meta, self.features, meta)
        old = dict(meta)
        del old[ACTION_SEMANTICS_KEY]
        with self.assertRaisesRegex(ValueError, "action_semantics"):
            validate_recording_semantics(old, self.features)
        with self.assertRaisesRegex(ValueError, "action_semantics"):
            validate_merge_semantics(meta, old)
        changed = copy.deepcopy(meta)
        changed["base_poses_in_world"]["left"][1] = 0.3
        with self.assertRaisesRegex(ValueError, "base_poses_in_world"):
            validate_recording_semantics(meta, self.features, changed)
        with self.assertRaisesRegex(ValueError, "Cartesian smoothing"):
            validate_gripper_smoothing(meta, {"action_smoothing": {"enabled": True}})


if __name__ == "__main__":
    unittest.main()
