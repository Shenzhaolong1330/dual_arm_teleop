"""Regression checks for rotation protocols, accepted targets and TCP offsets."""
from __future__ import annotations

import copy
from types import SimpleNamespace
import tempfile
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial.transform import Rotation as R

from robots.dual_agilex_nero.physical_protocol import (
    PROTOCOL, PhysicalProtocolMixin, integrate_pose, pose_delta,
    rpy_pose_to_rotvec, rotvec_pose_to_rpy, tool_pose, validate_receipt,
)
from robots.dual_agilex_nero.config_nero import NeroDualArmConfig
from robots.dual_agilex_nero.nero_dual_arm import NeroDualArm
from robots.dual_arm_schema import AXES, action_feature_names, observation_state_feature_names, recording_semantics, validate_recording_semantics
from robots.physical_alignment import physical_metadata, require_cross_body_alignment
from scripts.utils.dataset_utils import validate_merge_semantics
from scripts.tools.physical_calibration import pivot_calibration, world_calibration


class Server(PhysicalProtocolMixin):
    """Pure legacy dynamics: component limits, scaling, and rotation freezing."""
    tcp_offset = [0.0] * 6

    def __init__(self):
        self.left_cur_pose = np.array([0.3, -0.2, 0.4, 0.7, -0.6, 0.8])
        self.scale = 1.0
        self.freeze = False
        self.sent = []

    def servo_p_OL(self, arm, wire, delta):
        wire = self._physical_prepare_delta(arm, wire)
        wire = np.clip(wire, [-0.03] * 3 + [-0.35] * 3, [0.03] * 3 + [0.35] * 3)
        wire *= self.scale
        if self.freeze:
            wire[3:] = 0
        before = self.left_cur_pose.copy()
        after = rotvec_pose_to_rpy(integrate_pose(rpy_pose_to_rotvec(before), rpy_pose_to_rotvec(wire)))
        self.sent.append(after.copy())
        self.left_cur_pose = after
        self._physical_receipt = before, after
        return True


class PhysicalAlignmentTest(unittest.TestCase):
    def test_rpy_roundtrip_and_no_axis_reversal(self):
        rng = np.random.default_rng(42)
        for raw in rng.uniform(-1, 1, (100, 6)):
            pose = rpy_pose_to_rotvec(raw)
            np.testing.assert_allclose(R.from_rotvec(pose[3:]).as_matrix(), R.from_euler("xyz", raw[3:]).as_matrix(), atol=1e-12)
            np.testing.assert_allclose(rotvec_pose_to_rpy(pose), raw, atol=1e-12)

    def test_tcp_rotation_keeps_contact_point_stationary(self):
        server = Server()
        tcp = [0.12, 0.02, -0.03, 1, 0, 0, 0]
        before = tool_pose(rpy_pose_to_rotvec(server.left_cur_pose), tcp)
        requested = [0, 0, 0, 0.02, -0.01, 0.03]
        reply = server.servo_p_OL_v2("left_robot", requested, tcp)
        after = tool_pose(rpy_pose_to_rotvec(server.sent[-1]), tcp)
        np.testing.assert_allclose(after[:3], before[:3], atol=1e-12)
        np.testing.assert_allclose(validate_receipt(reply), requested, atol=1e-12)
        self.assertGreater(np.linalg.norm(server.sent[-1][:3] - reply["reference_pose"][:3]), 0)

    def test_receipt_reports_clipping_scaling_and_freeze(self):
        server = Server()
        server.scale, server.freeze = 0.25, True
        reply = server.servo_p_OL_v2("left_robot", [0.1, 0, 0, 0.2, 0.1, -0.1])
        np.testing.assert_allclose(validate_receipt(reply), [0.0075, 0, 0, 0, 0, 0], atol=1e-12)
        bad = copy.deepcopy(reply)
        bad["effective_delta"][0] = 0.1
        with self.assertRaises(RuntimeError):
            validate_receipt(bad)
        with self.assertRaises(RuntimeError):
            validate_receipt(True)
        with self.assertRaises(RuntimeError):
            validate_receipt({"protocol": PROTOCOL, "accepted": False, "reason": "unknown"})

    def test_adapter_records_server_delta_in_world_not_requested_delta(self):
        with tempfile.TemporaryDirectory() as tmp:
            q = R.from_euler("z", 90, degrees=True).as_quat()[[3, 0, 1, 2]].tolist()
            robot = NeroDualArm(NeroDualArmConfig(
                cameras={}, use_gripper=False, debug=False, calibration_dir=Path(tmp),
                observation_pose_frame="world", left_base_pose_in_world=[0, .27, 0] + q,
                right_base_pose_in_world=[0, -.27, 0, 1, 0, 0, 0],
            ))
            robot._robot = Mock()
            reply = Server().servo_p_OL_v2("left_robot", [0.1, 0, 0, 0, 0, 0])
            robot._robot.servo_p_OL_v2.return_value = reply
            robot.is_connected = True
            action = dict.fromkeys(action_feature_names(use_gripper=False), 0.0)
            action["left_delta_ee_pose.y"] = 0.1
            with patch.object(robot, "_should_send_action", return_value=True):
                effective = robot.send_action(action)
            delta = np.array([effective[f"left_delta_ee_pose.{a}"] for a in AXES])
            np.testing.assert_allclose(delta, [0, .03, 0, 0, 0, 0], atol=1e-12)

    def test_old_unknown_or_changed_calibration_is_not_compatible(self):
        config = NeroDualArmConfig(cameras={})
        info = physical_metadata(config)
        with self.assertRaises(ValueError):
            require_cross_body_alignment(info)
        with self.assertRaises(ValueError):
            validate_merge_semantics(info, {})
        config.physical_calibration = {
            "status": "model_based", "tcp_definition": "finger_midpoint_x_closing_z_approach_v1",
            "evidence": "fixture manufacturing drawing and controller tool snapshot",
        }
        require_cross_body_alignment(physical_metadata(config))

    def test_pivot_recovers_offset_and_rejects_unobservable_samples(self):
        rotations = R.random(30, random_state=23)
        offset, point = np.array([.13, -.02, .01]), np.array([.4, -.2, .3])
        poses = np.c_[point - rotations.apply(offset), rotations.as_rotvec()]
        result = pivot_calibration(poses)
        np.testing.assert_allclose(result["tcp_translation_in_native_m"], offset, atol=1e-12)
        self.assertFalse(result["orientation_calibrated"])
        with self.assertRaises(ValueError):
            pivot_calibration(np.repeat(poses[:1], 10, axis=0))

    def test_same_body_merge_rejects_different_calibration_provenance(self):
        from scripts.tools.merge_lerobot_datasets import _validate_compatible_sources
        reference = SimpleNamespace(info={"robot_type": "nero_dual_arm", "physical_calibration": {"sha256": "one"}})
        candidate = SimpleNamespace(info={"robot_type": "nero_dual_arm", "physical_calibration": {"sha256": "two"}})
        with self.assertRaisesRegex(ValueError, "physical_calibration mismatch"):
            _validate_compatible_sources(reference, candidate, reference_name="one", candidate_name="two",
                                         strict_schema=True, strict_robot_type=True)

    def test_world_fit_recovers_installation_and_rejects_bad_pairing(self):
        source = np.random.default_rng(3).normal(size=(20, 3))
        rotation = R.from_euler("xyz", [.2, -.1, .7])
        target = rotation.apply(source) + [.1, .25, -.02]
        result = world_calibration(source, target)
        np.testing.assert_allclose(result["pose_in_world"][:3], [.1, .25, -.02], atol=1e-12)
        np.testing.assert_allclose(R.from_quat(np.array(result["pose_in_world"])[[4,5,6,3]]).as_matrix(), rotation.as_matrix(), atol=1e-12)
        with self.assertRaises(ValueError):
            world_calibration(source, target[::-1])

    def test_missing_feedback_never_reuses_previous_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            robot = NeroDualArm(NeroDualArmConfig(cameras={}, use_gripper=False, calibration_dir=Path(tmp)))
            robot._robot = Mock()
            robot.is_connected = True
            robot._prev_observation = {"left_ee_pose.x": 123}
            robot._robot.get_ee_pose_v2.side_effect = ConnectionError("link lost")
            with self.assertRaises(ConnectionError):
                robot.get_observation()

    def test_cross_body_merge_keeps_per_frame_body_and_calibration_ids(self):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.datasets.utils import write_info
        from robots.dual_franka.config_franka import FrankaDualArmConfig
        from scripts.tools.merge_lerobot_datasets import merge_lerobot_datasets
        features = {
            "action": {"dtype": "float32", "shape": (14,), "names": list(action_feature_names())},
            "observation.state": {"dtype": "float32", "shape": (14,), "names": list(observation_state_feature_names())},
        }
        qualified = {"status": "model_based", "tcp_definition": "finger_midpoint_x_closing_z_approach_v1",
                     "evidence": "synthetic test calibration"}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i, cfg in enumerate([FrankaDualArmConfig(), NeroDualArmConfig(observation_pose_frame="world")]):
                cfg.physical_calibration = qualified
                ds = LeRobotDataset.create(f"body{i}", fps=30, root=root/f"body{i}", features=features,
                                           robot_type=cfg.name, use_videos=False)
                ds.meta.info.update(recording_semantics(features, cfg))
                write_info(ds.meta.info, ds.root)
                ds.add_frame({"action": np.full(14, i, dtype=np.float32),
                              "observation.state": np.zeros(14, dtype=np.float32), "task": "synthetic"})
                ds.save_episode()
                ds.finalize()
            merging_cfg = {"source": {"parent_dir": str(root), "datasets": [{"name": f"body{i}", "root": str(root/f"body{i}")} for i in range(2)],
                                      "strict_schema": True, "strict_robot_type": False},
                           "output": {"dataset_name": "mixed", "root": str(root/"mixed")}}
            merge_lerobot_datasets(merging_cfg)
            mixed = LeRobotDataset("mixed", root=root/"mixed")
            self.assertEqual(mixed.meta.info["robot_type"], "mixed")
            self.assertEqual([int(mixed[i]["embodiment_id"].item()) for i in range(2)], [0, 1])
            self.assertEqual([int(mixed[i]["calibration_id"].item()) for i in range(2)], [0, 1])
            self.assertEqual(len(mixed.meta.info["source_calibrations"]), 2)
            self.assertNotIn("physical_calibration", mixed.meta.info)


if __name__ == "__main__":
    unittest.main()
