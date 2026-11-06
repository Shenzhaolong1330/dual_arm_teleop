"""Verify configured controller position and axial rotation in the common frame."""

from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from scripts.core.run_record import RecordConfig
from teleoperators.oculus_teleoperator import OculusTeleop
from robots.dual_flexiv_rizon4s.flexiv_dual_arm import FlexivDualArm
from robots.dual_flexiv_rizon4s.pose_frames import BaseToWorldTransform


class Reader:
    def __init__(self):
        self.poses = {side: np.eye(4) for side in ("l", "r")}

    def get_transformations_and_buttons(self):
        return self.poses, {"LG": True, "RG": True}

    def stop(self):
        pass


class WorldTeleopAxesTests(unittest.TestCase):
    def test_six_axes_are_consistent_for_both_hands_and_config_entries(self):
        root = Path(__file__).resolve().parents[3]
        record = yaml.safe_load((root / "scripts/config/record_cfg.yaml").read_text())["record"]
        record["run_mode"] = "run_record"
        files = [root / "scripts/config/robots/flexiv_config.yaml"]
        # Face-to-face routing rotates controller deltas 180 degrees about world Z.
        mapping = np.diag([-1, -1, 1]) @ np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
        for file in files:
            cfg = RecordConfig({**record, "robot_type": "flexiv_dual_arm", "das_config_path": str(file)})
            teleop_cfg = replace(cfg.create_teleop_config(), processing_profile=None, action_smoothing_method="none",
                                 action_deadband_translation=0, action_deadband_rotation=0)
            self.assertTrue(teleop_cfg.mirror_teleop)
            self.assertEqual(cfg.robot_extra_config["observation_pose_frame"], "world")
            for side in ("left", "right"):
                self.assertEqual(cfg.robot_extra_config[f"{side}_mount_yaw_deg"], 0.0)
                np.testing.assert_allclose(cfg.robot_extra_config[f"{side}_base_pose_in_world"][3:],
                                           [1, 0, 0, 0])
            for rotation in (False, True):
                for axis in range(3):
                    with self.subTest(config=file.name, rotation=rotation, axis=axis):
                        reader = Reader()
                        teleop = OculusTeleop(teleop_cfg)
                        with patch("teleoperators.oculus_teleoperator.oculus.oculus_dual_arm_robot.OculusReader",
                                   return_value=reader):
                            teleop.connect()
                        try:
                            teleop.get_action()
                            delta = np.eye(3)[axis] * 0.01
                            for pose in reader.poses.values():
                                if rotation:
                                    pose[:3, :3] = Rotation.from_rotvec(delta).as_matrix()
                                else:
                                    pose[:3, 3] = delta
                            action = teleop.get_action()
                            for side in ("left", "right"):
                                vector = np.array([action[f"{side}_delta_ee_pose.{a}"]
                                                   for a in ("x", "y", "z", "rx", "ry", "rz")])
                                base_vector = FlexivDualArm._apply_mount_yaw(
                                    vector, cfg.robot_extra_config[f"{side}_mount_yaw_deg"],
                                )
                                transform = BaseToWorldTransform(
                                    cfg.robot_extra_config[f"{side}_base_pose_in_world"],
                                    name=side,
                                )
                                pose_world = transform.apply([0, 0, 0, 1, 0, 0, 0])
                                world_rotation = Rotation.from_quat(np.array(pose_world)[[4, 5, 6, 3]])
                                start = 3 if rotation else 0
                                np.testing.assert_allclose(world_rotation.apply(base_vector[start:start + 3]),
                                                           mapping @ delta, atol=1e-12)
                        finally:
                            teleop.disconnect()


if __name__ == "__main__":
    unittest.main()
