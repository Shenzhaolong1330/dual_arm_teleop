"""Hardware-free checks for common-world Flexiv observations and control caches."""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation

from robots.dual_flexiv_rizon4s.config_flexiv import FlexivDualArmConfig
from robots.dual_flexiv_rizon4s.flexiv_dual_arm import FlexivDualArm, _apply_delta_to_pose7
from robots.dual_flexiv_rizon4s.pose_frames import BaseToWorldTransform


def pose7(position, rotation):
    quaternion = rotation.as_quat()
    return np.r_[position, quaternion[[3, 0, 1, 2]]]


def rotation_of(pose):
    return Rotation.from_quat(np.asarray(pose)[[4, 5, 6, 3]])


def world_config(**overrides):
    values = {
        "observation_pose_frame": "world",
        "left_mount_yaw_deg": 45.0,
        "right_mount_yaw_deg": -45.0,
        "left_base_pose_in_world": pose7([0, 0.16, 0], Rotation.from_euler("z", -45, degrees=True)).tolist(),
        "right_base_pose_in_world": pose7([0, -0.16, 0], Rotation.from_euler("z", 45, degrees=True)).tolist(),
    }
    values.update(overrides)
    return FlexivDualArmConfig(**values)


class StateOnlyRobot:
    """No command API: accidental hardware/control calls fail in these tests."""

    def __init__(self, tcp_pose):
        self.pose = np.asarray(tcp_pose, dtype=float).copy()

    def states(self):
        return SimpleNamespace(tcp_pose=self.pose.copy())


class PoseFrameTests(unittest.TestCase):
    def test_same_world_pose_seen_from_both_bases_matches(self):
        world_position = np.array([0.45, 0.02, 0.12])
        world_rotation = Rotation.from_euler("xyz", [0.3, -0.5, 0.8])
        config = world_config()
        for side in ("left", "right"):
            with self.subTest(side=side):
                extrinsic = np.array(getattr(config, f"{side}_base_pose_in_world"))
                base_rotation = rotation_of(extrinsic)
                tcp_in_base = pose7(
                    base_rotation.inv().apply(world_position - extrinsic[:3]),
                    base_rotation.inv() * world_rotation,
                )
                actual = BaseToWorldTransform(extrinsic, name=side).apply(tcp_in_base)
                assert_allclose(actual[:3], world_position, atol=1e-12)
                self.assertLess((rotation_of(actual).inv() * world_rotation).magnitude(), 1e-12)

    def test_identity_tcp_returns_base_origin_and_orientation(self):
        config = world_config()
        for side in ("left", "right"):
            extrinsic = np.array(getattr(config, f"{side}_base_pose_in_world"))
            actual = BaseToWorldTransform(extrinsic, name=side).apply([0, 0, 0, 1, 0, 0, 0])
            assert_allclose(actual, extrinsic, atol=1e-12)
        self.assertAlmostEqual(
            np.linalg.norm(
                np.array(config.left_base_pose_in_world[:3]) - config.right_base_pose_in_world[:3]
            ),
            0.32,
        )

    def test_full_orientation_is_composed_not_added(self):
        base_rotation = Rotation.from_euler("xyz", [0.4, -0.2, 0.6])
        tcp_rotation = Rotation.from_euler("xyz", [-0.5, 0.7, 0.2])
        transform = BaseToWorldTransform(pose7([0.1, 0.2, 0.3], base_rotation), name="base")
        actual = transform.apply(pose7([0.3, -0.4, 0.5], tcp_rotation))
        expected_matrix = base_rotation.as_matrix() @ tcp_rotation.as_matrix()
        assert_allclose(rotation_of(actual).as_matrix(), expected_matrix, atol=1e-12)

    def test_invalid_extrinsics_fail_before_hardware_connection(self):
        for value in ([0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [np.nan, 0, 0, 1, 0, 0, 0]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                FlexivDualArm(world_config(left_base_pose_in_world=value))
        with self.assertRaisesRegex(ValueError, "right_base_pose_in_world"):
            FlexivDualArm(world_config(right_base_pose_in_world=None))
        with self.assertRaisesRegex(ValueError, "observation_pose_frame"):
            FlexivDualArm(FlexivDualArmConfig(observation_pose_frame="unknown"))

    def test_observations_are_world_but_control_and_servo_caches_stay_base(self):
        config = world_config()
        robot = FlexivDualArm(config)
        raw = pose7([0.4, 0.03, 0.15], Rotation.from_euler("xyz", [0.2, 0.4, -0.3]))
        robot._left_robot = StateOnlyRobot(raw)
        robot._right_robot = StateOnlyRobot(raw)
        robot.is_connected = True
        obs = robot.get_observation()
        for side in ("left", "right"):
            extrinsic = np.array(getattr(config, f"{side}_base_pose_in_world"))
            actual_position = [obs[f"{side}_ee_pose.{axis}"] for axis in ("x", "y", "z")]
            actual_rotation = Rotation.from_rotvec(
                [obs[f"{side}_ee_pose.{axis}"] for axis in ("rx", "ry", "rz")]
            )
            assert_allclose(
                actual_position, rotation_of(extrinsic).apply(raw[:3]) + extrinsic[:3], atol=1e-12
            )
            expected_rotation = rotation_of(extrinsic) * rotation_of(raw)
            self.assertLess((actual_rotation.inv() * expected_rotation).magnitude(), 1e-12)
            assert_allclose(getattr(robot, f"_cached_{side}_pose7"), raw)
        robot._refresh_cached_poses()
        assert_allclose(robot._servo_left_target_pose7, raw)
        assert_allclose(robot._servo_right_target_pose7, raw)
        self.assertEqual(
            robot.observation_features, FlexivDualArm(FlexivDualArmConfig()).observation_features
        )

    def test_default_base_observation_remains_compatible(self):
        robot = FlexivDualArm(FlexivDualArmConfig())
        raw = pose7([0.4, -0.1, 0.2], Rotation.from_euler("xyz", [0.2, -0.3, 0.4]))
        obs = {}
        robot._add_arm_observation(obs, "left", StateOnlyRobot(raw))
        assert_allclose([obs[f"left_ee_pose.{a}"] for a in ("x", "y", "z")], raw[:3])
        assert_allclose([obs[f"left_ee_pose.{a}"] for a in ("rx", "ry", "rz")], rotation_of(raw).as_rotvec())

    def test_existing_action_mount_mapping_is_inverse_of_observation_mapping(self):
        config = world_config()
        robot = FlexivDualArm(config)
        raw = pose7([0.4, 0.05, 0.2], Rotation.from_euler("xyz", [0.4, -0.2, 0.3]))
        world_delta = np.array([0.001, -0.002, 0.003, 0.01, 0.02, -0.03])
        for side in ("left", "right"):
            with self.subTest(side=side):
                transform = robot._observation_transforms[side]
                base_delta = robot._apply_mount_yaw(world_delta, getattr(config, f"{side}_mount_yaw_deg"))
                actual = transform.apply(_apply_delta_to_pose7(raw, base_delta))
                expected = _apply_delta_to_pose7(transform.apply(raw), world_delta)
                assert_allclose(actual[:3], expected[:3], atol=1e-12)
                self.assertLess((rotation_of(actual).inv() * rotation_of(expected)).magnitude(), 1e-12)

    def test_transform_does_not_modify_input_poses(self):
        base = np.array([0, 0.16, 0, 2, 0, 0, 0], dtype=float)
        tcp = np.array([0.4, 0.2, 0.1, 1, 0, 0, 0], dtype=float)
        base_before, tcp_before = base.copy(), tcp.copy()
        actual = BaseToWorldTransform(base, name="base").apply(tcp)
        assert_allclose(base, base_before)
        assert_allclose(tcp, tcp_before)
        assert_allclose(actual[:3], [0.4, 0.36, 0.1])

    def test_station_configs_use_confirmed_32_cm_geometry(self):
        folder = Path(__file__).resolve().parents[1]
        for name in ("station_deepcybo_5090.yaml", "station_deepcybo_5090_readonly.yaml"):
            with self.subTest(name=name):
                config = yaml.safe_load((folder / name).read_text())["robot"]
                self.assertEqual(config["observation_pose_frame"], "world")
                assert_allclose(config["left_base_pose_in_world"][:3], [0, 0.16, 0])
                assert_allclose(config["right_base_pose_in_world"][:3], [0, -0.16, 0])
                for side in ("left", "right"):
                    mount = Rotation.from_euler("z", config[f"{side}_mount_yaw_deg"], degrees=True)
                    base_to_world = rotation_of(config[f"{side}_base_pose_in_world"])
                    self.assertLess((base_to_world * mount).magnitude(), 1e-12)


if __name__ == "__main__":
    unittest.main()
