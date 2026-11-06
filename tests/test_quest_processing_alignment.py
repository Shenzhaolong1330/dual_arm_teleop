"""Replay identical Quest streams through the real configured teleop pipelines."""

from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from robots.dual_arm_schema import recording_semantics, validate_recording_semantics
from scripts.core.run_record import RecordConfig
from scripts.utils.dataset_utils import validate_merge_semantics
from teleoperators.oculus_teleoperator import OculusTeleop, OculusTeleopConfig
from teleoperators.oculus_teleoperator.processing_profiles import (
    CROSS_EMBODIMENT_PROFILE, PROCESSING_PARAMETER_KEYS, processing_profile_parameters,
)

ROOT = Path(__file__).resolve().parents[1]
INSTALLATIONS = json.loads((ROOT / "tests/fixtures/quest_installation_mappings.json").read_text())


def record_config(path):
    record = yaml.safe_load((ROOT / "scripts/config/record_cfg.yaml").read_text())["record"]
    record["run_mode"] = "run_record"
    record["robot_type"] = "franka_dual_arm" if "franka" in str(path) else (
        "nero_dual_arm" if "nero_cofig" in str(path) else "flexiv_dual_arm"
    )
    record["das_config_path"] = str(ROOT / path)
    return RecordConfig(record)


class Reader:
    def __init__(self):
        self.poses = {hand: np.eye(4) for hand in ("l", "r")}
        self.missing = False
        self.buttons = {"LG": True, "RG": True, "leftTrig": (0.5,), "rightTrig": (0.5,)}

    def get_transformations_and_buttons(self):
        return {} if self.missing else self.poses, self.buttons

    def advance(self, left, right):
        for hand, delta in (("l", left), ("r", right)):
            self.poses[hand][:3, 3] += delta[:3]
            self.poses[hand][:3, :3] = Rotation.from_rotvec(delta[3:]).as_matrix() @ self.poses[hand][:3, :3]


def connect(config):
    reader = Reader()
    teleop = OculusTeleop(config)
    with patch("teleoperators.oculus_teleoperator.oculus.oculus_dual_arm_robot.OculusReader", return_value=reader):
        teleop.connect()
    return teleop, reader


def controller_frame_vectors(action, config):
    """Undo installation/routing only, so common response traces can be compared."""
    translation_basis = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
    rotation_basis = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    mirror = np.diag([-1, -1, 1]) if config.mirror_teleop else np.eye(3)
    result = {}
    for destination in ("left", "right"):
        source = ("right" if destination == "left" else "left") if config.mirror_teleop else destination
        signs = getattr(config, f"{source}_channel_signs")
        position = np.array([action[f"{destination}_delta_ee_pose.{axis}"] for axis in ("x", "y", "z")])
        rotation = np.array([action[f"{destination}_delta_ee_pose.{axis}"] for axis in ("rx", "ry", "rz")])
        position_map = mirror @ np.diag(signs[:3]) @ np.eye(3)[config.position_axis_order] @ translation_basis
        rotation_map = mirror @ np.diag(signs[3:]) @ np.eye(3)[config.rotation_axis_order] @ rotation_basis
        result[source] = np.r_[np.linalg.solve(position_map, position), np.linalg.solve(rotation_map, rotation)]
    return result


class QuestProcessingAlignmentTest(unittest.TestCase):
    def test_all_config_entries_share_profile_and_preserve_installations(self):
        expected = processing_profile_parameters()
        for path, baseline in INSTALLATIONS.items():
            with self.subTest(path=path):
                raw = yaml.safe_load((ROOT / path).read_text())
                for key, value in baseline["mapping"].items():
                    self.assertEqual(raw["teleop"]["oculus_config"][key], value)
                for key, value in baseline["installation"].items():
                    self.assertEqual(raw["robot"][key], value)
                cfg = record_config(path).create_teleop_config()
                self.assertEqual(cfg.processing_profile, CROSS_EMBODIMENT_PROFILE)
                self.assertEqual(cfg.processing_metadata()["parameters"], expected)
                self.assertTrue(set(PROCESSING_PARAMETER_KEYS).isdisjoint(raw["teleop"]["oculus_config"]))

    def test_direct_config_uses_identical_defaults_without_shared_mutable_lists(self):
        first = OculusTeleopConfig(processing_profile=CROSS_EMBODIMENT_PROFILE)
        second = OculusTeleopConfig(processing_profile=CROSS_EMBODIMENT_PROFILE)
        self.assertEqual(first.processing_metadata()["parameters"], processing_profile_parameters())
        first.left_pose_scaler[0] = 9.0
        self.assertEqual(second.left_pose_scaler, [1.0, 1.0])
        self.assertEqual(processing_profile_parameters()["left_pose_scaler"], [1.0, 1.0])
        with self.assertRaisesRegex(ValueError, "left_pose_scaler"):
            first.processing_metadata()

    def test_named_profile_rejects_local_drift_and_unknown_versions(self):
        base = yaml.safe_load((ROOT / "scripts/config/record_cfg.yaml").read_text())["record"]
        base["run_mode"] = "run_record"
        for key, value in (("left_pose_scaler", [0.5, 1]), ("action_smoothing_method", "ema"),
                           ("action_deadband_rotation", 0), ("gripper_trigger_gamma", 0.7),
                           ("processing_profile", "typo_v1")):
            raw = deepcopy(base)
            raw["robot_type"] = "nero_dual_arm"
            raw["teleop"] = {"oculus_config": {key: value}}
            with self.subTest(key=key), self.assertRaises(ValueError):
                RecordConfig(raw)
        with self.assertRaisesRegex(ValueError, "gripper_trigger_gamma"):
            OculusTeleopConfig(processing_profile=CROSS_EMBODIMENT_PROFILE, gripper_trigger_gamma=0.7)
        raw = deepcopy(base)
        raw["fps"] = 15
        with self.assertRaisesRegex(ValueError, "fps=30"):
            RecordConfig(raw)
        # Deliberate custom experiments remain possible without masquerading as v1.
        raw["teleop"] = {"oculus_config": {"processing_profile": None, "action_smoothing_method": "ema"}}
        custom = RecordConfig(raw).create_teleop_config()
        self.assertEqual(custom.processing_metadata()["profile"], "custom")
        self.assertEqual(custom.action_smoothing_method, "ema")

    def test_identical_motion_streams_have_identical_response_after_removing_geometry(self):
        traces = []
        for path in INSTALLATIONS:
            cfg = record_config(path).create_teleop_config()
            teleop, reader = connect(cfg)
            try:
                samples = [teleop.get_action()]
                deltas = [
                    ([0.001, -0.002, 0.003, 0.004, -0.005, 0.006], [-0.003, 0.001, 0.002, -0.007, 0.004, 0.005]),
                    ([0.00001] * 6, [-0.00001] * 6),  # Both below deadbands.
                    ([0.003, 0.001, -0.002, 0.006, 0.005, -0.004], [0.002, -0.003, 0.001, 0.003, -0.004, 0.005]),
                    ([0] * 6, [0] * 6),  # Filter tail.
                    ([0.2, 0, 0, 0.01, 0, 0], [0.2, 0, 0, 0.01, 0, 0]),  # Rejected spikes.
                    ([0.002, 0, 0, 0, 0.004, 0], [0, 0.002, 0, 0.004, 0, 0]),
                ]
                for left, right in deltas:
                    reader.advance(np.array(left), np.array(right))
                    samples.append(teleop.get_action())
                reader.missing = True
                samples.extend(teleop.get_action() for _ in range(3))
                reader.missing = False
                reader.buttons.update(LG=False, RG=False)
                samples.append(teleop.get_action())
                reader.advance(np.full(6, 0.01), np.full(6, 0.01))
                reader.buttons.update(LG=True, RG=True)
                samples.append(teleop.get_action())  # New clutch reference, no jump.
                traces.append([controller_frame_vectors(sample, cfg) for sample in samples])
            finally:
                teleop.disconnect()
        reference = traces[0]
        self.assertGreater(np.linalg.norm(reference[1]["left"]), 0)
        self.assertLess(np.linalg.norm(reference[1]["left"]), np.linalg.norm([0.001, -0.002, 0.003, 0.004, -0.005, 0.006]))
        for trace in traces:
            for i, sample in enumerate(trace):
                for side in ("left", "right"):
                    np.testing.assert_allclose(sample[side], reference[i][side], atol=1e-12)
            for i in (0, 5, 9, 10, 11):
                for side in ("left", "right"):
                    np.testing.assert_array_equal(trace[i][side], 0)

    def test_deadband_and_rotation_spike_are_identical_from_a_fresh_reference(self):
        for path in INSTALLATIONS:
            for delta in ([0.00001] * 6, [0, 0, 0, 0.2, 0, 0]):
                cfg = record_config(path).create_teleop_config()
                teleop, reader = connect(cfg)
                try:
                    teleop.get_action()
                    reader.advance(np.array(delta), np.array(delta))
                    output = controller_frame_vectors(teleop.get_action(), cfg)
                    for value in output.values():
                        np.testing.assert_array_equal(value, 0)
                finally:
                    teleop.disconnect()

    def test_trigger_curve_matches_normalized_aperture_with_hardware_limits_retained(self):
        for path in INSTALLATIONS:
            cfg = record_config(path).create_teleop_config()
            teleop, reader = connect(cfg)
            try:
                for trigger in (0, 0.01, 0.02, 0.25, 0.5, 0.75, 0.98, 0.99, 1):
                    reader.buttons.update(leftTrig=(trigger,), rightTrig=(1 - trigger,))
                    actual = teleop.get_action()
                    for destination in ("left", "right"):
                        source = ("right" if destination == "left" else "left") if cfg.mirror_teleop else destination
                        raw_trigger = trigger if source == "left" else 1 - trigger
                        shaped = np.clip((raw_trigger - 0.02) / 0.96, 0, 1)
                        expected_open = 1 - shaped
                        width = actual[f"{destination}_gripper_width"]
                        normalized = (width - cfg.gripper_min_width) / (cfg.gripper_max_open - cfg.gripper_min_width)
                        self.assertAlmostEqual(normalized, expected_open)
                        self.assertGreaterEqual(width, cfg.gripper_min_width)
                        self.assertLessEqual(width, cfg.gripper_max_open)
            finally:
                teleop.disconnect()

    def test_processing_metadata_allows_other_installations_but_detects_response_drift(self):
        features = {"action": {"names": ["left_delta_ee_pose.x", "left_gripper_width", "right_gripper_width"]}}
        entries = [recording_semantics(features, teleop_config=record_config(path).create_teleop_config()) for path in INSTALLATIONS]
        for meta in entries:
            validate_recording_semantics(meta, features, meta)
            validate_merge_semantics(entries[0], meta)
        changed = deepcopy(entries[0])
        changed["quest_processing"]["parameters"]["gripper_trigger_gamma"] = 0.7
        with self.assertRaisesRegex(ValueError, "quest_processing"):
            validate_merge_semantics(entries[0], changed)
        with self.assertRaisesRegex(ValueError, "quest_processing"):
            validate_recording_semantics(entries[0], features, changed)
        old = deepcopy(entries[0])
        del old["quest_processing"]
        with self.assertRaisesRegex(ValueError, "quest_processing"):
            validate_merge_semantics(entries[0], old)


if __name__ == "__main__":
    unittest.main()
