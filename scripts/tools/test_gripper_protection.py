"""Regression tests for physical-width gripper protection (no video IO)."""
import copy
import unittest

import numpy as np

from preprocess_dataset import _gripper_event_mask, _trim_static_runs


NAMES = ['left_gripper_width', 'right_gripper_width']
CFG = {
    'gripper_events': {'enabled': True, 'change_threshold': 0.0001,
                       'include_observation_state': True, 'require_observation_state': True,
                       'require_gripper_features': True, 'keep_radius_frames': 15},
    'static_trim': {'enabled': True, 'min_static_frames': 10,
                    'keep_start_frames': 5, 'keep_end_frames': 5},
}


class GripperProtectionTests(unittest.TestCase):
    def test_either_gripper_and_either_channel(self):
        for channel in (0, 1):
            for hand in (0, 1):
                signals = [np.zeros((120, 2)), np.zeros((120, 2))]
                signals[channel][60:, hand] = 0.00037444  # Smallest observed step, metres.
                mask = _gripper_event_mask(signals[0], NAMES, CFG, signals[1], NAMES)
                self.assertTrue(mask[44:76].all())
                keep = _trim_static_runs(mask, CFG)
                self.assertTrue(keep[44:76].all())
                self.assertGreater(keep.sum(), 10)

    def test_stable_grippers_can_trim(self):
        values = np.full((120, 2), 0.085)
        mask = _gripper_event_mask(values, NAMES, CFG, values, NAMES)
        self.assertFalse(mask.any())
        self.assertEqual(_trim_static_runs(mask, CFG).sum(), 10)

    def test_below_threshold_noise(self):
        values = np.zeros((120, 2))
        values[::2] = 0.000001
        self.assertFalse(_gripper_event_mask(values, NAMES, CFG, values, NAMES).any())

    def test_no_cross_episode_event(self):
        for width in (0.0, 0.085):
            values = np.full((120, 2), width)
            self.assertFalse(_gripper_event_mask(values, NAMES, CFG, values, NAMES).any())

    def test_legacy_threshold_rejected(self):
        cfg = copy.deepcopy(CFG)
        cfg['gripper_events']['change_threshold'] = 0.5
        with self.assertRaisesRegex(ValueError, 'metres'):
            _gripper_event_mask(np.zeros((120, 2)), NAMES, cfg, np.zeros((120, 2)), NAMES)

    def test_missing_state_rejected(self):
        with self.assertRaisesRegex(ValueError, 'observation.state'):
            _gripper_event_mask(np.zeros((120, 2)), NAMES, CFG)

    def test_nan_rejected(self):
        values = np.zeros((120, 2))
        values[60, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'Non-finite'):
            _gripper_event_mask(values, NAMES, CFG, values, NAMES)

    def test_short_episode(self):
        values = np.zeros((1, 2))
        mask = _gripper_event_mask(values, NAMES, CFG, values, NAMES)
        self.assertEqual(_trim_static_runs(mask, CFG).sum(), 1)


if __name__ == '__main__':
    unittest.main()
