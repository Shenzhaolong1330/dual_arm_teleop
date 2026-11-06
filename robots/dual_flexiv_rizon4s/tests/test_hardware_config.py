"""Read-only diagnostics inherit calibration/Home, with automatic actions disabled."""
import unittest

from robots.dual_flexiv_rizon4s.hardware_config import (
    HARDWARE_CONFIG_PATH, READ_ONLY_FLAGS, load_hardware_config,
)


class HardwareConfigTests(unittest.TestCase):
    def test_readonly_inherits_hardware_and_disables_automatic_control(self):
        normal = load_hardware_config()
        readonly = load_hardware_config(readonly=True)
        self.assertTrue(HARDWARE_CONFIG_PATH.is_file())
        self.assertEqual(normal["cameras"], readonly["cameras"])
        self.assertEqual(normal["teleop"], readonly["teleop"])
        for key, value in normal["robot"].items():
            if key not in READ_ONLY_FLAGS:
                self.assertEqual(readonly["robot"][key], value, key)
        for key in READ_ONLY_FLAGS:
            self.assertIs(readonly["robot"][key], False, key)
        readonly["robot"]["left_home_joints"][0] += 1
        self.assertNotEqual(readonly["robot"]["left_home_joints"], load_hardware_config()["robot"]["left_home_joints"])


if __name__ == "__main__":
    unittest.main()
