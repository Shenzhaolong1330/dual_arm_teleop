"""LeRobot robot plugin for Agilex Nero dual-arm robot.

This package is discovered automatically by `register_third_party_devices`
because its top-level module name starts with `lerobot_robot_`.
"""

from lerobot.robots.config import RobotConfig

from robots.dual_agilex_nero import NeroDualArm, NeroDualArmConfig

# Register an alias used by DAgger/train configs in this workspace.
RobotConfig.register_subclass("agilex_nero")(NeroDualArmConfig)

__all__ = ["NeroDualArm", "NeroDualArmConfig"]
