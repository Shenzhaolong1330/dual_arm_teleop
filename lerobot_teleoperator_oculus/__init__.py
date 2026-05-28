"""LeRobot teleoperator plugin for Oculus teleoperation.

This package is discovered automatically by `register_third_party_devices`
because its top-level module name starts with `lerobot_teleoperator_`.
"""

from lerobot.teleoperators.config import TeleoperatorConfig

from teleoperators.oculus_teleoperator import OculusTeleop, OculusTeleopConfig

# Register an alias used by DAgger/train configs in this workspace.
TeleoperatorConfig.register_subclass("oculus")(OculusTeleopConfig)

__all__ = ["OculusTeleop", "OculusTeleopConfig"]
