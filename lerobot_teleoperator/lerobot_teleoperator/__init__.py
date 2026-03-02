"""
Teleoperation module for Dobot dual-arm system.
Uses Oculus Quest for teleoperation control.
"""

# Configuration classes
from .config_teleop import (
    BaseTeleopConfig,
    OculusTeleopConfig,
    OculusDualArmTeleopConfig,
)

# Base class
from .base_teleop import BaseTeleop

# Teleoperation implementations
from .oculus_teleop import OculusTeleop
from .oculus_dual_arm_teleop import OculusDualArmTeleop

# Factory functions
from .teleop_factory import create_teleop, create_teleop_config, get_action_features


__all__ = [
    # Configuration classes
    "BaseTeleopConfig",
    "OculusTeleopConfig",
    "OculusDualArmTeleopConfig",
    # Base class
    "BaseTeleop",
    # Teleoperation implementations
    "OculusTeleop",
    "OculusDualArmTeleop",
    # Factory functions
    "create_teleop",
    "create_teleop_config",
    "get_action_features",
]
