"""
Dobot Nova5 dual-arm robot module.
Provides robot interface for Dobot Nova5 with Robotiq 2F-85 grippers.
"""

# Dobot configurations
from .config_dobot import DobotDualArmConfig, DobotSingleArmConfig

# Dobot robot classes
from .dobot_dual_arm import DobotSingleArm, DobotDualArm

# Dobot interface clients
from .dobot_interface_client import DobotArmClient, DobotDualArmClient, DobotDualInterfaceClient


__all__ = [
    # Dobot
    "DobotDualArmConfig",
    "DobotDualArm",
    "DobotDualArmClient",
]
