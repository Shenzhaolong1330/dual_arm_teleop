"""
Franka dual-arm robot module.
Provides robot interface for Franka dual-arm robot.
"""

from .config_franka import FrankaDualArmConfig
from .franka_dual_arm import FrankaDualArm

__all__ = [
    "FrankaDualArmConfig",
    "FrankaDualArm",
]
