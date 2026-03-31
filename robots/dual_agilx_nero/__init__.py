"""
Nero dual-arm robot module.
Provides robot interface for Nero dual-arm robot.
"""

from .config_nero import NeroDualArmConfig
from .nero_dual_arm import NeroDualArm

__all__ = [
    "NeroDualArmConfig",
    "NeroDualArm",
]
