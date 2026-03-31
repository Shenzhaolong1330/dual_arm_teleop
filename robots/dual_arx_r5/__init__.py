"""
Arx dual-arm robot module.
Provides robot interface for Arx dual-arm robot.
"""

from .config_arx import ArxDualArmConfig
from .arx_dual_arm import ArxDualArm

__all__ = [
    "ArxDualArmConfig",
    "ArxDualArm",
]
