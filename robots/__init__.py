"""
Robot module for dual-arm teleoperation system.
Supports multiple robot types: Franka (single-arm), Dobot (dual-arm), etc.

Action input format (from teleop):
- Single-arm robot: uses right_delta_ee_pose.{axis}, right_gripper_cmd_bin
- Dual-arm robot: uses left/right_delta_ee_pose.{axis}, left/right_gripper_cmd_bin
"""

from typing import Dict, Any, Type

# Import robot configurations
from .franka.config_franka import FrankaConfig
from .dual_dobot.config_dobot import DobotDualArmConfig
from .dual_agilx_nero.config_nero import NeroDualArmConfig

# Import robot classes
from .franka.franka import Franka
from .dual_dobot.dobot_dual_arm import DobotDualArm
from .dual_agilx_nero.nero_dual_arm import NeroDualArm


# Robot type registry: {robot_type: (ConfigClass, RobotClass)}
ROBOT_CONFIG_REGISTRY: Dict[str, tuple] = {
    # Single-arm robots
    "franka": (FrankaConfig, Franka),
    # Dual-arm robots
    "dobot_dual_arm": (DobotDualArmConfig, DobotDualArm),
    "nero_dual_arm": (NeroDualArmConfig, NeroDualArm)
}

# Supported robot types
SUPPORTED_ROBOTS = list(ROBOT_CONFIG_REGISTRY.keys())


def get_robot_config_class(robot_type: str) -> Type:
    """Get the robot configuration class by robot type."""
    if robot_type not in ROBOT_CONFIG_REGISTRY:
        raise ValueError(
            f"Unsupported robot type: {robot_type}. "
            f"Supported types: {SUPPORTED_ROBOTS}"
        )
    return ROBOT_CONFIG_REGISTRY[robot_type][0]


def get_robot_class(robot_type: str) -> Type:
    """Get the robot class by robot type."""
    if robot_type not in ROBOT_CONFIG_REGISTRY:
        raise ValueError(
            f"Unsupported robot type: {robot_type}. "
            f"Supported types: {SUPPORTED_ROBOTS}"
        )
    return ROBOT_CONFIG_REGISTRY[robot_type][1]


def create_robot_config(robot_type: str, **kwargs) -> Any:
    """Create a robot configuration instance."""
    config_class = get_robot_config_class(robot_type)
    return config_class(**kwargs)


def create_robot(robot_type: str, config: Any):
    """Create a robot instance."""
    robot_class = get_robot_class(robot_type)
    return robot_class(config)


__all__ = [
    # Configuration classes
    "FrankaConfig",
    "DobotDualArmConfig",
    # Robot classes
    "Franka",
    "DobotDualArm",
    # Registry and factory functions
    "ROBOT_CONFIG_REGISTRY",
    "SUPPORTED_ROBOTS",
    "get_robot_config_class",
    "get_robot_class",
    "create_robot_config",
    "create_robot",
]
