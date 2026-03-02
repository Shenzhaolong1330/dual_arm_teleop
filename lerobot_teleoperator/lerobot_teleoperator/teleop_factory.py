#!/usr/bin/env python

"""
Factory for creating teleoperation instances for Dobot dual-arm system.
Uses Oculus Quest for teleoperation control.
"""

from typing import Union

from .base_teleop import BaseTeleop
from .config_teleop import (
    BaseTeleopConfig,
    OculusTeleopConfig,
    OculusDualArmTeleopConfig,
)
from .oculus_teleop import OculusTeleop
from .oculus_dual_arm_teleop import OculusDualArmTeleop


def create_teleop(config: BaseTeleopConfig) -> BaseTeleop:
    """
    Create a teleoperation instance based on the configuration.
    
    Args:
        config: Teleoperation configuration
    
    Returns:
        A teleoperation instance
    
    Raises:
        ValueError: If the control mode is not supported
    """
    if isinstance(config, OculusDualArmTeleopConfig):
        return OculusDualArmTeleop(config)
    
    elif isinstance(config, OculusTeleopConfig) or config.control_mode == "oculus":
        return OculusTeleop(config if isinstance(config, OculusTeleopConfig) else OculusTeleopConfig())
    
    else:
        raise ValueError(f"Unsupported control mode: {config.control_mode}. "
                        f"Supported mode: oculus")


def create_teleop_config(control_mode: str, dual_arm: bool = False, **kwargs) -> BaseTeleopConfig:
    """
    Create a teleoperation configuration based on the control mode.
    
    Args:
        control_mode: The teleoperation mode ("oculus")
        dual_arm: Whether to create a dual-arm configuration
        **kwargs: Configuration parameters specific to each mode
    
    Returns:
        A teleoperation configuration instance
    
    Raises:
        ValueError: If the control mode is not supported
    """
    if control_mode == "oculus":
        if dual_arm:
            return OculusDualArmTeleopConfig(**kwargs)
        return OculusTeleopConfig(**kwargs)
    else:
        raise ValueError(f"Unsupported control mode: {control_mode}. "
                        f"Supported mode: oculus")


def get_action_features(control_mode: str, use_gripper: bool = True, dual_arm: bool = False) -> dict:
    """
    Get the action features for a given control mode.
    
    Args:
        control_mode: The teleoperation mode ("oculus")
        use_gripper: Whether gripper is used
        dual_arm: Whether to return features for dual-arm system
    
    Returns:
        Dictionary of action features
    """
    if dual_arm:
        return get_dual_arm_action_features(control_mode, use_gripper)
    
    if control_mode == "oculus":
        features = {}
        for axis in ["x", "y", "z", "rx", "ry", "rz"]:
            features[f"delta_ee_pose.{axis}"] = float
        if use_gripper:
            features["gripper_cmd_bin"] = float
        return features
    
    else:
        raise ValueError(f"Unsupported control mode: {control_mode}")


def get_dual_arm_action_features(control_mode: str, use_gripper: bool = True) -> dict:
    """
    Get the action features for dual-arm teleoperation.
    
    Args:
        control_mode: The teleoperation mode ("oculus")
        use_gripper: Whether grippers are used
    
    Returns:
        Dictionary of action features for dual-arm system
    """
    features = {}
    
    if control_mode == "oculus":
        # Left arm delta pose
        for axis in ["x", "y", "z", "rx", "ry", "rz"]:
            features[f"left_delta_ee_pose.{axis}"] = float
        # Right arm delta pose
        for axis in ["x", "y", "z", "rx", "ry", "rz"]:
            features[f"right_delta_ee_pose.{axis}"] = float
        if use_gripper:
            features["left_gripper_cmd_bin"] = float
            features["right_gripper_cmd_bin"] = float
    
    else:
        raise ValueError(f"Unsupported control mode: {control_mode}")
    
    return features
