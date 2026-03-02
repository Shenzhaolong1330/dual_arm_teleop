#!/usr/bin/env python

"""
Oculus Quest dual-arm teleoperation implementation.
Uses both Oculus controllers to control a dual-arm robot system.
Left controller -> Left arm, Right controller -> Right arm.
"""

import logging
from typing import Any, Dict

from .base_teleop import BaseTeleop
from .config_teleop import OculusDualArmTeleopConfig
from .oculus.oculus_dual_arm_robot import OculusDualArmRobot

logger = logging.getLogger(__name__)


class OculusDualArmTeleop(BaseTeleop):
    """
    Dual-arm teleoperation using both Oculus Quest controllers.
    
    This teleoperation mode uses both Oculus Quest controllers to simultaneously
    control two robot arms in Cartesian space. The output is delta pose for each arm.
    
    Controls:
    - LG (Left Grip):    Must be pressed to enable left arm action recording
    - RG (Right Grip):   Must be pressed to enable right arm action recording
    - LTr (Left Trigger):  Controls left gripper  (0.0 = open, 1.0 = closed)
    - RTr (Right Trigger): Controls right gripper (0.0 = open, 1.0 = closed)
    - Left controller pose:  Controls left arm end-effector delta pose
    - Right controller pose: Controls right arm end-effector delta pose
    - A button: Request robot reset
    """
    
    config_class = OculusDualArmTeleopConfig
    name = "OculusDualArmTeleop"
    
    def __init__(self, config: OculusDualArmTeleopConfig):
        super().__init__(config)
        self.oculus_robot: OculusDualArmRobot = None
    
    def _get_teleop_name(self) -> str:
        return "OculusDualArmTeleop"
    
    @property
    def action_features(self) -> dict:
        """Return action features for dual-arm oculus mode."""
        features = {}
        # Left arm delta pose
        for axis in ["x", "y", "z", "rx", "ry", "rz"]:
            features[f"left_delta_ee_pose.{axis}"] = float
        # Right arm delta pose
        for axis in ["x", "y", "z", "rx", "ry", "rz"]:
            features[f"right_delta_ee_pose.{axis}"] = float
        # Gripper commands
        if self.cfg.use_gripper:
            features["left_gripper_cmd_bin"] = float
            features["right_gripper_cmd_bin"] = float
        return features
    
    def _connect_impl(self) -> None:
        """Connect to Oculus Quest for dual-arm control."""
        self.oculus_robot = OculusDualArmRobot(
            ip=self.cfg.ip,
            use_gripper=self.cfg.use_gripper,
            left_pose_scaler=self.cfg.left_pose_scaler,
            left_channel_signs=self.cfg.left_channel_signs,
            right_pose_scaler=self.cfg.right_pose_scaler,
            right_channel_signs=self.cfg.right_channel_signs,
        )
        logger.info(f"[TELEOP] Oculus dual-arm connected at IP: {self.cfg.ip}")
        logger.info(f"[TELEOP]   Left arm  - pose_scaler: {self.cfg.left_pose_scaler}, "
                    f"channel_signs: {self.cfg.left_channel_signs}")
        logger.info(f"[TELEOP]   Right arm - pose_scaler: {self.cfg.right_pose_scaler}, "
                    f"channel_signs: {self.cfg.right_channel_signs}")
    
    def _disconnect_impl(self) -> None:
        """Disconnect from Oculus Quest."""
        # OculusDualArmRobot doesn't have explicit disconnect
        pass
    
    def _get_action_impl(self) -> Dict[str, Any]:
        """Get delta pose from both Oculus controllers."""
        return self.oculus_robot.get_observations()
