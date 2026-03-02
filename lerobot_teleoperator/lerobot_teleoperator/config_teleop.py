"""
Teleoperation configuration for Dobot dual-arm system.
Uses Oculus Quest for dual-arm teleoperation control.
"""
from dataclasses import dataclass, field
from typing import List

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("dobot_teleop")
@dataclass
class BaseTeleopConfig(TeleoperatorConfig):
    """Base configuration for all teleoperation modes."""
    control_mode: str = "oculus"
    use_gripper: bool = True
    dual_arm: bool = False


@TeleoperatorConfig.register_subclass("oculus_teleop")
@dataclass
class OculusTeleopConfig(BaseTeleopConfig):
    """Configuration for single-arm Oculus Quest teleoperation."""
    control_mode: str = "oculus"
    ip: str = "192.168.110.62"
    pose_scaler: List[float] = field(default_factory=lambda: [1.0, 1.0])  # [position_scale, orientation_scale]
    channel_signs: List[int] = field(default_factory=lambda: [1, 1, 1, 1, 1, 1])  # [x, y, z, rx, ry, rz]


# ==================== Dual-Arm Teleoperation Configuration ====================

@TeleoperatorConfig.register_subclass("oculus_dual_arm_teleop")
@dataclass
class OculusDualArmTeleopConfig(BaseTeleopConfig):
    """
    Configuration for dual-arm Oculus Quest teleoperation.
    Uses both Oculus controllers to control both arms simultaneously.
    Left controller -> Left arm, Right controller -> Right arm.
    """
    control_mode: str = "oculus"
    dual_arm: bool = True
    ip: str = "192.168.110.62"
    
    # Left controller (controls left arm)
    left_pose_scaler: List[float] = field(default_factory=lambda: [1.0, 1.0])
    left_channel_signs: List[int] = field(default_factory=lambda: [1, 1, 1, 1, 1, 1])
    
    # Right controller (controls right arm)
    right_pose_scaler: List[float] = field(default_factory=lambda: [1.0, 1.0])
    right_channel_signs: List[int] = field(default_factory=lambda: [1, 1, 1, 1, 1, 1])
    
    # Gripper control
    use_gripper: bool = True
    # Left gripper: Left Trigger (LTr)
    # Right gripper: Right Trigger (RTr)
