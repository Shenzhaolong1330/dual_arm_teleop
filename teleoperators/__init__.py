"""
Teleoperation module for dual-arm system.
Provides unified VR (Oculus) teleoperation interface for all robots.

Output format (unified dual-arm):
- left_delta_ee_pose.{x,y,z,rx,ry,rz}: Left arm delta pose
- right_delta_ee_pose.{x,y,z,rx,ry,rz}: Right arm delta pose
- left_gripper_width: Left gripper target aperture in metres
- right_gripper_width: Right gripper target aperture in metres

For single-arm robots: use right_* data only
For dual-arm robots: use both left_* and right_* data
"""

from .oculus_teleoperator import (
    OculusTeleopConfig,
    OculusTeleop,
)

__all__ = [
    "OculusTeleopConfig",
    "OculusTeleop",
]
