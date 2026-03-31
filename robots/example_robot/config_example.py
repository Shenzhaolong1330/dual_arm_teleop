"""
Example robot configuration template.

Configuration class defines all parameters needed to control the robot:
- Network settings (IP, port)
- Gripper settings (max open width, force, speed)
- Control mode settings
- Safety limits

Implementation:
- Inherit from RobotConfig and use @dataclass decorator
- Register as a subclass using @RobotConfig.register_subclass("robot_name")
- Define all configuration parameters with default values
- Use type hints for each parameter
"""
from dataclasses import dataclass, field
from typing import Optional

from lerobot.cameras import CameraConfig
from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("example_robot")
@dataclass
class ExampleRobotConfig(RobotConfig):
    """Configuration for Example Robot.
    
    TODO: Replace with your robot's actual configuration parameters.
    """
    
    # ===== Robot Identification =====
    name: str = "example_robot"
    
    # ===== Network Configuration =====
    robot_ip: str = "192.168.1.100"  # Robot server IP address
    robot_port: int = 4242  # Robot server port (for zerorpc communication)
    
    # ===== Gripper Configuration =====
    use_gripper: bool = True
    gripper_ip: str = "localhost"  # Gripper server IP (if separate from robot)
    gripper_port: int = 4243  # Gripper server port
    gripper_max_open: float = 0.085  # Maximum gripper opening in meters
    gripper_force: float = 10.0  # Gripping force in Newtons
    gripper_speed: float = 0.1  # Gripper speed in m/s
    gripper_reverse: bool = False  # Whether to reverse gripper open/close
    close_threshold: float = 0.5  # Threshold for binary gripper control (0-1)
    gripper_bin_threshold: float = 0.98  # Threshold to detect if gripper is closed
    
    # ===== Control Configuration =====
    control_mode: str = "oculus"  # Control mode: "oculus", "spacemouse", "isoteleop"
    execute_mode: str = "ee_pose"  # Execute mode: "ee_pose" (cartesian) or "joint" (via IK)
    debug: bool = True  # If True, robot won't execute actions (dry run)
    
    # ===== Robot Kinematics =====
    num_joints: int = 7  # Number of joints per arm
    
    # ===== Safety Limits =====
    max_joint_velocity: float = 2.0  # Maximum joint velocity in rad/s
    max_ee_velocity: float = 0.5  # Maximum end-effector velocity in m/s
    max_joint_delta: float = 0.3  # Maximum joint change per step in rad
    
    # ===== Cameras =====
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

