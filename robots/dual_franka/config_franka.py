"""
Configuration for the ROS2 dual-Franka + Robotiq robot.

The LeRobot process talks to a ZeroRPC server. The server owns the ROS2 node
and the exp_env_interact DualFrankaRobotiqEnv instance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("franka_dual_arm")
@dataclass
class FrankaDualArmConfig(RobotConfig):
    """Configuration for dual Franka arms with two Robotiq 2F-85 grippers."""

    name: str = "franka_dual_arm"

    # Unified ZeroRPC endpoint. The server controls both arms and both grippers.
    robot_ip: str = "127.0.0.1"
    robot_port: int = 4242
    rpc_timeout_sec: float = 30.0

    # Compatibility fields used by the generic recording config parser.
    gripper_ip: str = "127.0.0.1"
    gripper_port: int = 4242

    use_gripper: bool = True
    gripper_max_open: float = 0.085
    gripper_force: float = 10.0
    gripper_speed: float = 0.1
    gripper_reverse: bool = False
    close_threshold: float = 0.5
    open_grippers_on_connect: bool = False
    reset_opens_grippers: bool = True
    reset_go_home: bool = True
    go_home_duration_sec: float | None = None
    go_home_rate_hz: float | None = None

    control_mode: str = "oculus"
    debug: bool = True

    num_joints_per_arm: int = 7
    max_joint_velocity: float = 2.0
    max_ee_velocity: float = 0.5
    max_joint_delta: float = 0.3
    # None disables wrapper-side action clipping. Keep controller/server-side
    # safety limits separate so recorded actions match what this wrapper sends.
    max_cartesian_delta: float | None = None
    max_rotation_delta: float | None = None

    cameras: dict[str, CameraConfig] = field(default_factory=dict)
