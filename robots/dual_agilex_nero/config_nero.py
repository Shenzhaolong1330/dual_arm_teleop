"""
Configuration for Nero dual-arm robot system.
Each arm has 7 DOF
"""
from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("nero_dual_arm")
@dataclass
class NeroDualArmConfig(RobotConfig):
    """Configuration for Nero dual-arm robot with agx_grippers."""
    
    # Robot identification
    name: str = "nero_dual_arm"
    
    # Network configuration - single port for dual-arm control
    robot_ip: str = "192.168.110.114"  # Nero server ip
    robot_port: int = 4242  # dual-arm zerorpc port (single port for both arms)
    
    # Gripper configuration (agx_grippers)
    gripper_ip: str = "192.168.110.114"  # gripper zerorpc ip, if different from robot_ip, set to robot_ip
    gripper_port: int = 4243  # gripper zerorpc port (single port for both arms)
    use_gripper: bool = True
    gripper_min_width: float = 0.0
    gripper_max_open: float = 0.095  # first-generation Pika Gripper nominal aperture: 95 mm
    gripper_force: float = 2.0  # Gripping force in N
    gripper_speed: float = 0.1  # Speed in m/s
    gripper_reverse: bool = False  # Whether to reverse gripper command
    close_threshold: float = 0.05  # Threshold for binary gripper control
    gripper_command_epsilon: float = 0.0005
    
    # Control configuration
    control_mode: str = "oculus"
    debug: bool = True

    # Public observations/actions use the same frame. World mode requires
    # explicit T_world_base=[x,y,z,qw,qx,qy,qz] for both arms.
    observation_pose_frame: str = "base"  # base | world
    left_base_pose_in_world: list[float] | None = None
    right_base_pose_in_world: list[float] | None = None
    # v2 is mandatory for effective target labels. Legacy mode is an explicit
    # diagnostic option and is marked incompatible with aligned datasets.
    cartesian_protocol: str = "v2"
    expected_server_tcp_offset_rpy: list[float] = field(default_factory=lambda: [0.0] * 6)
    left_tcp_in_native: list[float] = field(default_factory=lambda: [0, 0, 0, 1, 0, 0, 0])
    right_tcp_in_native: list[float] = field(default_factory=lambda: [0, 0, 0, 1, 0, 0, 0])
    physical_calibration: dict = field(default_factory=dict)
    
    # Joint configuration (7 DOF per arm)
    num_joints_per_arm: int = 7
    
    # Cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    
    # Safety limits
    max_joint_velocity: float = 2.0  # rad/s
    max_ee_velocity: float = 0.5  # m/s
    max_joint_delta: float = 0.3  # rad - max joint change per step
